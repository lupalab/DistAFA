import os
import logging
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import roc_auc_score

from datasets import get_dataset, get_ood_datasets, data_transform, inverse_data_transform
from datasets.masking import get_mask
from models import get_model, get_sigmas
from losses import get_optimizer
from models.ema import EMAHelper
from dose import get_dose

__all__ = ['OODRunner']

class OODRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        
        mask_fn = get_mask(self.config)
        
        # load pretrained scorenet
        if self.config.ood_scoring.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.ood_scoring.ckpt_id}.pth'),
                                map_location=self.config.device)
            
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        score.load_state_dict(states[0], strict=True)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            
        score.eval()
        
        sigmas = get_sigmas(self.config)
        
        # build dosenet
        dose = get_dose(self.config)
        optimizer = get_optimizer(self.config, dose.parameters())
        
        start_epoch = 0
        step = 0
        
        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                dose.train()
                step += 1
                
                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                
                mask = mask_fn(X)
                
                states = self.get_states(score, X, mask, sigmas)
                logp = dose(states, mask)
                loss = -logp.mean()
                
                logging.info("step: {}, loss: {}".format(step, loss.item()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    dose.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)
                    
                    test_mask = mask_fn(test_X)
                    
                    test_states = self.get_states(score, test_X, test_mask, sigmas)
                    with torch.no_grad():
                        test_logp = dose(test_states, test_mask)
                        test_loss = -test_logp.mean()
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))
                        
                        del test_logp
                        
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        dose.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    torch.save(states, os.path.join(self.args.log_path, 'ood_checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'ood_checkpoint.pth'))
        
    def test(self):
        # load pretrained scorenet
        if self.config.ood_scoring.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.ood_scoring.ckpt_id}.pth'),
                                map_location=self.config.device)
            
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        score.load_state_dict(states[0], strict=True)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            
        score.eval()
        
        sigmas = get_sigmas(self.config)
        
        # load pretrained dosenet
        if self.config.ood_scoring.ood_ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'ood_checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'ood_checkpoint_{self.config.ood_scoring.ood_ckpt_id}.pth'),
                                map_location=self.config.device)
            
        dose = get_dose(self.config)
        dose.load_state_dict(states[0], strict=True)
        dose.eval()
        
        # datasets
        dataset, test_dataset = get_dataset(self.args, self.config)
        ood_datasets, ood_test_datasets = get_ood_datasets(self.args, self.config)
        ood_datasets['in_distribution'] = dataset
        ood_test_datasets['in_distribution'] = test_dataset
        
        def make_mask_fn(N):
            num_pixels = self.config.data.image_size ** 2
            rng = np.random.RandomState(42)
            idx = rng.choice(num_pixels, N, replace=False)
            def mask_fn(batch):
                gen_shape = list(batch.shape)
                gen_shape[1] = 1
                mask = torch.zeros(gen_shape).to(batch).view([gen_shape[0], -1])
                mask[:, idx] = 1.0
                mask = mask.view(gen_shape)
                return mask
            return mask_fn
        
        for n in [self.config.data.image_size ** 2]:
            mask_fn = make_mask_fn(n)
            dataset_logp_dict = {}
            for dataset_name, dataset in ood_test_datasets.items():
                dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)
                dataset_logp = []
                for i, (X, y) in enumerate(dataloader):
                    if i >= self.config.ood_scoring.num_batches4test: break
                    X = X.to(self.config.device)
                    X = data_transform(self.config, X)
                    
                    mask = mask_fn(X)
                    
                    states = self.get_states(score, X, mask, sigmas)
                    with torch.no_grad():
                        logp = dose(states, mask)
                    dataset_logp.append(logp.data.cpu().numpy())
                dataset_logp = np.concatenate(dataset_logp, axis=0)
                dataset_logp_dict[dataset_name] = dataset_logp
                
            # save
            with open(os.path.join(self.args.image_folder, f'dataset_logp_dict_{n}.pkl'), 'wb') as f:
                pickle.dump(dataset_logp_dict, f)
                
            # plot
            fig, ax = plt.subplots()
            sns.histplot(dataset_logp_dict, legend=True, kde=True, ax=ax)
            plt.savefig(os.path.join(self.args.image_folder, f'hist_{n}.png'))
            plt.close(fig=fig)
            
    def search(self):
        # load pretrained scorenet
        if self.config.ood_scoring.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.ood_scoring.ckpt_id}.pth'),
                                map_location=self.config.device)
            
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        score.load_state_dict(states[0], strict=True)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            
        score.eval()
            
        sigmas = get_sigmas(self.config)
        
        # load pretrained dosenet
        if self.config.ood_scoring.ood_ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'ood_checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'ood_checkpoint_{self.config.ood_scoring.ood_ckpt_id}.pth'),
                                map_location=self.config.device)
            
        dose = get_dose(self.config)
        dose.load_state_dict(states[0], strict=True)
        dose.eval()
        
        # datasets
        dataset, test_dataset = get_dataset(self.args, self.config)
        ood_datasets, ood_test_datasets = get_ood_datasets(self.args, self.config)
        ood_datasets['in_distribution'] = dataset
        ood_test_datasets['in_distribution'] = test_dataset
        
        datasets = {}
        datasets['MNIST'] = ood_test_datasets['in_distribution']
        datasets['KMNIST'] = ood_test_datasets['KMNIST']
        
        def make_mask_fn(N, seed):
            num_pixels = self.config.data.image_size ** 2
            rng = np.random.RandomState(seed)
            idx = rng.choice(num_pixels, N, replace=False)
            def mask_fn(batch):
                gen_shape = list(batch.shape)
                gen_shape[1] = 1
                mask = torch.zeros(gen_shape).to(batch).view([gen_shape[0], -1])
                mask[:, idx] = 1.0
                mask = mask.view(gen_shape)
                return mask
            return mask_fn
        
        avg_logp = []
        auroc = []
        results = {}
        for search_id in range(50):
            mask_fn = make_mask_fn(100, search_id)
            dataset_logp_dict = {}
            for dataset_name, dataset in datasets.items():
                dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)
                dataset_logp = []
                for i, (X, y) in enumerate(dataloader):
                    if i >= self.config.ood_scoring.num_batches4test: break
                    X = X.to(self.config.device)
                    X = data_transform(self.config, X)
                    
                    mask = mask_fn(X)
                    
                    states = self.get_states(score, X, mask, sigmas)
                    with torch.no_grad():
                        logp = dose(states, mask)
                    dataset_logp.append(logp.data.cpu().numpy())
                dataset_logp = np.concatenate(dataset_logp, axis=0)
                dataset_logp_dict[dataset_name] = dataset_logp
            neg_logp = np.concatenate([dataset_logp_dict['MNIST'], dataset_logp_dict['KMNIST']]) * -1.
            label = np.concatenate([np.zeros_like(dataset_logp_dict['MNIST']), np.ones_like(dataset_logp_dict['KMNIST'])])
            results[search_id] = (neg_logp, label)
            
            roc_auc = roc_auc_score(label, neg_logp)
            auroc.append(roc_auc)
            avg_logp.append(np.mean(dataset_logp_dict['MNIST']))
            
            logging.info(f"#{search_id}: AUROC: {roc_auc} Logp: {np.mean(dataset_logp_dict['MNIST'])}")
            
        # save
        with open(os.path.join(self.args.image_folder, f'auroc_100.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # plot
        fig = plt.figure()
        plt.hist(auroc, bins=10)
        plt.title('AUROC: MNIST / KMNIST')
        plt.savefig(os.path.join(self.args.image_folder, f'auroc_100.png'))
        plt.close(fig=fig)
        
        fig = plt.figure()
        plt.scatter(avg_logp, auroc)
        plt.xlabel('logp')
        plt.ylabel('auroc')
        plt.savefig(os.path.join(self.args.image_folder, f'logp_auroc.png'))
        plt.close(fig=fig)
        
    def plot(self):
        if self.config.ood_scoring.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.ood_scoring.ckpt_id}.pth'),
                                map_location=self.config.device)
            
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)
        
        score.load_state_dict(states[0], strict=True)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)
            
        score.eval()
            
        sigmas = get_sigmas(self.config)
        
        dataset, test_dataset = get_dataset(self.args, self.config)
        ood_datasets, ood_test_datasets = get_ood_datasets(self.args, self.config)
        ood_datasets['in_distribution'] = dataset
        ood_test_datasets['in_distribution'] = test_dataset
        
        def make_mask_fn(N):
            num_pixels = self.config.data.image_size ** 2
            rng = np.random.RandomState(42)
            idx = rng.choice(num_pixels, N, replace=False)
            def mask_fn(batch):
                gen_shape = list(batch.shape)
                gen_shape[1] = 1
                mask = torch.zeros(gen_shape).to(batch).view([gen_shape[0], -1])
                mask[:, idx] = 1.0
                mask = mask.view(gen_shape)
                return mask
            return mask_fn
        
        for n in [5, 10, 20, 50, 100, self.config.data.image_size ** 2]:
            mask_fn = make_mask_fn(n)
            dataset_state_dict = {}
            for dataset_name, dataset in ood_test_datasets.items():
                dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=self.config.data.num_workers, drop_last=True)
                dataset_states = []
                for i, (X, y) in enumerate(dataloader):
                    if i >= self.config.ood_scoring.num_batches4plot: break
                    X = X.to(self.config.device)
                    X = data_transform(self.config, X)
                    
                    mask = mask_fn(X)
                    
                    states = self.get_states(score, X, mask, sigmas)
                    dataset_states.append(states.cpu().numpy())
                dataset_states = np.concatenate(dataset_states, axis=0)
                dataset_state_dict[dataset_name] = dataset_states
            
            # save
            with open(os.path.join(self.args.image_folder, f'dataset_state_dict_{n}.pkl'), 'wb') as f:
                pickle.dump(dataset_state_dict, f)
            
            # plot
            data = np.concatenate(list(dataset_state_dict.values()))
            label = [np.ones([v.shape[0]]) * i for i, v in enumerate(dataset_state_dict.values())]
            label = np.concatenate(label).astype(np.int64)
            label_str = list(dataset_state_dict.keys())
            color = 'bckgmry'
            ## tsne
            embed = TSNE().fit_transform(data)
            fig = plt.figure()
            for lab in np.unique(label):
                f = embed[label==lab]
                plt.scatter(f[:,0], f[:,1], s=5, c=color[lab], label=label_str[lab])
            plt.legend()
            plt.savefig(os.path.join(self.args.image_folder, f'tsne_{n}.png'))
            plt.close(fig=fig)
            ## umap
            embed = UMAP().fit_transform(data)
            fig = plt.figure()
            for lab in np.unique(label):
                f = embed[label==lab]
                plt.scatter(f[:,0], f[:,1], s=5, c=color[lab], label=label_str[lab])
            plt.legend()
            plt.savefig(os.path.join(self.args.image_folder, f'umap_{n}.png'))
            plt.close(fig=fig)
    
    @torch.no_grad()
    def get_states(self, scorenet, x, m, sigmas):
        states = []
        with torch.no_grad():
            for i, sig in enumerate(sigmas):
                y = torch.ones([x.shape[0]], device=x.device).long() * i
                score = scorenet(x*m, m, y)
                score = score * m * sig
                state = torch.norm(score.view(x.shape[0], -1), dim=-1)
                states.append(state)
        states = torch.stack(states, dim=-1).data
        return states
            
        