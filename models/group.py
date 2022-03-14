import os
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tensorflow as tf
from pprint import pformat
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import FeatureAgglomeration, SpectralClustering
from sklearn.covariance import GraphicalLassoCV

from utils.hparams import HParams
from models import get_model
from datasets import get_dataset

class GraphGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        model_cfg.batch_size = args.batch_size
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        self.num_batches = min(args.num_batches, self.dataset.num_batches)
        
    def __call__(self):
        # collect data
        X = []
        for _ in range(self.num_batches):
            batch = self.dataset.next_batch()
            X.append(batch['x'])
        X = np.concatenate(X, axis=0)
        X = X.reshape([len(X), -1]) / 255.
        X += np.random.randn(*X.shape) / 255.
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        # graph lasso
        cov = GraphicalLassoCV().fit(X)
        with open(f'{self.args.exp_dir}/cov.pkl', 'wb') as f:
            pickle.dump(cov, f)
        # spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.args.num_clusters,
            affinity='precomputed',
            assign_labels='discretize').fit(np.abs(cov.covariance_))
        with open(f'{self.args.exp_dir}/clustering.pkl', 'wb') as f:
            pickle.dump(clustering, f)
        # labels
        labels = clustering.labels_
        with open(f'{self.args.exp_dir}/labels.pkl', 'wb') as f:
            pickle.dump({'labels':labels}, f)
        # balance
        num_pixels = np.prod(self.dataset.image_shape[:2])
        pixels_per_cluster = num_pixels // self.args.num_clusters
        group2idx = defaultdict(list)
        for idx, lab in enumerate(labels):
            group2idx[lab].append(idx)
        group_size = Counter(labels)
        seq_idx = []
        for lab, size in group_size.items():
            seq_idx.extend(group2idx[lab])
        groups = np.arange(num_pixels) // pixels_per_cluster
        labels = np.zeros(num_pixels, np.int64)
        labels[seq_idx] = groups
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
            
        for i in range(self.args.num_clusters):
            mask = (labels == i).reshape(self.dataset.image_shape[:2]).astype(np.float32)
            plt.imsave(f'{self.args.exp_dir}/label_{i}.png', mask)
        

class ImgGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        
    def __call__(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        pixels_per_cluster = num_pixels // self.args.num_clusters
        labels = (np.arange(num_pixels) // pixels_per_cluster).astype(np.int64)
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
            
        for i in range(self.args.num_clusters):
            mask = (labels == i).reshape(self.dataset.image_shape[:2]).astype(np.float32)
            plt.imsave(f'{self.args.exp_dir}/label_{i}.png', mask)
            

class RandGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        
    def __call__(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        pixels_per_cluster = num_pixels // self.args.num_clusters
        labels = (np.arange(num_pixels) // pixels_per_cluster).astype(np.int64)
        np.random.shuffle(labels)
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
            
        for i in range(self.args.num_clusters):
            mask = (labels == i).reshape(self.dataset.image_shape[:2]).astype(np.float32)
            plt.imsave(f'{self.args.exp_dir}/label_{i}.png', mask)
        

class MIGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        model_cfg.batch_size = args.batch_size
        self.model = get_model(model_cfg)
        self.model.load()
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        self.num_batches = min(args.num_batches, self.dataset.num_batches)
        
    def _make_mask(self, i, j):
        mask = np.zeros([self.args.batch_size]+self.dataset.image_shape, dtype=np.uint8)
        mask = np.reshape(mask, [self.args.batch_size,-1,self.dataset.image_shape[-1]])
        mask[:,i] = 1
        mask[:,j] = 1
        return np.reshape(mask, [self.args.batch_size]+self.dataset.image_shape)
        
    def _mutual_information(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        all_mutual_info = []
        for _ in range(self.num_batches):
            batch = self.dataset.next_batch()
            pi = []
            for i in range(num_pixels):
                b = np.zeros_like(batch['b'])
                m = self._make_mask(i, i)
                feed_dict = {self.model.x: batch['x'], self.model.b: b, self.model.m: m}
                logp = self.model.execute(self.model.logp, feed_dict)
                pi.append(logp)
            mutual_info = []
            for i in range(num_pixels):
                for j in range(num_pixels):
                    b = np.zeros_like(batch['b'])
                    m = self._make_mask(i, j)
                    feed_dict = {self.model.x: batch['x'], self.model.b: b, self.model.m: m}
                    logp = self.model.execute(self.model.logp, feed_dict)
                    mi = (logp - pi[i] - pi[j]).mean()
                    mutual_info.append(mi)
            all_mutual_info.append(mutual_info)
        mi = np.array(all_mutual_info).mean(axis=0)
        mi = mi.reshape([num_pixels, num_pixels])
        return mi
        
    def __call__(self):
        logging.info('started')
        mi = self._mutual_information()
        with open(f'{self.args.exp_dir}/mi.pkl', 'wb') as f:
            pickle.dump(mi, f)
        logging.info('mutual information computed')
        
        agglo = FeatureAgglomeration(n_clusters=self.args.num_clusters, affinity='precomputed', linkage='average')
        agglo = agglo.fit(-mi)
        labels = agglo.labels_
        logging.info('fitted')
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'mi':mi, 'agglo':agglo, 'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
 

class UpperMIGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        model_cfg.batch_size = args.batch_size
        self.model = get_model(model_cfg)
        self.model.load()
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        self.num_batches = min(args.num_batches, self.dataset.num_batches)
        
    def _make_mask(self, i):
        mask = np.zeros([self.args.batch_size]+self.dataset.image_shape, dtype=np.uint8)
        mask = np.reshape(mask, [self.args.batch_size,-1,self.dataset.image_shape[-1]])
        mask[:,i] = 1
        return np.reshape(mask, [self.args.batch_size]+self.dataset.image_shape)
        
    def _upper_mutual_information(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        all_pi, all_cpi = [], []
        for _ in range(self.num_batches):
            batch = self.dataset.next_batch()
            pi = []
            cpi = []
            for i in range(num_pixels):
                b = np.zeros_like(batch['b'])
                m = self._make_mask(i)
                feed_dict = {self.model.x: batch['x'], 
                             self.model.y: batch['y'], 
                             self.model.b: b, 
                             self.model.m: m}
                logp, cond_logp = self.model.execute([self.model.logp, self.model.cond_logp], feed_dict)
                pi.append(logp)
                cpi.append(cond_logp)
            pi = np.stack(pi, axis=1)
            cpi = np.stack(cpi, axis=1)
            all_pi.append(pi)
            all_cpi.append(cpi)
        all_pi = np.concatenate(all_pi, axis=0)
        all_cpi = np.concatenate(all_cpi, axis=0)
        Hi = np.mean(all_pi, axis=0)
        cHi = np.mean(all_cpi, axis=0)
        d = np.expand_dims(Hi, axis=0) + np.expand_dims(Hi, axis=1) + \
            2 * np.expand_dims(cHi, axis=0) + 2 * np.expand_dims(cHi, axis=1)
        return d
        
    def __call__(self):
        logging.info('start')
        dist = self._upper_mutual_information()
        with open(f'{self.args.exp_dir}/dist.pkl', 'wb') as f:
            pickle.dump(dist, f)
        logging.info('distance computed')
        
        agglo = FeatureAgglomeration(n_clusters=self.args.num_clusters, affinity='precomputed', linkage='average')
        agglo = agglo.fit(dist)
        labels = agglo.labels_
        logging.info('fitted')
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'dist':dist, 'agglo':agglo, 'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))

        
class ApproxGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        model_cfg.batch_size = args.batch_size
        self.model = get_model(model_cfg)
        self.model.load()
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        self.num_batches = min(args.num_batches, self.dataset.num_batches)
        
    def _make_mask(self, i):
        mask = np.zeros([self.args.batch_size]+self.dataset.image_shape, dtype=np.uint8)
        mask = np.reshape(mask, [self.args.batch_size,-1,self.dataset.image_shape[-1]])
        mask[:,i] = 1
        return np.reshape(mask, [self.args.batch_size]+self.dataset.image_shape)
        
    def _mutual_information(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        all_py = []
        for _ in range(self.num_batches):
            batch = self.dataset.next_batch()
            py = []
            for i in range(num_pixels):
                b = np.zeros_like(batch['b'])
                m = self._make_mask(i)
                feed_dict = {self.model.x: batch['x'], 
                             self.model.y: batch['y'], 
                             self.model.b: b, 
                             self.model.m: m}
                xent = self.model.execute(self.model.xent, feed_dict)
                py.append(xent)
            py = np.stack(py, axis=1)
            all_py.append(py)
        all_py = np.concatenate(all_py, axis=0)
        mi = np.mean(all_py, axis=0)
        return mi
    
    def __call__(self):
        logging.info('start')
        mi = self._mutual_information()
        with open(f'{self.args.exp_dir}/mi.pkl', 'wb') as f:
            pickle.dump(mi, f)
        logging.info('mutual information computed')
        
        num_pixels = np.prod(self.dataset.image_shape[:2])
        pixels_per_cluster = num_pixels // self.args.num_clusters
        idx = np.argsort(mi)
        groups = np.arange(num_pixels) // pixels_per_cluster
        labels = np.zeros(num_pixels, np.int64)
        labels[idx] = groups
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'mi':mi, 'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
            
        for i in range(self.args.num_clusters):
            mask = (labels == i).reshape(self.dataset.image_shape[:2]).astype(np.float32)
            plt.imsave(f'{self.args.exp_dir}/label_{i}.png', mask)


class EntGrouper(object):
    def __init__(self, args):
        self.args = args
        model_cfg = HParams(args.model_cfg_file)
        model_cfg.batch_size = args.batch_size
        self.model = get_model(model_cfg)
        self.model.load()
        self.dataset = get_dataset(model_cfg, 'valid')
        self.dataset.initialize()
        self.num_batches = min(args.num_batches, self.dataset.num_batches)

    def _make_mask(self, i):
        mask = np.zeros([self.args.batch_size]+self.dataset.image_shape, dtype=np.uint8)
        mask = np.reshape(mask, [self.args.batch_size,-1,self.dataset.image_shape[-1]])
        mask[:,i] = 1
        return np.reshape(mask, [self.args.batch_size]+self.dataset.image_shape)

    def _entropy(self):
        num_pixels = np.prod(self.dataset.image_shape[:2])
        all_pi = []
        for _ in range(self.num_batches):
            batch = self.dataset.next_batch()
            pi = []
            for i in range(num_pixels):
                b = np.zeros_like(batch['b'])
                m = self._make_mask(i)
                feed_dict = {self.model.x: batch['x'],
                             self.model.b: b, 
                             self.model.m: m}
                logp = self.model.execute(self.model.logp, feed_dict)
                pi.append(-logp)
            pi = np.stack(pi, axis=-1)
            all_pi.append(pi)
        all_pi = np.concatenate(all_pi, axis=0)
        ent = np.mean(all_pi, axis=0)
        return ent

    def __call__(self):
        logging.info('start')
        ent = self._entropy()
        with open(f'{self.args.exp_dir}/ent.pkl', 'wb') as f:
            pickle.dump(ent, f)
        logging.info('entropy computed')

        num_pixels = np.prod(self.dataset.image_shape[:2])
        pixels_per_cluster = num_pixels // self.args.num_clusters
        idx = np.argsort(ent)
        groups = np.arange(num_pixels) // pixels_per_cluster
        labels = np.zeros(num_pixels, np.int64)
        labels[idx] = groups
        
        with open(f'{self.args.exp_dir}/results.pkl', 'wb') as f:
            pickle.dump({'ent':ent, 'labels':labels}, f)
        
        with open(f'{self.args.exp_dir}/results.txt', 'w') as f:
            f.write('labels:\n')
            f.write(pformat(labels))
            f.write('counter:\n')
            f.write(pformat(Counter(labels)))
            
        for i in range(self.args.num_clusters):
            mask = (labels == i).reshape(self.dataset.image_shape[:2]).astype(np.float32)
            plt.imsave(f'{self.args.exp_dir}/label_{i}.png', mask)