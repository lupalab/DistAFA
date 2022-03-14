import os
import torch
from torch.utils.data.dataset import TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, LSUN, MNIST, KMNIST, SVHN, FashionMNIST, Omniglot
from .celeba import CelebA
from .ffhq import FFHQ
from torch.utils.data import Subset
import numpy as np
import pickle
import copy
import argparse

def get_ood_datasets(args, config):
    datasets, test_datasets = {}, {}
    for dst_cfg_dict in config.ood_data:
        name = dst_cfg_dict['dataset']
        new_config  = copy.deepcopy(config)
        new_config.data = argparse.Namespace()
        for key, value in dst_cfg_dict.items():
            setattr(new_config.data, key, value)
        dataset, test_dataset = get_dataset(args, new_config)
        datasets[name] = dataset
        test_datasets[name] = test_dataset
        
    return datasets, test_datasets

def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == 'MNIST':
        dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist'), train=True, download=True,
                        transform=tran_transform)
        test_dataset = MNIST(os.path.join(args.exp, 'datasets', 'mnist'), train=False, download=True,
                             transform=test_transform)
        
    elif config.data.dataset == 'FMNIST':
        dataset = FashionMNIST(os.path.join(args.exp, 'datasets', 'fashion_mnist'), train=True, download=True,
                        transform=tran_transform)
        test_dataset = FashionMNIST(os.path.join(args.exp, 'datasets', 'fashion_mnist'), train=False, download=True,
                             transform=test_transform)
        
    elif config.data.dataset == 'KMNIST':
        dataset = KMNIST(os.path.join(args.exp, 'datasets', 'kmnist'), train=True, download=True,
                        transform=tran_transform)
        test_dataset = KMNIST(os.path.join(args.exp, 'datasets', 'kmnist'), train=False, download=True,
                             transform=test_transform)
        
    elif config.data.dataset == 'Omniglot':
        dataset = Omniglot(os.path.join(args.exp, 'datasets', 'omniglot'), background=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(config.data.image_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: 1.0 - x)
                           ]))
        test_dataset = Omniglot(os.path.join(args.exp, 'datasets', 'omniglot'), background=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize(config.data.image_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: 1.0 - x)
                                ]))
        
    elif config.data.dataset == 'SVHN':
        dataset = SVHN(os.path.join(args.exp, 'datasets', 'svhn'), split='train', download=True,
                       transform=tran_transform)
        test_dataset = SVHN(os.path.join(args.exp, 'datasets', 'svhn'), split='test', download=True,
                            transform=test_transform)
        
    elif config.data.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=False, download=True,
                               transform=test_transform)
        
    elif config.data.dataset == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.exp, 'datasets', 'cifar100'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR100(os.path.join(args.exp, 'datasets', 'cifar100'), train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset == 'CELEBA':
        if config.data.random_flip:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                             ]), download=True)
        else:
            dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba'), split='train',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

        test_dataset = CelebA(root=os.path.join(args.exp, 'datasets', 'celeba_test'), split='test',
                              transform=transforms.Compose([
                                  transforms.CenterCrop(140),
                                  transforms.Resize(config.data.image_size),
                                  transforms.ToTensor(),
                              ]), download=True)


    elif config.data.dataset == 'LSUN':
        train_folder = '{}_train'.format(config.data.category)
        val_folder = '{}_val'.format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor(),
                             ]))
        else:
            dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[train_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

        test_dataset = LSUN(root=os.path.join(args.exp, 'datasets', 'lsun'), classes=[val_folder],
                             transform=transforms.Compose([
                                 transforms.Resize(config.data.image_size),
                                 transforms.CenterCrop(config.data.image_size),
                                 transforms.ToTensor(),
                             ]))

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]), resolution=config.data.image_size)
        else:
            dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=config.data.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)

    return dataset, test_dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
