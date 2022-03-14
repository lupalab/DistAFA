import numpy as np

def get_dataset(hps, split):
    if hps.dataset == 'tiny_mnist':
        from .tiny_mnist import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type)
    elif hps.dataset == 'mnist':
        from .mnist import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type)
    elif hps.dataset == 'fmnist':
        from .fmnist import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type)
    elif hps.dataset == 'omniglot':
        from .omniglot import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type)
    elif hps.dataset == 'cifar10':
        from .cifar10 import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type, hps.augment)
    elif hps.dataset == 'svhn':
        from .svhn import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type, hps.augment)
    elif hps.dataset == 'celeba':
        from .celeba import Dataset
        dataset = Dataset(split, hps.batch_size, hps.mask_type)
    elif hps.dataset == 'uci':
        from .uci import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size, hps.mask_type, hps.augment)
    else:
        raise Exception()

    return dataset

def inf_generator(dataset):
    while True:
        try:
            batch = dataset.next_batch()
            yield batch
        except:
            dataset.initialize()