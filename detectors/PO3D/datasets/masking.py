import numpy as np
import torch

def get_mask(config):
    if 'bernoulli' in config.data.mask:
        p = float(config.data.mask.split('-')[1])
        return ImageMCARGenerator(p)
    elif 'uniform' in config.data.mask:
        N = config.data.mask.split('-')[1]
        N = None if N == 'all' else int(N)
        return ImageUniformGenerator(N)
    elif 'vector' in config.data.mask:
        N = config.data.mask.split('-')[1]
        N = None if N == 'all' else int(N)
        return UniformGenerator(N)
    elif 'cartesian' in config.data.mask:
        N = config.data.mask.split('-')[1]
        full = N == 'full'
        N = None if N == 'all' or N == 'full' else int(N)
        return CartesianGenerator(N, full)

class ImageMCARGenerator:
    """
    Samples mask from component-wise independent Bernoulli distribution
    with probability of _pixel_ to be unobserved p.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        gen_shape = list(batch.shape)
        gen_shape[1] = 1
        bernoulli_mask_numpy = np.random.choice(2, size=gen_shape, p=[1 - self.p, self.p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).to(batch)
        return bernoulli_mask
    
class ImageUniformGenerator:
    def __init__(self, max_pixels=None):
        self.max_pixels = max_pixels
        
    def __call__(self, batch):
        gen_shape = list(batch.shape)
        gen_shape[1] = 1
        mask = np.zeros(gen_shape, dtype=np.float32).reshape([gen_shape[0], -1])
        num_pixels = np.prod(gen_shape[2:])
        max_pixels = self.max_pixels or num_pixels
        N = [np.random.randint(1, max_pixels+1) for _ in range(gen_shape[0])]
        indexes = [np.random.choice(num_pixels, n, replace=False) for n in N]
        for i, idx in enumerate(indexes):
            mask[i, idx] = 1.0
        mask = mask.reshape(gen_shape)
        mask = torch.from_numpy(mask).to(batch)
        return mask

class UniformGenerator:
    def __init__(self, max_features=None):
        self.max_features = max_features

    def __call__(self, batch):
        B, d = batch.shape
        mask = np.zeros([B,d], dtype=np.float32)
        num_features = batch.shape[1]
        max_features = self.max_features or num_features
        N = [np.random.randint(1, max_features+1) for _ in range(B)]
        indexes = [np.random.choice(num_features, n, replace=False) for n in N]
        for i, idx in enumerate(indexes):
            mask[i, idx] = 1.0
        mask = torch.from_numpy(mask).to(batch)
        return mask

class CartesianGenerator:
    def __init__(self, max_lines=None, full=False):
        self.max_lines = max_lines
        self.full = full

    def __call__(self, batch):
        gen_shape = list(batch.shape)
        gen_shape[1] = 1
        if self.full:
            mask = torch.ones(gen_shape).to(batch)
            return mask
        mask = np.zeros(gen_shape, dtype=np.float32)
        mask[...,:5] = 1
        mask[...,-5:] = 1
        max_lines = self.max_lines or gen_shape[-1]
        N = [np.random.randint(1, max_lines-9) for _ in range(gen_shape[0])]
        indexes = [np.random.choice(np.arange(5,gen_shape[-1]-5), n, replace=False) for n in N]
        for i, idx in enumerate(indexes):
            mask[i,:,:,idx] = 1.0
        mask = torch.from_numpy(mask).to(batch)
        return mask