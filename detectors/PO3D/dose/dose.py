import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from .made import ResidualMADE
from .utils import ConvNet
from .mixture import MixtureSameFamily

class DOSE1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.ood_scoring.state_dim
        self.n_components = config.ood_scoring.made_components
        self.gaussian_min_scale = config.ood_scoring.gaussian_min_scale
        self.config = config

        self.made = ResidualMADE(config.ood_scoring.state_dim, 
                                 config.ood_scoring.made_residual_blocks, 
                                 config.ood_scoring.made_hidden_dim,
                                 config.ood_scoring.made_components * 3, 
                                 conditional=True, 
                                 conditioning_dim=config.data.dimension,
                                 use_batch_norm=config.ood_scoring.use_batch_norm)

    def forward(self, state, mask):
        made_outputs = self.made(state, mask)
        made_outputs = made_outputs.reshape(-1, self.dim, self.n_components * 3)
        logits = made_outputs[..., :self.n_components]
        locs = made_outputs[..., self.n_components:self.n_components*2]
        scales = self.gaussian_min_scale + F.softplus(made_outputs[..., self.n_components*2:])
        del made_outputs
        mix = D.Categorical(logits=logits)
        comp = D.Normal(locs, scales)
        gmm = MixtureSameFamily(mix, comp)
        del logits, locs, scales
        logp = gmm.log_prob(state).sum(dim=-1)
        
        return logp

class DOSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.ood_scoring.state_dim
        self.n_components = config.ood_scoring.made_components
        self.gaussian_min_scale = config.ood_scoring.gaussian_min_scale
        self.config = config
        
        self.mask_embed_net = ConvNet(1, 
                                      config.ood_scoring.mask_embed_layers, 
                                      config.ood_scoring.mask_embed_dim)
        self.made = ResidualMADE(config.ood_scoring.state_dim, 
                                 config.ood_scoring.made_residual_blocks, 
                                 config.ood_scoring.made_hidden_dim,
                                 config.ood_scoring.made_components * 3, 
                                 conditional=True, 
                                 conditioning_dim=config.ood_scoring.mask_embed_dim,
                                 use_batch_norm=config.ood_scoring.use_batch_norm)
        
    def forward(self, state, mask):
        mask_embed = self.mask_embed_net(mask)
        made_outputs = self.made(state, mask_embed)
        made_outputs = made_outputs.reshape(-1, self.dim, self.n_components * 3)
        logits = made_outputs[..., :self.n_components]
        locs = made_outputs[..., self.n_components:self.n_components*2]
        scales = self.gaussian_min_scale + F.softplus(made_outputs[..., self.n_components*2:])
        del made_outputs
        mix = D.Categorical(logits=logits)
        comp = D.Normal(locs, scales)
        gmm = MixtureSameFamily(mix, comp)
        del logits, locs, scales
        logp = gmm.log_prob(state).sum(dim=-1)
        
        return logp
        
