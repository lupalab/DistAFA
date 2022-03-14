import os
import yaml
import logging
import argparse
import torch
import numpy as np

from .PO3D.datasets import data_transform
from .PO3D.models import get_model, get_sigmas
from .PO3D.models.ema import EMAHelper
from .PO3D.dose import get_dose
from .PO3D.datasets import ifftshift, to_magnitude

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class PO3DWrapper(object):
    def __init__(self, hps):
        self.hps = hps

        if hps.detector_cfg_file == "" or hps.detector_log_path == "":
            return

        with open(hps.detector_cfg_file, 'r') as f:
            self.config = dict2namespace(yaml.load(f))
        self.config.device = 'cuda'
        self.log_path = hps.detector_log_path

        # load pretrained scorenet
        if self.config.ood_scoring.ckpt_id is None:
            states = torch.load(os.path.join(self.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.log_path, f'checkpoint_{self.config.ood_scoring.ckpt_id}.pth'),
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
        self.score = score
        
        self.sigmas = get_sigmas(self.config)

        # load pretrained dosenet
        if self.config.ood_scoring.ood_ckpt_id is None:
            states = torch.load(os.path.join(self.log_path, 'ood_checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.log_path, f'ood_checkpoint_{self.config.ood_scoring.ood_ckpt_id}.pth'),
                                map_location=self.config.device)
            
        dose = get_dose(self.config)
        dose.load_state_dict(states[0], strict=True)
        dose.eval()
        self.dose = dose

    @torch.no_grad()
    def get_states(self, x, m):
        states = []
        with torch.no_grad():
            for i, sig in enumerate(self.sigmas):
                y = torch.ones([x.shape[0]], device=x.device).long() * i
                score = self.score(x*m, m, y)
                score = score * m * sig
                state = torch.norm(score.view(x.shape[0], -1), dim=-1)
                states.append(state)
        states = torch.stack(states, dim=-1).data
        return states

    @torch.no_grad()
    def log_prob(self, x, m):
        assert x.ndim == 4 # for 2d images
        x = torch.from_numpy(x).float() / 255.
        x = x.permute(0,3,1,2)
        x = x.to(self.config.device)
        x = data_transform(self.config, x)
        m = torch.from_numpy(m).float()
        m = m.permute(0,3,1,2)
        m = m.to(self.config.device)
        m = m[:,:1] # PO3D use 1 channel
        states = self.get_states(x, m)
        with torch.no_grad():
            logp = self.dose(states, m)
        return logp.data.cpu().numpy()

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        if not np.all(done): return np.zeros([state.shape[0]], dtype=np.float32)

        logp = self.log_prob(next_state, next_mask)
        return logp

