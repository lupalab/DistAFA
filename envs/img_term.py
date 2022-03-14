import os
import logging
import numpy as np
import tensorflow as tf

from datasets import get_dataset, inf_generator

class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.cost = hps.cost
        # build dataset
        self.name = hps.dataset
        self.dataset = get_dataset(hps, split)
        self.data_gen = inf_generator(self.dataset)
        
    def reset(self):
        batch = next(self.data_gen)
        self.x = batch['x']
        self.y = batch['y']
        self.m = np.zeros_like(self.x)
        
        state = self.x * self.m
        mask = self.m.copy()
        
        return state, mask
    
    def step(self, action):
        B,H,W,C = self.m.shape
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        done = np.zeros([action.shape[0]], dtype=np.bool)
        isvalid = np.logical_and(action>=0, action<H*W) # invalid action indicates termination
        if np.any(isvalid):
            a = action[isvalid]
            m = self.m[isvalid]
            m = m.reshape([len(m), H*W, C])
            m[np.arange(len(a)), a] = 1.
            m = m.reshape([len(m),H,W,C])
            self.m[isvalid] = m.copy()
        reward[isvalid] = -self.cost
        done[~isvalid] = True
        state = self.x * self.m
        mask = self.m.copy()
        
        return state, mask, reward, done
            