import os
import logging
import numpy as np
import tensorflow as tf

from datasets import get_dataset, inf_generator

class Env(object):
    def __init__(self, hps, split):
        self.hps = hps
        self.budget = hps.budget
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
        self.m[np.arange(len(action)), action] = 1.

        state = self.x * self.m
        mask = self.m.copy()
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        done = np.sum(self.m, axis=1) >= self.budget
        
        return state, mask, reward, done
