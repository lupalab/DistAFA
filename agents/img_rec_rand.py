import os
import logging
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

class RandPolicy():
    def __init__(self, hps, env, model=None, detector=None):
        self.hps = hps
        self.env = env
        self.model = model
        self.detector = detector

    def evaluate(self, num_episodes=10, num_trails=5):
        metrics = []

        for i in range(num_episodes):
            rng = np.random.RandomState(i)
            _, _ = self.env.reset()
            x = self.env.x
            N = self.env.budget
            mse = []
            for _ in range(num_trails):
                m = np.zeros_like(x)
                B,H,W,C = m.shape
                idx = rng.choice(H*W, N, replace=False)
                m = m.reshape([B,H*W,C])
                m[:,idx] = 1
                m = m.reshape([B,H,W,C])
                s = x * m

                pred = self.model.predict(s, m)
                mse.append(np.mean(np.square((pred - self.env.x) / 255.), axis=(1,2,3)))
            mse = np.stack(mse, axis=1)
            metrics.append(mse)

        metrics = np.concatenate(metrics, axis=0)
        mse = np.mean(metrics)

        # log
        logging.info('#'*20)
        logging.info('evaluate:')
        logging.info(f'mse: {mse}')

    def detect(self, ood_env, num_episodes=10, num_trails=5):
        normal_name = self.env.name
        ood_name = ood_env.name

        logp_norm = []
        for i in range(num_episodes):
            rng = np.random.RandomState(i)
            _, _ = self.env.reset()
            x = self.env.x
            N = self.env.budget
            logp = []
            for _ in range(num_trails):
                m = np.zeros_like(x)
                B,H,W,C = m.shape
                idx = rng.choice(H*W, N, replace=False)
                m = m.reshape([B,H*W,C])
                m[:,idx] = 1
                m = m.reshape([B,H,W,C])
                s = x * m
                logp.append(self.detector.log_prob(s, m))
            logp = np.stack(logp, axis=1)
            logp_norm.append(logp)
        logp_norm = np.concatenate(logp_norm, axis=0)

        self.env = ood_env
        logp_ood = []
        for i in range(num_episodes):
            rng = np.random.RandomState(i)
            _, _ = self.env.reset()
            x = self.env.x
            N = self.env.budget
            logp = []
            for _ in range(num_trails):
                m = np.zeros_like(x)
                B,H,W,C = m.shape
                idx = rng.choice(H*W, N, replace=False)
                m = m.reshape([B,H*W,C])
                m[:,idx] = 1
                m = m.reshape([B,H,W,C])
                s = x * m
                logp.append(self.detector.log_prob(s, m))
            logp = np.stack(logp, axis=1)
            logp_ood.append(logp)
        logp_ood = np.concatenate(logp_ood, axis=0)

        neg_logp = np.concatenate([logp_norm, logp_ood], axis=0) * -1.
        label = np.concatenate([np.zeros(logp_norm.shape[0]), np.ones(logp_ood.shape[0])])
        auc = []
        for i in range(num_trails):
            auc.append(roc_auc_score(label, neg_logp[:,i]))
        auc = np.mean(auc)
        
        # log
        logging.info('#'*20)
        logging.info('evaluate:')
        logging.info(f'auc: {auc}')

