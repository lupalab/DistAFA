import os
import logging
import pickle
import numpy as np
import scipy.stats
import scipy.special
import torch
import torch.nn.functional as F
import tensorflow as tf
from collections import defaultdict
from pprint import pformat
import itertools

from models import get_model
from utils.hparams import HParams

def get_wrapper(hps):
    if hps.agent == 'img_cls_ppo':
        return FAFA_ppo(hps)
    elif hps.agent == 'img_cls_hppo':
        return FAFA_hppo(hps)
    elif hps.agent == 'img_cls_hgcppo':
        return FAFA_hgcppo(hps)
    elif hps.agent == 'img_cls_hgkppo':
        return FAFA_hgkppo(hps)
    elif hps.agent == 'img_cls_tppo':
        return FAFA_tppo(hps)
    elif hps.agent == 'img_rec_ppo':
        return FAIR_ppo(hps)
    elif hps.agent == 'img_rec_hppo':
        return FAIR_hppo(hps)
    elif hps.agent == 'img_rec_hgkppo':
        return FAIR_hgkppo(hps)
    elif hps.agent == 'vec_cls_ppo':
        return TAFA_ppo(hps)
    elif hps.agent == 'vec_cls_hppo':
        return TAFA_hppo(hps)
    elif hps.agent == 'vec_cls_gkppo':
        return TAFA_gkppo(hps)
    else:
        raise NotImplementedError()


class FAFA_ppo(object):
    def __init__(self, hps):
        self.hps = hps
        H, W, C = hps.image_shape
        self.num_actions = H*W
        
        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()

    def predict(self, state, mask, return_prob=False):
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        if return_prob: return prob
        return np.argmax(prob, axis=-1)

    @property
    def auxiliary_shape(self):
        H, W, C = self.hps.image_shape
        K = self.hps.num_classes
        return [H,W,K+C*4]

    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        K = self.hps.num_classes
        N = self.hps.num_samples
        
        # probability
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        prob = np.repeat(np.repeat(np.reshape(prob, [B,1,1,K]), H, axis=1), W, axis=2)
        
        # sample p(x_u | x_o) and p(x_u | x_o, pred)
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam, pred_sam = self.model.execute([self.model.sam, self.model.pred_sam],
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,H,W,C]) / 255.
        pred_sam = np.reshape(pred_sam, [B,N,H,W,C]) / 255.
        
        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)
        pred_sam_mean = np.mean(pred_sam, axis=1)
        pred_sam_std = np.std(pred_sam, axis=1)
        
        # auxiliary
        aux = np.concatenate([prob, sam_mean, sam_std, pred_sam_mean, pred_sam_std], axis=-1)
        
        return aux

    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        B, H, W, C = mask.shape
        return np.logical_not(mask[:,:,:,0].reshape([B,H*W]).astype(np.bool))

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})
        cmi = pre_ent - post_ent
        
        if not np.all(done): return cmi
        
        # prediction reward
        xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask,
                               self.model.y: target})
        
        return cmi - xent
    
class FAFA_tppo(FAFA_ppo):
    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        empty = action == -1
        terminal = action == self.num_actions
        normal = np.logical_and(~empty, ~terminal)
        reward = np.zeros([action.shape[0]], dtype=np.float32)
        if np.any(terminal):
            s = state[terminal]
            m = mask[terminal]
            y = target[terminal]
            xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: s,
                               self.model.b: m,
                               self.model.m: m,
                               self.model.y: y})
            reward[terminal] = -xent
        if np.any(normal):
            s = state[normal]
            m = mask[normal]
            sn = next_state[normal]
            mn = next_mask[normal]
            pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: s,
                               self.model.b: m,
                               self.model.m: m})
            post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: sn,
                               self.model.b: mn,
                               self.model.m: mn})
            cmi = pre_ent - post_ent
            reward[normal] = cmi
            
        return reward
    
class FAFA_hppo(object):
    def __init__(self, hps):
        self.hps = hps
        H, W, C = hps.image_shape
        
        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()
        
        planner_cfg = HParams(hps.planner_cfg_file)
        self.num_groups = planner_cfg.num_clusters
        self.group_size = H * W // self.num_groups
        
        with open(f'{planner_cfg.exp_dir}/results.pkl', 'rb') as f:
            labels = pickle.load(f)['labels']
        group2idx = defaultdict(list)
        idx2group = dict()
        group_count = [0]*self.num_groups
        for i, lab in enumerate(labels):
            lab = int(lab)
            group2idx[lab].append(i)
            idx2group[i] = (lab, group_count[lab])
            group_count[lab] = group_count[lab] + 1
        self.group2idx = group2idx
        self.idx2group = idx2group
        
        logging.info('>'*30)
        logging.info('action groups -> pixel index:')
        logging.info(f'\n{pformat(self.group2idx)}\n')
        logging.info('pixel index -> (action groups, action index):')
        logging.info(f'\n{pformat(self.idx2group)}\n')
        logging.info('<'*30)
        
    def predict(self, state, mask, return_prob=False):
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        if return_prob: return prob
        return np.argmax(prob, axis=-1)
    
    @property
    def auxiliary_shape(self):
        H, W, C = self.hps.image_shape
        K = self.hps.num_classes
        return [H,W,K+C*4]
    
    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        K = self.hps.num_classes
        N = self.hps.num_samples
        
        # probability
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        prob = np.repeat(np.repeat(np.reshape(prob, [B,1,1,K]), H, axis=1), W, axis=2)
        
        # sample p(x_u | x_o) and p(x_u | x_o, pred)
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam, pred_sam = self.model.execute([self.model.sam, self.model.pred_sam],
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,H,W,C]) / 255.
        pred_sam = np.reshape(pred_sam, [B,N,H,W,C]) / 255.
        
        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)
        pred_sam_mean = np.mean(pred_sam, axis=1)
        pred_sam_std = np.std(pred_sam, axis=1)
        
        # auxiliary
        aux = np.concatenate([prob, sam_mean, sam_std, pred_sam_mean, pred_sam_std], axis=-1)
        
        return aux
        
    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        B, H, W, C = mask.shape
        avail_action = np.ones([mask.shape[0], self.num_groups, self.group_size], dtype=np.bool)
        for i, m in enumerate(mask):
            m = m[...,0].reshape(H*W)
            indexes = np.where(m)[0]
            for idx in indexes:
                avail_action[i, self.idx2group[idx][0], self.idx2group[idx][1]] = 0
        avail_group = np.sum(~avail_action, axis=2) != self.group_size
        
        return avail_group, avail_action
        
    def get_action(self, group, action):
        pixel_idx = []
        for g, a in zip(group, action):
            pixel_idx.append(self.group2idx[g][a])
        return np.array(pixel_idx, dtype=np.int32)
    
    def get_prediction_entropy(self, state, mask):
        ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        return ent
    
    def get_intermediate_reward(self, state, mask, next_state, next_mask):
        # information gain
        pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})
        cmi = pre_ent - post_ent
        
        return cmi
    
    def get_classification_reward(self, state, mask, target):
        # prediction accuracy
        acc = self.model.execute(self.model.acc,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask,
                               self.model.y: target})
        return acc
    
    def get_final_reward(self, state, mask, target):
        # prediction reward
        xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask,
                               self.model.y: target})
        return -xent
        
    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})
        cmi = pre_ent - post_ent
        
        if not np.all(done): return cmi
        
        # prediction reward
        xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask,
                               self.model.y: target})
        
        return cmi - xent

class FAFA_hgcppo(FAFA_hppo):
    def __init__(self, hps):
        self.num_classes = hps.num_classes
        self.goals = list(itertools.combinations(range(self.num_classes), 2))
        self.num_goals = len(self.goals)
        self.num_steps_per_goal = hps.num_steps_per_goal

        self.goal_to_idx = {}
        for i, g in enumerate(self.goals):
            self.goal_to_idx[tuple(g)] = i

        super().__init__(hps)

    def get_intrinsic_reward(self, state, mask, goal, action, next_state, next_mask, done, target):
        pre_prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})

        pre_prob_sub, post_prob_sub = [], [] 
        for i, g in enumerate(goal):
            c1, c2 = self.goals[g]
            pre_prob_sub.append([pre_prob[i][c1], pre_prob[i][c2], 1.0 - pre_prob[i][c1] - pre_prob[i][c2]])
            post_prob_sub.append([post_prob[i][c1], post_prob[i][c2], 1.0 - post_prob[i][c1] - post_prob[i][c2]])
        pre_prob_sub = np.clip(np.array(pre_prob_sub), 0.0, 1.0)
        post_prob_sub = np.clip(np.array(post_prob_sub), 0.0, 1.0)
        
        pre_ent = scipy.stats.entropy(pre_prob_sub, axis=1)
        post_ent = scipy.stats.entropy(post_prob_sub, axis=1)
        cmi = pre_ent - post_ent
        
        return cmi

    def get_meta_reward(self, state, mask, action, next_state, next_mask, done, target):
        return super().get_reward(state, mask, action, next_state, next_mask, done, target)

class FAFA_hgkppo(FAFA_hppo):
    def __init__(self, hps):
        self.num_classes = hps.num_classes
        self.num_clusters = hps.num_clusters
        self.k_clusters = hps.k_clusters
        self.num_steps_per_goal = hps.num_steps_per_goal

        with open(hps.gmm_cfg_file, 'rb') as f:
            self.gmm = pickle.load(f)

        super().__init__(hps)

    def goal_representation(self, goal):
        rep = np.zeros([goal.shape[0], self.num_clusters], dtype=np.float32)
        for i, g in enumerate(goal):
            rep[i][g] = 1.0
        return rep

    def cluster(self, state, mask, return_prob=False, return_raw=False):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        K = self.hps.num_clusters
        N = self.hps.num_samples

        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam = self.model.execute(self.model.sam,
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B*N,H*W*C]).astype(np.float32) / 255.
        prob = self.gmm.predict_proba(sam)
        prob = prob.reshape([B,N,K])
        if return_raw: return prob
        if return_prob: return prob.mean(axis=1)
        return np.argmax(prob.mean(axis=1), axis=-1)

    def get_intrinsic_reward_rest(self, state, mask, goal, action, next_state, next_mask, done, target):
        pre_prob = self.cluster(state, mask, return_prob=True)
        post_prob = self.cluster(next_state, next_mask, return_prob=True)

        pre_prob = pre_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])
        post_prob = post_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])

        pre_rest = 1.0 - np.sum(pre_prob, axis=1, keepdims=True)
        post_rest = 1.0 - np.sum(post_prob, axis=1, keepdims=True)

        pre_prob = np.concatenate([pre_prob, pre_rest], axis=1)
        post_prob = np.concatenate([post_prob, post_rest], axis=1)

        pre_prob = np.clip(pre_prob, 0.0, 1.0)
        post_prob = np.clip(post_prob, 0.0, 1.0)
        
        pre_ent = scipy.stats.entropy(pre_prob, axis=1)
        post_ent = scipy.stats.entropy(post_prob, axis=1)
        cmi = pre_ent - post_ent
        
        return cmi

    def get_intrinsic_reward(self, state, mask, goal, action, next_state, next_mask, done, target):
        pre_prob = self.cluster(state, mask, return_prob=True)
        post_prob = self.cluster(next_state, next_mask, return_prob=True)

        pre_prob = pre_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])
        post_prob = post_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])

        pre_prob /= np.sum(pre_prob, axis=1, keepdims=True) + 1e-8
        post_prob /= np.sum(post_prob, axis=1, keepdims=True) + 1e-8

        pre_prob = np.clip(pre_prob, 0.0, 1.0)
        post_prob = np.clip(post_prob, 0.0, 1.0)
        
        pre_ent = scipy.stats.entropy(pre_prob, axis=1)
        post_ent = scipy.stats.entropy(post_prob, axis=1)
        cmi = pre_ent - post_ent
        
        return cmi

    def get_meta_reward(self, state, mask, action, next_state, next_mask, done, target):
        return super().get_reward(state, mask, action, next_state, next_mask, done, target)
        
class FAIR_ppo(object):
    def __init__(self, hps):
        self.hps = hps
        H, W, C = hps.image_shape
        self.num_actions = H*W
        
        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()

    def predict(self, state, mask, return_samples=False):
        if return_samples:
            B = mask.shape[0]
            H, W, C = self.hps.image_shape
            N = self.hps.num_samples
            state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
            state_extended = np.reshape(state_extended, [B*N,H,W,C])
            mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
            mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
            sam = self.model.execute(self.model.sam,
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
            sam = np.reshape(sam, [B,N,H,W,C])

            return sam
        else:
            mean = self.model.execute(self.model.mean,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: np.ones_like(mask)})

            return mean

    @property
    def auxiliary_shape(self):
        H, W, C = self.hps.image_shape
        return [H,W,C*2]

    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        N = self.hps.num_samples
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam = self.model.execute(self.model.sam,
                feed_dict={self.model.x: state_extended,
                           self.model.b: mask_extended,
                           self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,H,W,C]) / 255.

        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)

        aux = np.concatenate([sam_mean, sam_std], axis=-1)

        return aux

    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        B, H, W, C = mask.shape
        return np.logical_not(mask[:,:,:,0].reshape([B,H*W]).astype(np.bool))

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_bpd = self.model.execute(self.model.bits_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: mask,
                               self.model.m: np.ones_like(mask)})
        post_bpd = self.model.execute(self.model.bits_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: next_mask,
                               self.model.m: np.ones_like(next_mask)})
        gain = pre_bpd - post_bpd

        if not np.all(done): return gain

        # prediction reward
        mse = self.model.execute(self.model.mse_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: next_mask,
                               self.model.m: np.ones_like(next_mask)})
        
        return gain - mse

class FAIR_hppo(object):
    def __init__(self, hps):
        self.hps = hps
        H, W, C = hps.image_shape
        
        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()
        
        planner_cfg = HParams(hps.planner_cfg_file)
        self.num_groups = planner_cfg.num_clusters
        self.group_size = H * W // self.num_groups

        with open(f'{planner_cfg.exp_dir}/results.pkl', 'rb') as f:
            labels = pickle.load(f)['labels']
        group2idx = defaultdict(list)
        idx2group = dict()
        group_count = [0]*self.num_groups
        for i, lab in enumerate(labels):
            lab = int(lab)
            group2idx[lab].append(i)
            idx2group[i] = (lab, group_count[lab])
            group_count[lab] = group_count[lab] + 1
        self.group2idx = group2idx
        self.idx2group = idx2group

        logging.info('>'*30)
        logging.info('action groups -> pixel index:')
        logging.info(f'\n{pformat(self.group2idx)}\n')
        logging.info('pixel index -> (action groups, action index):')
        logging.info(f'\n{pformat(self.idx2group)}\n')
        logging.info('<'*30)

    def predict(self, state, mask, return_samples=False):
        if return_samples:
            B = mask.shape[0]
            H, W, C = self.hps.image_shape
            N = self.hps.num_samples
            state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
            state_extended = np.reshape(state_extended, [B*N,H,W,C])
            mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
            mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
            sam = self.model.execute(self.model.sam,
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
            sam = np.reshape(sam, [B,N,H,W,C])

            return sam
        else:
            mean = self.model.execute(self.model.mean,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: np.ones_like(mask)})

            return mean
    
    @property
    def auxiliary_shape(self):
        H, W, C = self.hps.image_shape
        return [H,W,C*2]

    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        N = self.hps.num_samples
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam = self.model.execute(self.model.sam,
                feed_dict={self.model.x: state_extended,
                           self.model.b: mask_extended,
                           self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,H,W,C]) / 255.

        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)

        aux = np.concatenate([sam_mean, sam_std], axis=-1)

        return aux

    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        B, H, W, C = mask.shape
        avail_action = np.ones([mask.shape[0], self.num_groups, self.group_size], dtype=np.bool)
        for i, m in enumerate(mask):
            m = m[...,0].reshape(H*W)
            indexes = np.where(m)[0]
            for idx in indexes:
                avail_action[i, self.idx2group[idx][0], self.idx2group[idx][1]] = 0
        avail_group = np.sum(~avail_action, axis=2) != self.group_size
        
        return avail_group, avail_action

    def get_action(self, group, action):
        pixel_idx = []
        for g, a in zip(group, action):
            pixel_idx.append(self.group2idx[g][a])
        return np.array(pixel_idx, dtype=np.int32)

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_bpd = self.model.execute(self.model.bits_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: mask,
                               self.model.m: np.ones_like(mask)})
        post_bpd = self.model.execute(self.model.bits_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: next_mask,
                               self.model.m: np.ones_like(next_mask)})
        gain = pre_bpd - post_bpd

        if not np.all(done): return gain

        # prediction reward
        mse = self.model.execute(self.model.mse_per_dim,
                    feed_dict={self.model.x: target,
                               self.model.b: next_mask,
                               self.model.m: np.ones_like(next_mask)})
        
        return gain - mse

class FAIR_hgkppo(FAIR_hppo):
    def __init__(self, hps):
        self.num_classes = hps.num_classes
        self.num_clusters = hps.num_clusters
        self.k_clusters = hps.k_clusters
        self.num_steps_per_goal = hps.num_steps_per_goal

        with open(hps.gmm_cfg_file, 'rb') as f:
            self.gmm = pickle.load(f)

        super().__init__(hps)

    def goal_representation(self, goal):
        rep = np.zeros([goal.shape[0], self.num_clusters], dtype=np.float32)
        for i, g in enumerate(goal):
            rep[i][g] = 1.0
        return rep

    def cluster(self, state, mask, return_prob=False, return_raw=False):
        B = mask.shape[0]
        H, W, C = self.hps.image_shape
        K = self.hps.num_clusters
        N = self.hps.num_samples

        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,H,W,C])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,H,W,C])
        sam = self.model.execute(self.model.sam,
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B*N,H*W*C]).astype(np.float32) / 255.
        prob = self.gmm.predict_proba(sam)
        prob = prob.reshape([B,N,K])
        if return_raw: return prob
        if return_prob: return prob.mean(axis=1)
        return np.argmax(prob.mean(axis=1), axis=-1)

    def get_intrinsic_reward(self, state, mask, goal, action, next_state, next_mask, done, target):
        pre_prob = self.cluster(state, mask, return_prob=True)
        post_prob = self.cluster(next_state, next_mask, return_prob=True)

        pre_prob = pre_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])
        post_prob = post_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])

        pre_rest = 1.0 - np.sum(pre_prob, axis=1, keepdims=True)
        post_rest = 1.0 - np.sum(post_prob, axis=1, keepdims=True)

        pre_prob = np.concatenate([pre_prob, pre_rest], axis=1)
        post_prob = np.concatenate([post_prob, post_rest], axis=1)

        pre_prob = np.clip(pre_prob, 0.0, 1.0)
        post_prob = np.clip(post_prob, 0.0, 1.0)
        
        pre_ent = scipy.stats.entropy(pre_prob, axis=1)
        post_ent = scipy.stats.entropy(post_prob, axis=1)
        cmi = pre_ent - post_ent
        
        return cmi

    def get_meta_reward(self, state, mask, action, next_state, next_mask, done, target):
        return super().get_reward(state, mask, action, next_state, next_mask, done, target)

class TAFA_ppo(object):
    def __init__(self, hps):
        self.hps = hps
        dimension = hps.dimension
        self.num_actions = dimension

        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()

    def predict(self, state, mask, return_prob=False):
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        if return_prob: return prob
        return np.argmax(prob, axis=-1)

    @property
    def auxiliary_shape(self):
        d = self.hps.dimension
        K = self.hps.num_classes
        return [K+d*4]

    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        d = self.hps.dimension
        K = self.hps.num_classes
        N = self.hps.num_samples

        # probability
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        # sample p(x_u | x_o) and p(x_u | x_o, pred)
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,d])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,d])
        sam, pred_sam = self.model.execute([self.model.sam, self.model.pred_sam],
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,d])
        pred_sam = np.reshape(pred_sam, [B,N,d])

        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)
        pred_sam_mean = np.mean(pred_sam, axis=1)
        pred_sam_std = np.std(pred_sam, axis=1)
        
        # auxiliary
        aux = np.concatenate([prob, sam_mean, sam_std, pred_sam_mean, pred_sam_std], axis=-1)
        
        return aux

    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        return np.logical_not(mask.astype(np.bool))

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})
        cmi = pre_ent - post_ent
        
        if not np.all(done): return cmi
        
        # prediction reward
        xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask,
                               self.model.y: target})
        
        return cmi - xent

class TAFA_gkppo(TAFA_ppo):
    def __init__(self, hps):
        self.hps = hps
        dimension = hps.dimension
        self.num_actions = dimension
        self.num_classes = hps.num_classes
        self.num_clusters = hps.num_clusters
        self.k_clusters = hps.k_clusters
        self.num_steps_per_goal = hps.num_steps_per_goal

        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()

        with open(hps.gmm_cfg_file, 'rb') as f:
            self.gmm = pickle.load(f)

    def goal_representation(self, goal):
        rep = np.zeros([goal.shape[0], self.num_clusters], dtype=np.float32)
        for i, g in enumerate(goal):
            rep[i][g] = 1.0
        return rep

    def cluster(self, state, mask, return_prob=False, return_raw=False):
        B = mask.shape[0]
        d = self.hps.dimension
        K = self.hps.num_clusters
        N = self.hps.num_samples

        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,d])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,d])
        sam = self.model.execute(self.model.sam,
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B*N,d]).astype(np.float32)
        prob = self.gmm.predict_proba(sam)
        prob = prob.reshape([B,N,K])
        if return_raw: return prob
        if return_prob: return prob.mean(axis=1)
        return np.argmax(prob.mean(axis=1), axis=-1)

    def get_intrinsic_reward(self, state, mask, goal, action, next_state, next_mask, done, target):
        pre_prob = self.cluster(state, mask, return_prob=True)
        post_prob = self.cluster(next_state, next_mask, return_prob=True)

        pre_prob = pre_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])
        post_prob = post_prob[goal.astype(np.bool)].reshape([goal.shape[0], self.k_clusters])

        pre_rest = 1.0 - np.sum(pre_prob, axis=1, keepdims=True)
        post_rest = 1.0 - np.sum(post_prob, axis=1, keepdims=True)

        pre_prob = np.concatenate([pre_prob, pre_rest], axis=1)
        post_prob = np.concatenate([post_prob, post_rest], axis=1)

        pre_prob = np.clip(pre_prob, 0.0, 1.0)
        post_prob = np.clip(post_prob, 0.0, 1.0)
        
        pre_ent = scipy.stats.entropy(pre_prob, axis=1)
        post_ent = scipy.stats.entropy(post_prob, axis=1)
        cmi = pre_ent - post_ent
        
        return cmi

    def get_meta_reward(self, state, mask, action, next_state, next_mask, done, target):
        return super().get_reward(state, mask, action, next_state, next_mask, done, target)
    
class TAFA_hppo(object):
    def __init__(self, hps):
        self.hps = hps
        dimension = hps.dimension

        model_cfg = HParams(hps.model_cfg_file)
        self.model = get_model(model_cfg)
        self.model.load()

        planner_cfg = HParams(hps.planner_cfg_file)
        self.num_groups = planner_cfg.num_clusters
        self.group_size = dimension // self.num_groups

        with open(f'{planner_cfg.exp_dir}/results.pkl', 'rb') as f:
            labels = pickle.load(f)['labels']
        group2idx = defaultdict(list)
        idx2group = dict()
        group_count = [0]*self.num_groups
        for i, lab in enumerate(labels):
            lab = int(lab)
            group2idx[lab].append(i)
            idx2group[i] = (lab, group_count[lab])
            group_count[lab] = group_count[lab] + 1
        self.group2idx = group2idx
        self.idx2group = idx2group

        logging.info('>'*30)
        logging.info('action groups -> feature index:')
        logging.info(f'\n{pformat(self.group2idx)}\n')
        logging.info('feature index -> (action groups, action index):')
        logging.info(f'\n{pformat(self.idx2group)}\n')
        logging.info('<'*30)

    def predict(self, state, mask, return_prob=False):
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        if return_prob: return prob
        return np.argmax(prob, axis=-1)

    @property
    def auxiliary_shape(self):
        d = self.hps.dimension
        K = self.hps.num_classes
        return [K+d*4]

    def get_auxiliary(self, state, mask):
        B = mask.shape[0]
        d = self.hps.dimension
        K = self.hps.num_classes
        N = self.hps.num_samples

        # probability
        prob = self.model.execute(self.model.prob,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        # sample p(x_u | x_o) and p(x_u | x_o, pred)
        state_extended = np.repeat(np.expand_dims(state, axis=1), N, axis=1)
        state_extended = np.reshape(state_extended, [B*N,d])
        mask_extended = np.repeat(np.expand_dims(mask, axis=1), N, axis=1)
        mask_extended = np.reshape(mask_extended, [B*N,d])
        sam, pred_sam = self.model.execute([self.model.sam, self.model.pred_sam],
                    feed_dict={self.model.x: state_extended,
                               self.model.b: mask_extended,
                               self.model.m: np.ones_like(mask_extended)})
        sam = np.reshape(sam, [B,N,d])
        pred_sam = np.reshape(pred_sam, [B,N,d])

        sam_mean = np.mean(sam, axis=1)
        sam_std = np.std(sam, axis=1)
        pred_sam_mean = np.mean(pred_sam, axis=1)
        pred_sam_std = np.std(pred_sam, axis=1)
        
        # auxiliary
        aux = np.concatenate([prob, sam_mean, sam_std, pred_sam_mean, pred_sam_std], axis=-1)
        
        return aux

    def get_availability(self, state, mask):
        '''1: available    0: occupied'''
        avail_action = np.ones([mask.shape[0], self.num_groups, self.group_size], dtype=np.bool)
        for i, m in enumerate(mask):
            indexes = np.where(m)[0]
            for idx in indexes:
                avail_action[i, self.idx2group[idx][0], self.idx2group[idx][1]] = 0
        avail_group = np.sum(~avail_action, axis=2) != self.group_size
        
        return avail_group, avail_action

    def get_action(self, group, action):
        feat_idx = []
        for g, a in zip(group, action):
            feat_idx.append(self.group2idx[g][a])
        return np.array(feat_idx, dtype=np.int32)

    def get_reward(self, state, mask, action, next_state, next_mask, done, target):
        # information gain
        pre_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: state,
                               self.model.b: mask,
                               self.model.m: mask})
        post_ent = self.model.execute(self.model.ent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask})
        cmi = pre_ent - post_ent
        
        if not np.all(done): return cmi
        
        # prediction reward
        xent = self.model.execute(self.model.xent,
                    feed_dict={self.model.x: next_state,
                               self.model.b: next_mask,
                               self.model.m: next_mask,
                               self.model.y: target})
        
        return cmi - xent