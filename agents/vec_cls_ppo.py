import os
import logging
import pickle
import numpy as np
import tensorflow as tf
from pprint import pformat
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import roc_auc_score
from collections import namedtuple, defaultdict
import tensorflow_probability as tfp
tfd = tfp.distributions

from agents.base import BasePolicy
from agents.memory import ReplayMemory
from agents.networks import dense
from agents.utils import plot_dict, plot_prob
from utils.visualize import save_image, mosaic

class PPOPolicy(BasePolicy):
    def act(self, state, mask, aux, avail, rand=False):
        # random action for warm start
        if rand:
            action = np.array([np.random.choice(np.where(a)[0]) for a in avail], dtype=np.int32)
            return action

        # sample from policy
        prob = self.sess.run(self.actor_prob,
            feed_dict={self.state: state,
                       self.mask: mask,
                       self.auxiliary: aux,
                       self.avail: avail})

        action = np.array([np.random.choice(self.model.num_actions, p=p) for p in prob], dtype=np.int32)

        return action
    
    def _build_nets(self):
        num_actions = self.model.num_actions
        self.state = tf.placeholder(tf.float32, shape=[None,self.hps.dimension], name='state')
        self.mask = tf.placeholder(tf.float32, shape=[None,self.hps.dimension], name='mask')
        self.auxiliary = tf.placeholder(tf.float32, shape=[None]+self.model.auxiliary_shape, name='auxiliary')
        self.avail = tf.placeholder(tf.bool, shape=[None,self.hps.dimension], name='avail')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.old_logp = tf.placeholder(tf.float32, shape=[None], name='old_logp')
        self.v_target = tf.placeholder(tf.float32, shape=[None], name='v_target')
        self.adv = tf.placeholder(tf.float32, shape=[None], name='adv')
        
        with tf.variable_scope('embedding'):
            embed = tf.concat([self.state, self.mask, self.auxiliary], axis=-1)
            embed = dense(embed, self.hps.embed_layers, name='embed')
            self.embed_vars = self.scope_vars('embedding')
            
        with tf.variable_scope('actor'):
            actor_layers = self.hps.actor_layers + [num_actions]
            actor_logits = dense(embed, actor_layers, output='', name='actor')
            inf_tensor = -tf.ones_like(actor_logits) * np.inf
            self.actor_logits = tf.where(self.avail, actor_logits, inf_tensor)
            self.actor_ent = tfd.Categorical(logits=self.actor_logits).entropy()
            self.actor_prob = tf.nn.softmax(self.actor_logits)
            self.actor_log_prob = tf.nn.log_softmax(self.actor_logits)
            index = tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1)
            self.logp = tf.gather_nd(self.actor_log_prob, index)
            self.actor_vars = self.scope_vars('actor')

        with tf.variable_scope('critic'):
            inp = dense(embed, self.hps.critic_layers, name='critic')
            critic = tf.layers.dense(inp, 1, name='value')
            self.critic = tf.squeeze(critic, axis=-1)
            self.critic_vars = self.scope_vars('critic')

    def _build_ops(self):
        self.lr_a = tf.placeholder(tf.float32, shape=None, name='learning_rate_actor')
        self.lr_c = tf.placeholder(tf.float32, shape=None, name='learning_rate_critic')
        self.clip_range = tf.placeholder(tf.float32, shape=None, name='ratio_clip_range')

        with tf.variable_scope('actor_train'):
            ratio = tf.exp(self.logp - self.old_logp)
            ratio_clipped = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            loss_a = - tf.reduce_mean(tf.minimum(self.adv * ratio, self.adv * ratio_clipped))

            optim_a = tf.train.AdamOptimizer(self.lr_a)
            grads_and_vars = optim_a.compute_gradients(loss_a, var_list=self.actor_vars+self.embed_vars)
            grads_a, vars_a = zip(*grads_and_vars)
            if self.hps.clip_grad_norm > 0:
                grads_a, gnorm_a = tf.clip_by_global_norm(grads_a, clip_norm=self.hps.clip_grad_norm)
                gnorm_a = tf.check_numerics(gnorm_a, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_a', gnorm_a)
            grads_and_vars = zip(grads_a, vars_a)
            self.train_op_a = optim_a.apply_gradients(grads_and_vars)
        
        with tf.variable_scope('critic_train'):
            loss_c = tf.reduce_mean(tf.square(self.v_target - self.critic))
            
            optim_c = tf.train.AdamOptimizer(self.lr_c)
            grads_and_vars = optim_c.compute_gradients(loss_c, var_list=self.critic_vars+self.embed_vars)
            grads_c, vars_c = zip(*grads_and_vars)
            if self.hps.clip_grad_norm > 0:
                grads_c, gnorm_c = tf.clip_by_global_norm(grads_c, clip_norm=self.hps.clip_grad_norm)
                gnorm_c = tf.check_numerics(gnorm_c, "Gradient norm is NaN or Inf.")
                tf.summary.scalar('gnorm_c', gnorm_c)
            grads_and_vars = zip(grads_c, vars_c)
            self.train_op_c = optim_c.apply_gradients(grads_and_vars)
            
        self.train_ops = tf.group(self.train_op_a, self.train_op_c)

        with tf.variable_scope('summary'):
            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')

            self.summary = [
                tf.summary.scalar('loss/adv', tf.reduce_mean(self.adv)),
                tf.summary.scalar('loss/ratio', tf.reduce_mean(ratio)),
                tf.summary.scalar('loss/loss_actor', loss_a),
                tf.summary.scalar('loss/loss_critic', loss_c),
                tf.summary.scalar('episode_reward', self.ep_reward)
            ]
            
            self.merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

    def _generate_rollout(self, buffer, rand=False):
        states = []
        masks = []
        auxiliaries = []
        avails = []
        actions = []
        rewards = []
        dones = []
        logps = []
        vpreds = []

        logging.info('start rollout.')
        s, m  = self.env.reset()
        episode_reward = np.zeros([s.shape[0]], dtype=np.float32)
        done = np.zeros([s.shape[0]], dtype=np.bool)
        while not np.all(done):
            aux = self.model.get_auxiliary(s, m)
            avail = self.model.get_availability(s, m)
            action = self.act(s, m, aux, avail, rand=rand)
            s_next, m_next, re, done = self.env.step(action)
            ri = self.model.get_reward(s, m, action, s_next, m_next, done, self.env.y)
            r = re + ri
            if self.hps.detector_reward_coef > 0:
                rd = self.detector.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                r = r + rd * self.hps.detector_reward_coef
            logp, vpred = self.sess.run(
                [self.logp, self.critic],
                feed_dict={self.state: s,
                           self.mask: m,
                           self.auxiliary: aux,
                           self.avail: avail,
                           self.action: action})
            states.append(s)
            masks.append(m)
            auxiliaries.append(aux)
            avails.append(avail)
            actions.append(action)
            rewards.append(r)
            dones.append(done)
            logps.append(logp)
            vpreds.append(vpred)
            episode_reward += r
            s, m = s_next, m_next
        logging.info('rollout finished.')

        # Compute TD errors
        B = len(s)
        T = len(rewards)
        rewards = np.stack(rewards)
        td_errors = [rewards[t] + self.hps.gamma * vpreds[t+1] - vpreds[t] for t in range(T-1)]
        td_errors += [rewards[T-1] + self.hps.gamma * 0.0 - vpreds[T-1]]
        
        # Estimate advantage backwards.
        advs = []
        adv_so_far = 0.0
        for delta in td_errors[::-1]:
            adv_so_far = delta + self.hps.gamma * self.hps.lam * adv_so_far
            advs.append(adv_so_far)
        advs = advs[::-1]
        
        # Estimate critic target
        v_target = np.stack(advs) + np.stack(vpreds)

        # record this batch
        logging.info('record this batch.')
        n_rec = B * T
        for t in range(T):
            for j in range(B):
                item = buffer.tuple_class(
                    states[t][j],
                    masks[t][j],
                    auxiliaries[t][j],
                    avails[t][j],
                    actions[t][j],
                    logps[t][j],
                    v_target[t][j],
                    advs[t][j]
                )
                buffer.add(item)
        logging.info(f'record done: {n_rec} transitions added.')
        
        return np.mean(episode_reward), n_rec

    def _ratio_clip_fn(self, n_iter):
        clip = self.hps.ratio_clip_range
        if self.hps.ratio_clip_decay:
            delta = clip / self.hps.train_iters
            clip -= delta * n_iter

        return max(0.0, clip)

    def run(self):
        BufferRecord = namedtuple('BufferRecord', 
            ['state', 'mask', 'aux', 'avail', 'action', 'old_logp', 'v_target', 'adv'])
        buffer = ReplayMemory(tuple_class=BufferRecord, capacity=self.hps.buffer_size)
        
        reward_history = []
        reward_averaged = []
        best_reward = -np.inf
        step = 0
        total_rec = 0

        for n_iter in range(self.hps.n_iters):
            clip = self._ratio_clip_fn(n_iter)
            if self.hps.clean_buffer: buffer.clean()
            rand = False
            if hasattr(self.hps, 'warmup_iters') and n_iter <self.hps.warmup_iters:
                rand = True
            ep_reward, n_rec = self._generate_rollout(buffer, rand=rand)
            reward_history.append(ep_reward)
            reward_averaged.append(np.mean(reward_history[-10:]))
            total_rec += n_rec

            for batch in buffer.loop(self.hps.buffer_batch_size, self.hps.buffer_epochs):
                _, summ_str = self.sess.run(
                    [self.train_ops, self.merged_summary],
                    feed_dict={self.lr_a: self.hps.lr_a,
                               self.lr_c: self.hps.lr_c,
                               self.clip_range: clip,
                               self.state: batch['state'],
                               self.mask: batch['mask'],
                               self.auxiliary: batch['aux'],
                               self.avail: batch['avail'],
                               self.action: batch['action'],
                               self.old_logp: batch['old_logp'],
                               self.v_target: batch['v_target'],
                               self.adv: batch['adv'],
                               self.ep_reward: np.mean(reward_history[-10:]) if reward_history else 0.0,
                               }
                )
                self.writer.add_summary(summ_str, step)
                step += 1

            if self.hps.log_freq > 0 and (n_iter+1) % self.hps.log_freq == 0:
                logging.info("[iteration:{}/step:{}], best:{}, avg:{:.2f}, clip:{:.2f}; {} transitions.".format(
                    n_iter, step, np.max(reward_history), np.mean(reward_history[-10:]), clip, total_rec))

                data_dict = {
                    'reward': reward_history,
                    'reward_smooth10': reward_averaged,
                }
                plot_dict(f'{self.hps.exp_dir}/learning_curve.png', data_dict, xlabel='episode')
                
            if self.hps.eval_freq > 0 and n_iter % self.hps.eval_freq == 0:
                self.evaluate(load=False)
                
            if self.hps.save_freq > 0 and (n_iter+1) % self.hps.save_freq == 0:
                self.save()

            if np.mean(reward_history[-10:]) > best_reward:
                best_reward = np.mean(reward_history[-10:])
                self.save('best')

        # FINISH
        self.save()
        logging.info("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))
        
        # Evaluate
        self.evaluate()

    def evaluate(self, load=True, num_episodes=10):
        if load: self.load('best')
        
        metrics = defaultdict(list)
        acquisitions = []
        trajectories = []
        predictions = []

        for _ in range(num_episodes):
            s, m = self.env.reset()
            episode_reward = np.zeros([s.shape[0]], dtype=np.float32)
            done = np.zeros([s.shape[0]], dtype=np.bool)
            traj = []
            while not np.all(done):
                aux = self.model.get_auxiliary(s, m)
                avail = self.model.get_availability(s, m)
                action = self.act(s, m, aux, avail)
                s_next, m_next, re, done = self.env.step(action)
                ri = self.model.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                r = re + ri
                if self.hps.detector_reward_coef > 0:
                    rd = self.detector.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                    r = r + rd * self.hps.detector_reward_coef
                episode_reward += r
                prob = self.model.predict(s, m, return_prob=True)
                traj += [(self.env.x.copy(), self.env.y.copy(), s.copy(), m.copy(), prob.copy(), action.copy())]
                s, m = s_next, m_next
            prob = self.model.predict(s, m, return_prob=True)
            traj += [(self.env.x.copy(), self.env.y.copy(), s.copy(), m.copy(), prob.copy(), action.copy())]
            trajectories.append(traj)

            metrics['episode_reward'].append(episode_reward)
            
            pred = self.model.predict(s, m)
            acc = (pred == self.env.y).astype(np.float32)
            predictions.append(pred)
            
            metrics['accuracy'].append(acc)

            acquisitions.append((s, m))

        # concat metrics
        average_metrics = defaultdict(float)
        for k, v in metrics.items():
            metrics[k] = np.concatenate(v)
            average_metrics[k] = np.mean(metrics[k])
        
        # log
        logging.info('#'*20)
        logging.info('evaluate:')
        for k, v in average_metrics.items():
            logging.info(f'{k}: {v}')

        # save
        with open(f'{self.hps.exp_dir}/trajectories.pkl', 'wb') as f:
            pickle.dump((trajectories, predictions), f)

        return acquisitions