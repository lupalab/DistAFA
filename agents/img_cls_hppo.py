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
from agents.networks import convnet, dense
from agents.utils import plot_dict, plot_prob
from utils.visualize import save_image, mosaic

class PPOPolicy(BasePolicy):
    def act(self, state, mask, aux, avail_g, avail_a, rand=False):
        # random action for warm start
        if rand:
            group = np.array([np.random.choice(np.where(g)[0]) for g in avail_g], dtype=np.int32)
            avail_a = np.stack([a[g] for g, a in zip(group, avail_a)])
            action = np.array([np.random.choice(np.where(a)[0]) for a in avail_a], dtype=np.int32)
            return group, action

        # predict group
        group_prob = self.sess.run(self.group_prob,
            feed_dict={self.state: state,
                       self.mask: mask,
                       self.auxiliary: aux,
                       self.avail_group: avail_g})

        group = np.array([np.random.choice(self.model.num_groups, p=p) for p in group_prob], dtype=np.int32)
        
        # predict action in the group
        avail_a = np.stack([a[g] for g, a in zip(group, avail_a)])
        action_prob = self.sess.run(self.action_prob,
            feed_dict={self.state: state,
                       self.mask: mask,
                       self.auxiliary: aux,
                       self.group: group,
                       self.avail_action: avail_a})

        action = np.array([np.random.choice(self.model.group_size, p=p) for p in action_prob], dtype=np.int32)
            
        return group, action
        
    def _build_nets(self):
        num_groups = self.model.num_groups
        group_size = self.model.group_size
        self.state = tf.placeholder(tf.float32, shape=[None]+self.hps.image_shape, name='state')
        self.mask = tf.placeholder(tf.float32, shape=[None]+self.hps.image_shape, name='mask')
        self.auxiliary = tf.placeholder(tf.float32, shape=[None]+self.model.auxiliary_shape, name='auxiliary')
        self.avail_group = tf.placeholder(tf.bool, shape=[None, num_groups], name='avail_group')
        self.avail_action = tf.placeholder(tf.bool, shape=[None, group_size], name='avail_action')
        self.group = tf.placeholder(tf.int32, shape=[None], name='group')
        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.old_logp = tf.placeholder(tf.float32, shape=[None], name='old_logp')
        self.v_target = tf.placeholder(tf.float32, shape=[None], name='v_target')
        self.adv = tf.placeholder(tf.float32, shape=[None], name='adv')
        
        with tf.variable_scope('embedding'):
            embed = tf.concat([self.state / 255., self.mask, self.auxiliary], axis=-1)
            embed = convnet(embed, self.hps.embed_layers, name='embed')
            self.embed_vars = self.scope_vars('embedding')
            
        with tf.variable_scope('actor'):
            inp = dense(embed, self.hps.actor_layers, name='actor')
            # group
            group_logits = tf.layers.dense(inp, num_groups, name='group_logits')
            inf_tensor = -tf.ones_like(group_logits) * np.inf
            self.group_logits = tf.where(self.avail_group, group_logits, inf_tensor)
            self.group_ent = tfd.Categorical(logits=self.group_logits).entropy()
            self.group_prob = tf.nn.softmax(self.group_logits)
            self.group_log_prob = tf.nn.log_softmax(self.group_logits)
            index = tf.stack([tf.range(tf.shape(self.group)[0]), self.group], axis=1)
            self.logp_group = tf.gather_nd(self.group_log_prob, index) 
            # action
            inp = tf.concat([inp, tf.one_hot(self.group, num_groups)], axis=1)
            action_logits = tf.layers.dense(inp, group_size, name='action_logits')
            inf_tensor = -tf.ones_like(action_logits) * np.inf
            self.action_logits = tf.where(self.avail_action, action_logits, inf_tensor)
            self.action_ent = tfd.Categorical(logits=self.action_logits).entropy()
            self.action_prob = tf.nn.softmax(self.action_logits)
            self.action_log_prob = tf.nn.log_softmax(self.action_logits)
            index = tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1)
            self.logp_action = tf.gather_nd(self.action_log_prob, index)
            
            self.logp = self.logp_group + self.logp_action
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
        avail_groups = []
        avail_actions = []
        groups = []
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
            avail_g, avail_a = self.model.get_availability(s, m)
            g, a = self.act(s, m, aux, avail_g, avail_a, rand=rand)
            action = self.model.get_action(g, a)
            s_next, m_next, re, done = self.env.step(action)
            ri = self.model.get_reward(s, m, action, s_next, m_next, done, self.env.y)
            r = re + ri
            if self.hps.detector_reward_coef > 0:
                rd = self.detector.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                r = r + rd * self.hps.detector_reward_coef
            avail_a = np.stack([ai[gi] for gi, ai in zip(g, avail_a)])
            logp, vpred = self.sess.run(
                [self.logp, self.critic],
                {self.state: s, 
                 self.mask: m, 
                 self.auxiliary: aux,
                 self.avail_group: avail_g,
                 self.avail_action: avail_a,
                 self.group: g,
                 self.action: a})
            states.append(s)
            masks.append(m)
            auxiliaries.append(aux)
            avail_groups.append(avail_g)
            avail_actions.append(avail_a)
            groups.append(g)
            actions.append(a)
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
                    avail_groups[t][j],
                    avail_actions[t][j],
                    groups[t][j],
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
            ['state', 'mask', 'aux', 'avail_group', 'avail_action', 'group', 'action', 'old_logp', 'v_target', 'adv'])
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
                               self.avail_group: batch['avail_group'],
                               self.avail_action: batch['avail_action'],
                               self.group: batch['group'],
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
        
        for _ in range(num_episodes):
            s, m = self.env.reset()
            episode_reward = np.zeros([s.shape[0]], dtype=np.float32)
            done = np.zeros([s.shape[0]], dtype=np.bool)
            traj = []
            while not np.all(done):
                aux = self.model.get_auxiliary(s, m)
                avail_g, avail_a = self.model.get_availability(s, m)
                g, a = self.act(s, m, aux, avail_g, avail_a)
                action = self.model.get_action(g, a)
                s_next, m_next, re, done = self.env.step(action)
                ri = self.model.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                r = re + ri
                if self.hps.detector_reward_coef > 0:
                    rd = self.detector.get_reward(s, m, action, s_next, m_next, done, self.env.y)
                    r = r + rd * self.hps.detector_reward_coef
                episode_reward += r
                prob = self.model.predict(s, m, return_prob=True)
                traj += [(self.env.x.copy(), self.env.y.copy(), s.copy(), m.copy(), prob.copy(), g.copy(), a.copy(), action.copy())]
                s, m = s_next, m_next
            prob = self.model.predict(s, m, return_prob=True)
            traj += [(self.env.x.copy(), self.env.y.copy(), s.copy(), m.copy(), prob.copy(), g.copy(), a.copy(), action.copy())]
            trajectories.append(traj)
            
            metrics['episode_reward'].append(episode_reward)
            
            pred = self.model.predict(s, m)
            acc = (pred == self.env.y).astype(np.float32)
            
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
            pickle.dump(trajectories, f)

        return acquisitions
                
    def plot(self, num_episodes=10):
        save_dir = f'{self.hps.exp_dir}/figures'
        os.makedirs(save_dir, exist_ok=True)
        self.load('best')
        
        for i in range(num_episodes):
            logging.info(f'# episode: {i}')
            sdir = f'{save_dir}/{i}'
            os.makedirs(sdir, exist_ok=True)
            s, m = self.env.reset()
            done = np.zeros([s.shape[0]], dtype=np.bool)
            save_image(self.env.x.astype(np.uint8), f'{sdir}/x.png')
            pickle.dump(self.env.x.astype(np.uint8), open(f'{sdir}/x.pkl', 'wb'))
            step = 1
            while not np.all(done):
                aux = self.model.get_auxiliary(s, m)
                avail_g, avail_a = self.model.get_availability(s, m)
                g, a = self.act(s, m, aux, avail_g, avail_a)
                logging.info(f'step: {step}')
                logging.info(f'group: {pformat(g)}')
                logging.info(f'action: {pformat(a)}')
                action = self.model.get_action(g, a)
                s_next, m_next, re, done = self.env.step(action)
                s, m = s_next, m_next
                prob = self.model.predict(s, m, return_prob=True)
                save_image(mosaic(self.env.x, m), f'{sdir}/s{step}.png')
                pickle.dump(mosaic(self.env.x, m), open(f'{sdir}/s{step}.pkl', 'wb'))
                plot_prob(prob, f'{sdir}/p{step}.png')
                pickle.dump(prob, open(f'{sdir}/p{step}.pkl','wb'))
                step += 1

    def plot_ood(self, num_episodes=10):
        save_dir = f'{self.hps.exp_dir}/ood_figures'
        os.makedirs(save_dir, exist_ok=True)
        self.load('best')
        
        for i in range(num_episodes):
            sdir = f'{save_dir}/{i}'
            os.makedirs(sdir, exist_ok=True)
            s, m = self.env.reset()
            done = np.zeros([s.shape[0]], dtype=np.bool)
            save_image(self.env.x.astype(np.uint8), f'{sdir}/x.png')
            pickle.dump(self.env.x.astype(np.uint8), open(f'{sdir}/x.pkl', 'wb'))
            step = 1
            while not np.all(done):
                aux = self.model.get_auxiliary(s, m)
                avail_g, avail_a = self.model.get_availability(s, m)
                g, a = self.act(s, m, aux, avail_g, avail_a)
                action = self.model.get_action(g, a)
                s_next, m_next, re, done = self.env.step(action)
                s, m = s_next, m_next
                likel = self.detector.log_prob(s, m)
                save_image(mosaic(self.env.x, m), f'{sdir}/s{step}.png')
                pickle.dump(mosaic(self.env.x, m), open(f'{sdir}/s{step}.pkl', 'wb'))
                pickle.dump(likel, open(f'{sdir}/p{step}.pkl','wb'))
                step += 1

    def detect(self, ood_env, num_episodes=10):
        save_dir = f'{self.hps.exp_dir}/detection'
        os.makedirs(save_dir, exist_ok=True)

        normal_name = self.env.name
        ood_name = ood_env.name

        logging.info('>> normal data')
        acq_norm = self.evaluate(num_episodes=num_episodes)
        self.env = ood_env
        logging.info('>> ood data')
        acq_ood = self.evaluate(num_episodes=num_episodes)

        logp_norm = []
        for s, m in acq_norm:
            logp_norm.append(self.detector.log_prob(s, m))
        logp_norm = np.concatenate(logp_norm, axis=0)
        save_image(mosaic(s, m), f'{save_dir}/{normal_name}_acq.png')

        logp_ood = []
        for s, m in acq_ood:
            logp_ood.append(self.detector.log_prob(s, m))
        logp_ood = np.concatenate(logp_ood, axis=0)
        save_image(mosaic(s, m), f'{save_dir}/{ood_name}_acq.png')

        neg_logp = np.concatenate([logp_norm, logp_ood]) * -1.
        label = np.concatenate([np.zeros_like(logp_norm), np.ones_like(logp_ood)])
        auc = roc_auc_score(label, neg_logp)

        with open(f'{save_dir}/{normal_name}_{ood_name}.pkl', 'wb') as f:
            pickle.dump({
                'acq_norm': acq_norm, 
                'acq_ood': acq_ood,
                'logp_norm': logp_norm,
                'logp_ood': logp_ood}, f)
        
        fig, ax = plt.subplots()
        plt.title(f'AUROC:{auc}')
        sns.histplot({'normal': logp_norm, 'ood': logp_ood}, kde=True, legend=True, ax=ax)
        plt.savefig(f'{save_dir}/{normal_name}_{ood_name}_logp.png')
        plt.close(fig=fig)
