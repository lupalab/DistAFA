import os
import logging
import numpy as np
import tensorflow as tf
from pprint import pformat

from utils.visualize import save_image

class Runner(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        
    def set_dataset(self, trainset, validset, testset):
        self.trainset = trainset
        self.validset = validset
        self.testset = testset

    def make_feed_dict(self, batch, train=False):
        feed_dict = {self.model.x:batch['x'],
                     self.model.b:batch['b']}
        if hasattr(self.model, 'm') and 'm' in batch:
            feed_dict.update({self.model.m:batch['m']})
        if hasattr(self.model, 'y') and 'y' in batch: 
            feed_dict.update({self.model.y:batch['y']})
        if hasattr(self.model, 'is_training'):
            feed_dict.update({self.model.is_training:train})

        return feed_dict
        
    def train(self):
        train_metrics = []
        num_batches = self.trainset.num_batches
        self.trainset.initialize()
        for i in range(num_batches):
            batch = self.trainset.next_batch()
            feed_dict = self.make_feed_dict(batch, train=True)
            metric, summ, step, _ = self.model.execute(
                [self.model.metric, self.model.summ_op, 
                 self.model.global_step, self.model.train_op], 
                feed_dict)
            if (self.args.summ_freq > 0) and (i % self.args.summ_freq == 0):
                self.model.writer.add_summary(summ, step)
            train_metrics.append(metric)
        train_metrics = np.concatenate(train_metrics, axis=0)

        return np.mean(train_metrics)
    
    def valid(self):
        valid_metrics = []
        num_batches = self.validset.num_batches
        self.validset.initialize()
        for i in range(num_batches):
            batch = self.validset.next_batch()
            feed_dict = self.make_feed_dict(batch)
            metric = self.model.execute(self.model.metric, feed_dict)
            valid_metrics.append(metric)
        valid_metrics = np.concatenate(valid_metrics, axis=0)

        return np.mean(valid_metrics)
    
    def test(self):
        test_metrics = []
        num_batches = self.testset.num_batches
        self.testset.initialize()
        for i in range(num_batches):
            batch = self.testset.next_batch()
            feed_dict = self.make_feed_dict(batch)
            metric = self.model.execute(self.model.metric, feed_dict)
            test_metrics.append(metric)
        test_metrics = np.concatenate(test_metrics, axis=0)

        return np.mean(test_metrics)
    
    def run(self):
        logging.info('==== start training ====')
        best_train_metric = -np.inf
        best_valid_metric = -np.inf
        best_test_metric = -np.inf
        for epoch in range(self.args.epochs):
            train_metric = self.train()
            valid_metric = self.valid()
            test_metric = self.test()
            # save
            if train_metric > best_train_metric:
                best_train_metric = train_metric
            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                self.model.save()
            if test_metric > best_test_metric:
                best_test_metric = test_metric
            logging.info("Epoch %d, train: %.4f/%.4f, valid: %.4f/%.4f test: %.4f/%.4f" %
                 (epoch, train_metric, best_train_metric, 
                 valid_metric, best_valid_metric,
                 test_metric, best_test_metric))
            # evaluate
            if epoch % self.args.eval_freq == 0:
                logging.info('==== start evaluating ====')
                self.evaluate(folder=f'{epoch}', load=False, total_batches=1)
            self.model.save('last')
            
        # finish
        logging.info('==== start evaluating ====')
        self.evaluate(load=True)
        
    def evaluate(self, folder='test', load=True, total_batches=np.inf):
        save_dir = f'{self.args.exp_dir}/evaluate/{folder}/'
        os.makedirs(save_dir, exist_ok=True)
        if load: self.model.load()
        
        # accuracy
        if 'acc' in self.args.eval_metrics:
            self.testset.initialize()
            accuracy = []
            for i in range(self.testset.num_batches):
                if i >= total_batches: break
                batch = self.testset.next_batch()
                feed_dict = self.make_feed_dict(batch)
                acc = self.model.execute(self.model.acc, feed_dict)
                accuracy.append(acc)
            accuracy = np.mean(np.concatenate(accuracy))
            
            with open(f'{save_dir}/acc.txt', 'w') as f:
                f.write(f'accuracy: {accuracy}')

        # mse
        if 'mse' in self.args.eval_metrics:
            self.testset.initialize()
            mean_squared_error = []
            for i in range(self.testset.num_batches):
                if i >= total_batches: break
                batch = self.testset.next_batch()
                feed_dict = self.make_feed_dict(batch)
                mse = self.model.execute(self.model.mse, feed_dict)
                mean_squared_error.append(mse)
            mean_squared_error = np.mean(np.concatenate(mean_squared_error))

            with open(f'{save_dir}/mse.txt', 'w') as f:
                f.write(f'mse: {mean_squared_error}')
                
        # log_likel
        if 'nll' in self.args.eval_metrics:
            self.testset.initialize()
            log_likel = []
            for i in range(self.testset.num_batches):
                if i >= total_batches: break
                batch = self.testset.next_batch()
                feed_dict = self.make_feed_dict(batch)
                logp = self.model.execute(self.model.logp, feed_dict)
                log_likel.append(logp)
            log_likel = np.mean(np.concatenate(log_likel))
            
            with open(f'{save_dir}/nll.txt', 'w') as f:
                f.write(f'nll: {-log_likel}')

        # sample
        if 'sam' in self.args.eval_metrics:
            self.testset.initialize()
            batch = self.testset.next_batch()
            batch['m'] = np.ones_like(batch['m'])
            feed_dict = self.make_feed_dict(batch)
            sam = self.model.execute(self.model.sam, feed_dict)
            
            mask = (batch['b'] * 255).astype(np.uint8)
            save_image(mask.astype(np.uint8), f'{save_dir}/mask.png')
            save_image(sam.astype(np.uint8), f'{save_dir}/sam.png')
        
        # cond sample
        if 'cond_sam' in self.args.eval_metrics:
            self.testset.initialize()
            batch = self.testset.next_batch()
            batch['m'] = np.ones_like(batch['m'])
            feed_dict = self.make_feed_dict(batch)
            sam, cond_sam = self.model.execute([self.model.sam, self.model.cond_sam], feed_dict)
            
            mask = (batch['b'] * 255).astype(np.uint8)
            save_image(mask.astype(np.uint8), f'{save_dir}/mask.png')
            save_image(sam.astype(np.uint8), f'{save_dir}/sam.png')
            save_image(cond_sam.astype(np.uint8), f'{save_dir}/cond_sam.png')
            with open(f'{save_dir}/cond.txt', 'w') as f:
                f.write('labels:\n')
                f.write(pformat(batch['y']))