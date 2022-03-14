import tensorflow as tf
import numpy as np
import logging

class BasePolicy:
    def __init__(self, hps, env, model=None, detector=None):
        self.hps = hps
        self.env = env
        self.model = model
        self.detector = detector
        
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build model
            self._build_nets()
            self._build_ops()
            # initialize
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.writer = tf.summary.FileWriter(self.hps.exp_dir + '/summary')
            
    def save(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.save(self.sess, fname)
        
    def load(self, filename='params'):
        fname = f'{self.hps.exp_dir}/weights/{filename}.ckpt'
        self.saver.restore(self.sess, fname)
        
    def scope_vars(self, scope, only_trainable=True):
        collection = tf.GraphKeys.TRAINABLE_VARIABLES if only_trainable else tf.GraphKeys.VARIABLES
        variables = tf.get_collection(collection, scope=scope)
        assert len(variables) > 0
        logging.info(f"Variables in scope '{scope}':")
        for v in variables:
            logging.info("\t" + str(v))
        return variables
    
    def _build_nets(self):
        raise NotImplementedError()
    
    def _build_ops(self):
        raise NotImplementedError()