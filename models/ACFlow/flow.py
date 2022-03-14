import tensorflow as tf
import numpy as np

from models.ACFlow.modules import encoder_spec, decoder_spec
from models.ACFlow.utils import standard_normal_ll, standard_normal_sample
from models.ACFlow.utils import conditional_normal_ll, conditional_normal_sample, conditional_normal_mean
from models.ACFlow.logits import preprocess, postprocess

class Flow(object):
    def __init__(self, hps):
        self.hps = hps

    def forward(self, x, b, m):
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1.- b) * m # 1: query
        x, logdet = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z, ldet = encoder_spec(x, b, m, self.hps, self.hps.n_scale, use_batch_norm=False)
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet
        prior_ll = standard_normal_ll(z)
        prior_ll = tf.reduce_sum(prior_ll * r, [1, 2, 3])
        log_likel = prior_ll + logdet

        return log_likel

    def inverse(self, x, b, m):
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1. - b) * m
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = standard_normal_sample(tf.shape(x))
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, self.hps.n_scale, use_batch_norm=False)
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x

    def mean(self, x, b, m):
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1. - b) * m
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = tf.zeros(tf.shape(x))
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, self.hps.n_scale, use_batch_norm=False)
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x
    
    def cond_forward(self, x, y, b, m):
        image_shape = x.get_shape().as_list()[1:]
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1. - b) * m # 1: query
        x, logdet = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z, ldet = encoder_spec(x, b, m, self.hps, self.hps.n_scale, 
                               use_batch_norm=False, 
                               y_onehot=tf.one_hot(y, self.hps.num_classes))
        ldet = tf.reduce_sum(ldet, [1, 2, 3])
        logdet += ldet
        prior_ll = conditional_normal_ll(z, y, self.hps.num_classes, image_shape)
        prior_ll = tf.reduce_sum(prior_ll * r, [1, 2, 3])
        log_likel = prior_ll + logdet
        
        return log_likel
    
    def cond_inverse(self, x, y, b, m):
        image_shape = x.get_shape().as_list()[1:]
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1. - b) * m
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = conditional_normal_sample(y, self.hps.num_classes, image_shape)
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, 
                            self.hps.n_scale, 
                            use_batch_norm=False,
                            y_onehot=tf.one_hot(y, self.hps.num_classes))
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x

    def cond_mean(self, x, y, b, m):
        image_shape = x.get_shape().as_list()[1:]
        b = tf.cast(b, tf.float32)
        m = tf.cast(m, tf.float32)
        r = (1. - b) * m
        x, _ = preprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        z = conditional_normal_mean(y, self.hps.num_classes, image_shape)
        x = z * r + x * (1. - r)
        x, _ = decoder_spec(x, b, m, self.hps, 
                            self.hps.n_scale, 
                            use_batch_norm=False,
                            y_onehot=tf.one_hot(y, self.hps.num_classes))
        x, _ = postprocess(x, 1.-r, self.hps.data_constraint, self.hps.n_bits)
        
        return x
    