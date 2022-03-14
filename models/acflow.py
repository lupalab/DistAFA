import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

from models.base import BaseModel
from models.ACFlow import Flow

class ACFlow(BaseModel):
    def __init__(self, hps):
        self.flow = Flow(hps)
        super(ACFlow, self).__init__(hps)

    def build_net(self):
        with tf.variable_scope('flow', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None]+self.hps.image_shape)
            self.b = tf.placeholder(tf.float32, [None]+self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None]+self.hps.image_shape)

            # log p(x_u | x_o)
            self.logp = self.flow.forward(self.x, self.b, self.m)
            num_dimensions = tf.reduce_sum(self.m * (1-self.b), axis=[1,2,3]) + 1e-6
            self.bits_per_dim = -self.logp / num_dimensions
            # sample p(x_u | x_o)
            self.sam = self.flow.inverse(self.x, self.b, self.m)
            # mean p(x_u | x_o)
            self.mean = self.flow.mean(self.x, self.b, self.m)
            self.mse = tf.reduce_sum(tf.square((self.mean - self.x) / 255.), axis=[1,2,3])
            self.mse_per_dim = self.mse / num_dimensions

            # metric
            self.metric = -self.mse_per_dim

            # loss
            nll = tf.reduce_mean(self.bits_per_dim)
            tf.summary.scalar('nll', nll)
            mse = tf.reduce_mean(self.mse_per_dim)
            tf.summary.scalar('mse', mse)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            self.loss = (nll * self.hps.lambda_nll +
                         mse * self.hps.lambda_mse +
                         l2_reg * self.hps.lambda_reg)
            tf.summary.scalar('loss', self.loss)
