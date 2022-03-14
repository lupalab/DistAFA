import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions

from models.base import BaseModel
from models.ACFlow import Flow

class FlowClassifier(BaseModel):
    def __init__(self, hps):
        self.flow = Flow(hps)
        super(FlowClassifier, self).__init__(hps)
        
    def classify(self, x, b, m):
        '''
        logits = log p(x_u | x_o, y)
        logits: [B, N]
        '''
        # tile 10 times to compute p(. | y) for each y
        B = tf.shape(x)[0]
        H,W,C = x.get_shape().as_list()[1:]
        N = self.hps.num_classes
        x = tf.tile(tf.expand_dims(x,axis=1),[1,N,1,1,1])
        x = tf.reshape(x, [B*N,H,W,C])
        b = tf.tile(tf.expand_dims(b,axis=1),[1,N,1,1,1])
        b = tf.reshape(b, [B*N,H,W,C])
        m = tf.tile(tf.expand_dims(m,axis=1),[1,N,1,1,1])
        m = tf.reshape(m, [B*N,H,W,C])
        y = tf.tile(tf.expand_dims(tf.range(N),axis=0), [B,1])
        y = tf.reshape(y, [B*N])

        logp = self.flow.cond_forward(x, y, b, m)
        logits = tf.reshape(logp, [B,N])

        return logits
    
    def sample(self, x, y, b, m):
        B = tf.shape(x)[0]
        if y is None:
            y = tf.random_uniform([B], dtype=tf.int64, minval=0, maxval=self.hps.num_classes)
        sam = self.flow.cond_inverse(x, y, b, m)
        
        return sam
        
    def build_net(self):
        with tf.variable_scope('flow_classifier', reuse=tf.AUTO_REUSE):
            self.x = tf.placeholder(tf.float32, [None]+self.hps.image_shape)
            self.y = tf.placeholder(tf.int64, [None])
            self.b = tf.placeholder(tf.float32, [None]+self.hps.image_shape)
            self.m = tf.placeholder(tf.float32, [None]+self.hps.image_shape)
            
            # log p(x_u | x_o, y)
            self.logp_u = self.classify(self.x, self.b, self.m)
            # log p(x_o | y)
            self.logp_o = self.classify(self.x, self.b*(1-self.b), self.b)
            # log p(x_u, x_o | y)
            self.logits = self.logp_u + self.logp_o
            # p(y | x_o, x_u)
            self.prob = tf.nn.softmax(self.logits)
            self.ent = tfd.Categorical(logits=self.logits).entropy()
            self.pred = tf.argmax(self.logits, axis=1)
            self.acc = tf.cast(tf.equal(self.pred, self.y), tf.float32)
            self.xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            # log p(x_u | x_o)
            self.logp = (tf.reduce_logsumexp(self.logp_o + self.logp_u, axis=1) - 
                        tf.reduce_logsumexp(self.logp_o, axis=1))
            # log p(x_u | x_o, y)
            self.cond_logp = self.flow.cond_forward(self.x, self.y, self.b, self.m)
            # sample p(x_u | x_o)
            self.sam = self.sample(self.x, None, self.b, self.m)
            # sample p(x_u | x_o, y)
            self.cond_sam = self.sample(self.x, self.y, self.b, self.m)
            # sample p(x_u | x_o, ys)
            ys = tf.argmax(self.logp_o, axis=1)
            self.pred_sam = self.sample(self.x, ys, self.b, self.m)
            
            # metric
            self.metric = self.acc
            
            # loss
            xent1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            xent1 = tf.reduce_mean(xent1)
            tf.summary.scalar('xent1', xent1)
            xent2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logp_o, labels=self.y)
            xent2 = tf.reduce_mean(xent2)
            tf.summary.scalar('xent2', xent2)
            num_dimensions = tf.reduce_sum(self.m * (1-self.b), axis=[1,2,3]) + 1e-6
            nll = tf.reduce_mean(-self.logp)
            tf.summary.scalar('nll', nll)
            l2_reg = sum(
                [tf.reduce_sum(tf.square(v)) for v in tf.trainable_variables()
                 if ("magnitude" in v.name) or ("rescaling_scale" in v.name)])
            self.loss = (xent1 * self.hps.lambda_xent +
                         xent2 * self.hps.lambda_xent + 
                         nll * self.hps.lambda_nll + 
                         l2_reg * self.hps.lambda_reg)
            tf.summary.scalar('loss', self.loss)
            