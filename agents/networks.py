import numpy as np
import tensorflow as tf
from tensorflow.nn.rnn_cell import GRUCell, MultiRNNCell

def gru(inputs, layer_sizes, name='gru'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cells = [GRUCell(size) for size in layer_sizes]
        rnn_cell = MultiRNNCell(cells)
        state = rnn_cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=state, dtype=tf.float32)

    return outputs

def dense(inputs, layer_sizes, output='lrelu', name='dense'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            out = tf.layers.dense(out, size, name=f'l{i}')
            if i < len(layer_sizes)-1:
                out = tf.nn.leaky_relu(out)
                
        # output activation
        if output == 'lrelu':
            out = tf.nn.leaky_relu(out)
        elif output == 'tanh':
            out = tf.nn.tanh(out)
        else:
            out = tf.identity(out)
        
    return out

def convnet(inputs, layer_sizes, name='convnet'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            out = tf.layers.conv2d(out, size, 3, padding='same', name=f'c{i}')
            out = tf.contrib.layers.instance_norm(out)
            out = tf.nn.leaky_relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)

    return out

def deepset(inputs, layer_sizes, name, reduction='max'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            out = tf.layers.dense(out, size, name=f'l{i}')
            if reduction == 'max':
                res = tf.reduce_max(out, axis=1, keepdims=True)
            elif reduction == 'mean':
                res = tf.reduce_mean(out, axis=1, keepdims=True)
            else:
                raise NotImplementedError()
            out = out - res
            out = tf.nn.leaky_relu(out)
    
    return out

def set_attention(Q, K, dim, num_heads, name='set_attention'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        q = tf.layers.dense(Q, dim, name='query')
        k = tf.layers.dense(K, dim, name='key')
        v = tf.layers.dense(K, dim, name='value')

        q_ = tf.concat(tf.split(q, num_heads, axis=-1), axis=0)
        k_ = tf.concat(tf.split(k, num_heads, axis=-1), axis=0)
        v_ = tf.concat(tf.split(v, num_heads, axis=-1), axis=0)

        logits = tf.matmul(q_, k_, transpose_b=True)/np.sqrt(dim) # [B*Nh,Nq,Nk]
        A = tf.nn.softmax(logits, axis=-1)
        o = q_ + tf.matmul(A, v_)
        o = tf.concat(tf.split(o, num_heads, axis=0), axis=-1)
        o = o + tf.layers.dense(o, dim, activation=tf.nn.relu, name='output')

    return o

def set_transformer(inputs, layer_sizes, name, num_heads=4, num_inds=16):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out = inputs
        for i, size in enumerate(layer_sizes):
            inds = tf.get_variable(f'inds_{i}', shape=[1,num_inds,size], dtype=tf.float32, trainable=True,
                                    initializer=tf.contrib.layers.xavier_initializer())
            inds = tf.tile(inds, [tf.shape(out)[0],1,1])
            tmp = set_attention(inds, out, size, num_heads, name=f'self_attn_{i}_pre')
            out = set_attention(out, tmp, size, num_heads, name=f'self_attn_{i}_post')

    return out

def set_pooling(inputs, name, num_heads=4):
    B = tf.shape(inputs)[0]
    N = tf.shape(inputs)[1]
    d = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        seed = tf.get_variable('pool_seed', shape=[1,1,d], dtype=tf.float32, trainable=True, 
                                initializer=tf.contrib.layers.xavier_initializer())
        seed = tf.tile(seed, [B,1,1])
        out = set_attention(seed, inputs, d, num_heads, name='pool_attn')
        out = tf.squeeze(out, axis=1)

    return out