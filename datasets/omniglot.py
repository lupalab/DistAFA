import os
import pickle
import numpy as np
import tensorflow as tf

from .mask_generators import *

image_shape = [28,28,1]
data_path = '/data/omniglot/'
with open(data_path + 'train_vinyals_aug90.pkl', 'rb') as f:
    train_dict = pickle.load(f, encoding='bytes')
with open(data_path + 'val_vinyals_aug90.pkl', 'rb') as f:
    valid_dict = pickle.load(f, encoding='bytes')
with open(data_path + 'test_vinyals_aug90.pkl', 'rb') as f:
    test_dict = pickle.load(f, encoding='bytes')

mask_gen = OmniglotMaskGenerator() # 0: unobserved 1: observed
pixel_gen = PixelGenerator()
rand_gen = ImageMCARGenerator(0.5)

def generate_mask(image, mask_type):
    if mask_type == b'single_pixel':
        mask = mask_gen(image)
        miss = pixel_gen(image, mask)
        miss = mask + miss
    elif mask_type == b'random_pixel':
        mask = mask_gen(image)
        miss = rand_gen(image)
        miss = miss * (1-mask)
        miss = mask + miss
    elif mask_type == b'random_block':
        mask = mask_gen(image)
        miss = mask_gen(image)
        miss = miss * (1-mask)
        miss = mask + miss
    elif mask_type == b'remaining':
        mask = mask_gen(image)
        miss = np.ones_like(mask)
    else:
        raise ValueError()
    
    return mask, miss

def _parse_train(i, mask_type):
    image = train_dict[b'images'][i]
    label = train_dict[b'labels'][i]
    mask, miss = generate_mask(image, mask_type)

    return image, label, mask, miss

def _parse_valid(i, mask_type):
    image = valid_dict[b'images'][i]
    label = valid_dict[b'labels'][i]
    mask, miss = generate_mask(image, mask_type)

    return image, label, mask, miss

def _parse_test(i, mask_type):
    image = test_dict[b'images'][i]
    label = test_dict[b'labels'][i]
    mask, miss = generate_mask(image, mask_type)

    return image, label, mask, miss

def get_dst(split, mask_type):
    if split == 'train':
        size = train_dict[b'images'].shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.shuffle(size)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_train, [i, mask_type],
                       [tf.uint8, tf.int32, tf.uint8, tf.uint8])),
                      num_parallel_calls=16)
    elif split == 'valid':
        size = valid_dict[b'images'].shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_valid, [i, mask_type],
                       [tf.uint8, tf.int32, tf.uint8, tf.uint8])),
                      num_parallel_calls=16)
    else:
        size = test_dict[b'images'].shape[0]
        inds = tf.range(size, dtype=tf.int32)
        dst = tf.data.Dataset.from_tensor_slices(inds)
        dst = dst.map(lambda i: tuple(
            tf.py_func(_parse_test, [i, mask_type],
                       [tf.uint8, tf.int32, tf.uint8, tf.uint8])),
                      num_parallel_calls=16)

    return dst, size

class Dataset(object):
    def __init__(self, split, batch_size, mask_type):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            dst, size = get_dst(split, mask_type)
            self.size = size
            self.num_batches = self.size // batch_size
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, y, b, m = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size] + image_shape)
            self.y = tf.reshape(y, [batch_size])
            self.b = tf.reshape(b, [batch_size] + image_shape)
            self.m = tf.reshape(m, [batch_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)
        
    def next_batch(self):
        x, y, b, m = self.sess.run([self.x, self.y, self.b, self.m])
        return {'x':x, 'y':y, 'b':b, 'm':m}