import os
import tensorflow as tf
import numpy as np
from PIL import Image

from datasets.mask_generators import *

image_shape = [64, 64, 3]
path = '/data/celebA/img/'
list_file = '/data/celebA/Eval/list_eval_partition.txt'
with open(list_file, 'r') as f:
    all_list = f.readlines()
all_list = [e.strip().split() for e in all_list]
train_list = [e[0] for e in all_list if e[1] == '0']
train_list = [path + e for e in train_list]
valid_list = [e[0] for e in all_list if e[1] == '1']
valid_list = [path + e for e in valid_list]
test_list = [e[0] for e in all_list if e[1] == '2']
test_list = [path + e for e in test_list]

mask_gen = CelebAMaskGenerator()
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

def _parse(f, mask_type):
    image = Image.open(f).crop((25, 45, 153, 173)).resize((64, 64))
    image = np.array(image).astype('uint8')
    mask, miss = generate_mask(image, mask_type)

    return image, mask, miss

def get_dst(split, mask_type):
    if split == 'train':
        size = len(train_list)
        files = tf.constant(train_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)
        dst = dst.shuffle(size)
    elif split == 'valid':
        size = len(valid_list)
        files = tf.constant(valid_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)
    else:
        size = len(test_list)
        files = tf.constant(test_list, dtype=tf.string)
        dst = tf.data.Dataset.from_tensor_slices(files)

    dst = dst.map(lambda f: tuple(
        tf.py_func(_parse, [f, mask_type], [tf.uint8, tf.uint8, tf.uint8])),
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
            x, b, m = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size] + image_shape)
            self.b = tf.reshape(b, [batch_size] + image_shape)
            self.m = tf.reshape(m, [batch_size] + image_shape)
            self.image_shape = image_shape
            self.initializer = dst_it.initializer

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, b, m = self.sess.run([self.x, self.b, self.m])
        return {'x':x, 'b':b, 'm':m}