import tensorflow as tf
import numpy as np
import pickle

def uniform_mask(x, max_features=None):
    d = x.shape[1]
    mask = np.zeros_like(x)
    max_num_features = max_features or d
    N = np.random.randint(max_num_features+1)
    idx = np.random.choice(d, size=N, replace=False)
    mask[:,idx] = 1
    return mask.astype(np.float32)

def generate_mask(x, mask_type):
    if mask_type == 'remaining':
        b = uniform_mask(x)
        m = np.ones_like(b)
    elif mask_type == 'bias_remaining':
        b = uniform_mask(x, 64)
        m = np.ones_like(b)
    else:
        raise ValueError()

    return b, m

def augment(x):
    return x

class Dataset(object):
    def __init__(self, dfile, split, batch_size, mask_type, augment):
        g = tf.Graph()
        with g.as_default():
            # open a session
            config = tf.ConfigProto()
            config.log_device_placement = True
            config.allow_soft_placement = True
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config, graph=g)
            # build dataset
            with open(dfile, 'rb') as f:
                data_dict = pickle.load(f)
            data, label = data_dict[split]
            self.size = data.shape[0]
            self.dimension = data.shape[1]
            self.num_batches = self.size // batch_size
            dst = tf.data.Dataset.from_tensor_slices((data, label))
            if split == 'train':
                dst = dst.shuffle(self.size)
            dst = dst.batch(batch_size, drop_remainder=True)
            dst = dst.prefetch(1)

            dst_it = dst.make_initializable_iterator()
            x, y = dst_it.get_next()
            self.x = tf.reshape(x, [batch_size, self.dimension])
            self.y = tf.reshape(y, [batch_size])
            self.initializer = dst_it.initializer
            self.mask_type = mask_type
            self.augment = augment
            self.split = split

    def initialize(self):
        self.sess.run(self.initializer)

    def next_batch(self):
        x, y = self.sess.run([self.x, self.y])
        b, m = generate_mask(x, self.mask_type)
        if self.split == 'train' and self.augment:
            x = augment(x)
        return {'x':x, 'y':y, 'b':b, 'm':m}