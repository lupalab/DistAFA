import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import tensorflow as tf
import numpy as np
from pprint import pformat

from models.group import *
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
params = HParams(args.cfg_file)

logging.basicConfig(filename=params.exp_dir + f'/run.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)-15s %(message)s')
logging.info(pformat(params.dict))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

Grouper = {
    'img': ImgGrouper,
    'rand': RandGrouper,
    'graph': GraphGrouper,
    'mi': MIGrouper,
    'up': UpperMIGrouper,
    'ap': ApproxGrouper,
    'ep': EntGrouper,
}[params.method]

grouper = Grouper(params)
grouper()