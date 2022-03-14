import os
import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import torch
import tensorflow as tf
import numpy as np
from pprint import pformat
import copy

from envs import get_environment
from agents import get_agent
from models.model_wrapper import get_wrapper
from detectors.po3d_wrapper import PO3DWrapper
from utils.hparams import HParams

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str, required=True)
parser.add_argument('--mode', type=str, required=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--detector_device', type=int, default=0)
parser.add_argument('--model_device', type=int, default=1)
parser.add_argument('--agent_device', type=int, default=2)
args = parser.parse_args()
params = HParams(args.cfg_file)

################################################################
logging.basicConfig(filename=params.exp_dir + f'/{args.mode}.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')
logging.info(pformat(params.dict))

np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# detector
with torch.cuda.device(args.detector_device):
    detector = PO3DWrapper(params)

# dynamics model
with tf.device(f'/gpu:{args.model_device}'), torch.cuda.device(args.model_device):
    model = get_wrapper(params)

# train
if args.mode in ['resume', 'train']:
    env = get_environment(params, 'train')

    with tf.device(f'/gpu:{args.agent_device}'):
        agent = get_agent(params, env, model, detector)
        
    if args.mode == 'resume':
        agent.load()
        agent.run()
    if args.mode == 'train':
        agent.run()

# test
if args.mode in ['test', 'plot', 'detect']:
    env = get_environment(params, 'test')

    with tf.device(f'/gpu:{args.agent_device}'):
        agent = get_agent(params, env, model, detector)

    if args.mode == 'test':
        agent.evaluate(num_episodes=params.num_test_episodes)
    if args.mode == 'plot':
        agent.plot(num_episodes=params.num_plot_episodes)
    if args.mode == 'plot_ood':
        agent.plot_ood(num_episodes=params.num_plot_episodes)
    if args.mode == 'detect':
        for ood_dataset in params.ood_datasets:
            params_copy = copy.deepcopy(params)
            params_copy.dataset = ood_dataset
            ood_env = get_environment(params_copy, 'test')
            agent.detect(ood_env, num_episodes=params.num_detect_episodes)
    