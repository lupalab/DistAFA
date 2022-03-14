import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the embedding')
    parser.add_argument('--test', action='store_true', help='Whether to test')
    parser.add_argument('--search', action='store_true', help='Search masks')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)
    
    if not args.plot and not args.test and not args.search:
        with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'ood_stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)
    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)
        
        if args.plot:
            os.makedirs(os.path.join(args.exp, 'ood_embedding'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'ood_embedding', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
        elif args.test:
            os.makedirs(os.path.join(args.exp, 'ood_scoring'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'ood_scoring', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
        elif args.search:
            os.makedirs(os.path.join(args.exp, 'ood_searching'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'ood_searching', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            
    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)
    
    try:
        runner = OODRunner(args, config)
        if args.plot:
            runner.plot()
        elif args.test:
            runner.test()
        elif args.search:
            runner.search()
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())