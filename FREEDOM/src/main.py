# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


def run_training(model='FREEDOM', dataset='hm', config_dict=None, save_model=True):
    if config_dict is None:
        config_dict = {}
    config_dict.setdefault('gpu_id', 0)

    quick_start(model=model, dataset=dataset, config_dict=config_dict, save_model=save_model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='FREEDOM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='hm', help='name of datasets')
    parser.add_argument('--epochs', type=int, default=None, help='override number of training epochs')
    parser.add_argument('--stopping_step', '--stopping-step', type=int, default=None, help='early stopping patience in eval steps')
    parser.add_argument('--early_stopping_min_delta', '--early-stopping-min-delta', type=float, default=None, help='minimum validation improvement')
    parser.add_argument('--disable_early_stopping', '--disable-early-stopping', action='store_true', help='train until epochs without early stopping')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id for local training')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_args()
    config_dict = {
        'gpu_id': args.gpu_id,
    }
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.stopping_step is not None:
        config_dict['stopping_step'] = args.stopping_step
    if args.early_stopping_min_delta is not None:
        config_dict['early_stopping_min_delta'] = args.early_stopping_min_delta
    if args.disable_early_stopping:
        config_dict['early_stopping'] = False

    run_training(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


