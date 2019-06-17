from tdl import runner
from utils import dict_to_object
from itertools import product
import argparse
import pdb

args = {
    'env_name': 'Hopper-v2',
    'seed': 6666,
    'num_episode': 2000,
    'batch_size': 2048,
    'gamma': 0.995,
    'lamda': 0.97,
    'log_num_episode': 1,
    'val_num_episode': 10,
    'num_epoch': 60,
    'minibatch_size': 256,
    'loss_coeff_value': 0.5,
    'loss_coeff_entropy': 0.0,
    'lr': 1e-4,
    'num_parallel_run': 5,
    'use_cuda': True,
    'record_KL': True,
    # tricks
    'layer_norm': True,
    'state_norm': True,
    'lossvalue_norm': True,
    'advantage_norm': False,
    'append_time': True,
    'schedule_adam': 'linear',
    # experiments
    'label': 'myalgo',
    'method': 'direct',
    'init_std': 0.3,                    # hyperparameter for all methods
    'step_size': 1.0,                   # hyperparameter for ``method=ES, ES-MA1, ES-MA2``
    'schedule_stepsize': 'constant',
    'y2_max': 0.05,                     # hyperparameter for ``method=direct``
    'schedule_y2max': 'constant',
    'n_points': 2,                      # hyperparameter for ``method=ES-MA1, ES-MA2``
    'mean_ratio': 0.1,                  # hyperparameter for ``method=ES-MA1``
    'schedule_meanratio': 'constant',
    'beta': 0.1,                        # hyperparameter for ``method=ES-MA2``
}

def test(args, label):
    record_dfs = []
    for i in range(args.num_parallel_run):
        model = runner(args)
        args.seed += 1

def train(term_args):
    myalgo_args = args.copy()
    method = 'direct'
    for init_std, y2_max in product([0.3, 1.0], [0.025, 0.050, 0.100]):
        myalgo_args.update({
            'label': term_args.label,
            'env_name': term_args.env, 
            'method': method,
            'init_std': init_std, 
            'y2_max': y2_max,
        })
    myalgo_args = dict_to_object(myalgo_args)
    test(myalgo_args, term_args.label)

    myalgo_args = args.copy()
    method = 'ES'
    for init_std, step_size in product([0.3, 1.0], [0.05, 0.10, 0.50, 1.00]):
        myalgo_args.update({
            'label': term_args.label,
            'env_name': term_args.env, 
            'method': method,
            'init_std': init_std,
            'step_size': y2_max,
        })
    myalgo_args = dict_to_object(myalgo_args)
    test(myalgo_args, term_args.label)

    myalgo_args = args.copy()
    method = 'ES-MA1'
    for init_std, step_size, mean_ratio, n_points in product([0.3, 1.0], [0.05, 0.10, 0.50, 1.00], [0.1, 1.0], [2, 5]):
        myalgo_args.update({
            'label': term_args.label,
            'env_name': term_args.env, 
            'method': method,
            'init_std': init_std,
            'step_size': y2_max,
            'mean_ratio': mean_ratio,
            'n_points': n_points,
        })
    myalgo_args = dict_to_object(myalgo_args)
    test(myalgo_args, term_args.label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--label', type=str, default='default')
    term_args = parser.parse_args()
    train(term_args)

if __name__ == '__main__':
    main()
