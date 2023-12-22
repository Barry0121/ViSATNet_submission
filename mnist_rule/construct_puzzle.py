'''
This file takes the common constraints and add partial assignments to it to form a puzzle;
For example, if we want to classify an image, the partial assignments are 28*28 pixels
divided into patches (by default 16 patches of 7*7 chunks)
The variables to be solved are #patches * #bits_representation for the first layer of SATNet

For the second layer of SATNet, the partial assignments are the #patches * 3bits_representation
and the output (variables to be solved) are 10 classes
'''

from utils.get_dataloader import get_tarin_loader, get_test_loader
from tqdm import tqdm
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default="stats", choices=["construct", "solve", "stats"])
    parser.add_argument('--show_image', type=bool, default=False)
    
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--root_dir', type=str, default='data')

    # specify trained model
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=123)
    
    # specify the patches to be solved
    parser.add_argument('--config_start', type=int, default=0)
    parser.add_argument('--config_num', type=int, default=1)
    parser.add_argument('--patch_start', type=int, default=0)
    parser.add_argument('--patch_num', type=int, default=1)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def _get_model_path(args):
    """Return the target model/loss path given arguments.
    If the path doesn't exist, make it exist."""
    exp_name = os.path.join(args.save_dir, args.dataset,
                            (f"model_{args.model}_m_{args.m}_aux_{args.aux}"
                             f"_dim_{args.hidden_dim}_stride_{args.stride}"
                             f"_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}"))
    os.makedirs(exp_name, exist_ok=True)
    return exp_name
    
    
def get_patches(args):
    # read the file that saves patches samples.
    data_path = os.path.join(_get_model_path(args), "patch_config_data.pickle")
    print(f"Loading data from {data_path}")
    # data = np.load(data_path)
    # for key in data:
    #     print(key)
    
    # with np.load(data_path) as data:
    #     configs = data['configuration']
    #     patches = data['patches']
    
    with open(data_path, 'rb') as handle: 
        record = pickle.load(handle)
    
    for key in record:
        record[key] = np.unique(record[key], axis=0)
        
    # print(record.keys())
    # print(record['10000100'][200])
    # print(patches[0])
    
    # get the data patches <-> config to a dictionary
    
    # return the dictionary
    
    return record

def append_pixels(args, patch, config, i, fcommon):
    '''
    takes one patch and add to the common constraint file to create a puzzle file
    '''
    
    inf_dir = os.path.join(_get_model_path(args), 'solver_inference')
    puzzle_path_dir = os.path.join(inf_dir, 'maxsatfiles')
    # print("puzzle_path_dir is:", puzzle_path_dir)
    os.makedirs(puzzle_path_dir, exist_ok=True)
    
    fpuzzle = os.path.join(puzzle_path_dir, f'config_{config}_pat{i}.txt')
    
    if args.show_image:
        
        plt.imshow(patch.reshape(7,7), cmap='gray')
        plt.savefig(os.path.join(puzzle_path_dir, f'config_{config}_pat{i}_img.png'))
    
    patch[patch < 0.5] = 0
    patch[patch >= 0.5] = 1
    
    if args.show_image:
        plt.imshow(patch.reshape(7,7), cmap='gray')
        plt.savefig(os.path.join(puzzle_path_dir, f'config_{config}_pat{i}_img_thresh.png'))
    
    
    with open(fpuzzle, 'w') as fpuz:
        for i in range(2, len(patch)+2):
            if patch[i-2] > 0:
                fpuz.write( f'h {i} 0\n' )
            else:
                fpuz.write( f'h -{i} 0\n' )
    
    status = os.system(f'cat {os.path.join(inf_dir, fcommon)} >> {fpuzzle}')
    if status != 0:
        print(f"fail to append {fcommon} to {fpuzzle}")
        return ""
    
    

def construct_puzzle(args, record):
    
    configs = list(record.keys())
    
    for config in configs[args.config_start:args.config_start+args.config_num]:
        for i in range(args.patch_start, args.patch_start+args.patch_num):
            patch = record[config][i]
            append_pixels(args, patch, config, i, 'common_w1.txt')
        

def solve_puzzle(args, record):
    
    configs = list(record.keys())
    
    for config in configs[args.config_start:args.config_start+args.config_num]:
        for i in range(args.patch_start, args.patch_start+args.patch_num):
            puzzle_dir = os.path.join(_get_model_path(args), "solver_inference", "maxsatfiles")
            puzzle_fp = os.path.join(puzzle_dir, f"config_{config}_pat{i}.txt")
            
            status = os.system(f'./mnist_rule/cashwmaxsatcoreplus -m {puzzle_fp} -cpu-lim=500 >> {os.path.join(puzzle_dir, f"config_{config}_pat{i}.sol")}')
            
            if status != 0:
                print(f"fail to solve {puzzle_fp}")
                return ""
            
            print(f"Solved {puzzle_fp}")
            

def _get_solved(f):
    
    with open(f, 'r') as fin:
        for l in fin.read().splitlines():
            if l.startswith('v'):
                result = l.split(' ')
                solved = result[51:59]
                solved_51_58 = ''.join(['1' if int(ele) > 0 else '0' for ele in solved])
                # print(solved_51_58)

                solved = result[101:109]
                solved_101_108 = ''.join(['1' if int(ele) > 0 else '0' for ele in solved])
                       
                return solved_51_58, solved_101_108
            
            
def get_stats(args, record):
    
    configs = list(record.keys())
    
    sol_path = os.path.join(_get_model_path(args), "solver_inference", "maxsatfiles")
    
    correct_101_108 = 0
    correct_51_58 = 0
    
    all = 0
    
    for config in configs[args.config_start:args.config_start+args.config_num]:
        for i in range(args.patch_start, args.patch_start+args.patch_num):
            sol_file = os.path.join(sol_path, f"config_{config}_pat{i}.sol")
            
            result_51_58, result_101_108 = _get_solved(sol_file)
            if result_51_58 == config: 
                correct_51_58 += 1
                # print(result_51_58)
                # print(config)
            if result_101_108 == config: correct_101_108 += 1
            
            all += 1 
            
            
    print(f"Correct ratio if taking 51-58: {correct_51_58}/{all} = ", correct_51_58/all*100)
    print(f"Correct ratio if taking 101-108: {correct_101_108}/{all} = ", correct_101_108/all*100)
    
            
            
            
    
      
    
if __name__ == "__main__":
    args = get_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # get data samples into a dictionary
    
    record = get_patches(args)
    
    
    if args.mode == 'construct':
        construct_puzzle(args, record)
        
    elif args.mode == 'solve':
        solve_puzzle(args, record)
        
    elif args.mode == 'stats':
        get_stats(args, record)
        
    else:
        print("!! Invalid mode !!")
        