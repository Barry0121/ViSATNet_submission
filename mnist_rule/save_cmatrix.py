'''
save corresponding C_matrix given the model weights
currently assuming 2 layers of SATNet 
'''

import numpy as np
from models import *
from utils.get_dataloader import get_tarin_loader, get_test_loader
from utils.random_mask import generate_batches
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import argparse
from loss_func.loss import dice_loss
from time import time
from utils.setup_singlebitDataset import setup_one_digit

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--digit', type=str)
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    # original:
    # parser.add_argument('--model', type=str, default='hrgen')
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--hidden_dim2', type=int, default=10) # for three layer
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--load_weights', type=bool, default=False) # not working yet
    return parser.parse_args()


def _get_model_path(args):
    """Return the target model/loss path given arguments.
    If the path doesn't exist, make it exist."""
    exp_name = os.path.join(args.save_dir, args.dataset,
                            (f"model_{args.model}_m_{args.m}_aux_{args.aux}"
                             f"_dim_{args.hidden_dim}_stride_{args.stride}"
                             f"_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}"))
    os.makedirs(exp_name, exist_ok=True)
    return exp_name

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_weights(args):
    
    model = HierarchicalClassifier(m=args.m, aux=args.aux, stride=args.stride, hidden_dim=args.hidden_dim).to(device)
    
    exp_path = _get_model_path(args)
    weight_file = os.path.join(exp_path, "model/best_model.pt")
    
    print(f"loading model from {exp_path}")
    
    model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu'))['model_state_dict'])
    
    print("model loaded!")
    # print(model.state_dict()['conv1.sat.S'].shape)
    s_matrix1 = model.state_dict()['conv1.sat.S']
    print("shape of s_matrix1", s_matrix1.shape)
    c_matrix1 = torch.matmul(s_matrix1, torch.transpose(s_matrix1, 0, 1))
    print("shape of c_matrix 1", c_matrix1.shape)
    # print(c_matrix1[0])
    
    s_matrix2 = model.state_dict()['conv2.sat.S']
    print("shape of s_matrix2", s_matrix2.shape)
    c_matrix2 = torch.matmul(s_matrix2, torch.transpose(s_matrix2, 0, 1))
    print("shape of c_matrix2", c_matrix2.shape)
    # print(c_matrix2[0])
    
    new_weights = {
    'conv1.sat.C': c_matrix1,
    'conv2.sat.C': c_matrix2
    }
    
    save_path = os.path.join(exp_path, 'model/c_matrices.pt')
    torch.save(new_weights, save_path)
    print("c_matrices saved to ", save_path)
    
    
    
    
if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    get_weights(args)