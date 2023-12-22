from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from models import *
from utils.get_dataloader import get_test_loader
import numpy as np
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

import generate_dist



def get_args():
    parser = argparse.ArgumentParser()
    # specification
    parser.add_argument('--mode', type=str, default='config_distribution',
                        choices=['config_distribution', 'bits_distribution',
                                 'visualize_patches', 'extract_dataset', 'verify_patches'])
    parser.add_argument('--sampling', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument("--save_raw", type=bool, default=True)
    parser.add_argument("--save_formatted", type=bool, default=True)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument("--solve", type=bool, default=False)

    # specify trained model
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threshold', type=float, default=0.4)

    return parser.parse_args()


def main(args, device):
    
    debug_dir = '/u/sissij/projects/satnet_image_completion/mnist_rule/debug/debug_results'
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load the model with corresponding weights (from the model directory!)
    model, dataloader = generate_dist._get_model_dataset(args=args, device=device)
    model.eval()

    num_patches = generate_dist._get_num_patches(args.stride)
    print("num of patches: ", num_patches)

    # Input one image (16 patch)
    # Run evaluation
    print("Genearte bits for testing images...")
    loader = tqdm(dataloader)
    
    itr = iter(loader)
    image, label = next(itr)
    # this is the saved example of a "2" image
    image, label = next(itr)
    
    image = image.to(device)
    
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.savefig('/u/sissij/projects/satnet_image_completion/mnist_rule/debug/debug_results/orig_image.png')
    
    label = label.to(device)

    # Generate output
    pred = model(image)
    print("Predicted output from second layer of SANet", pred)

    pred = torch.argmax(pred, dim=-1)
    print("Predicted label: ", pred)
    # 1=correct, 0=false; (batch_size, 1)
    # correct = torch.where(pred == label, 1., 0.)
    


    # ----------- get the predicted configurations -----------

    # (batch_size, (784/stride**2)*bit_size)
    bit = model.hidden_act
    # (batch_size, (784/stride**2), bit_size)
    bit = bit.reshape(bit.size(0), num_patches, -1)
  
    threshold = args.threshold
    
    binary_bit = np.where(bit < threshold, 0, 1)
    print("binary_bits", binary_bit)
    
    
    
    # -------- solve for configs --------
    
    # Get patches
    # patch = image.squeeze(1).cpu()\
    #     .unfold(1, args.stride, args.stride)\
    #     .unfold(2, args.stride, args.stride)\
    #     .contiguous().view(image.size(0), -1, args.stride*args.stride)
    # print("patches shape: ", patch.shape)
    
    # num_patch, batch_size, patch_size ** 2
    patch = model.unfold(image).view(image.shape[0], model.stride**2, model.num_patches).permute(2,0,1)
    
    # batch_size, num_patch, patch_size ** 2
    patch = patch.permute(1, 0, 2)

    print("patches shape: ", patch.shape)
    
    
    solved_configs = []
    
    for i, p in enumerate(patch[0]):
        # print("patch values", p)
        plt.imshow(p.reshape(7,7), cmap='gray')
        plt.savefig('/u/sissij/projects/satnet_image_completion/mnist_rule/debug/debug_results/' + f'patch_{i}.png')
        
        p[p < 0.5] = 0
        p[p >= 0.5] = 1
        
        plt.imshow(p.reshape(7,7), cmap='gray')
        plt.savefig(os.path.join('/u/sissij/projects/satnet_image_completion/mnist_rule/debug/debug_results/', f'patch{i}_thresh.png'))
        
        # path is from the model folder!
        common_w_l1 = '/u/sissij/projects/satnet_image_completion/output/mnist/model_hrclf_m_200_aux_50_dim_8_stride_7_loss_ce_lr_0.002_seed_123/solver_inference/common_w1.txt'
        
        
        fpuzzle = os.path.join(debug_dir, f"patch{i}.txt")
        print(f"constructing {fpuzzle}")
        with open(fpuzzle, 'w') as fpuz:
            for j in range(2, len(p)+2):
                if p[j-2] > 0:
                    fpuz.write( f'h {j} 0\n' )
                else:
                    fpuz.write( f'h -{j} 0\n' )
        
        status = os.system(f'cat {common_w_l1} >> {fpuzzle}')
        if status != 0:
            print(f"fail to append {common_w_l1} to {fpuzzle}")
            return ""
        fsol = os.path.join(debug_dir, f'patch{i}.sol')
        
        if args.solve:
            print(f"solving {fpuzzle}")
            
            if os.path.exists(fsol):
                os.system(f'rm {fsol}')
                
            status = os.system(f'./mnist_rule/cashwmaxsatcoreplus -m {fpuzzle} -cpu-lim=500 >> {fsol}')
            
            if status != 0:
                print(f"fail to solve {fpuzzle}")
                return ""
        
        solved_config = _get_solved(fsol)
        solved_configs.append(solved_config)
    
    
    
    print("solved configs", solved_configs)
    
    str_bits = []
    for k in range(len(binary_bit[0])):
        print(binary_bit[0][k])
        str_bits.append(''.join([str(list(binary_bit[0][k])[m]) for m in range(len(binary_bit[0][k]))]))
        
    
    result = [(solved_configs[i], str_bits[i]) for i in range(len(solved_configs))]
    print(result)
    
    
    
def _get_solved(f):
    
    with open(f, 'r') as fin:
        for l in fin.read().splitlines():
            if l.startswith('v'):
                result = l.split(' ')

                # 1 truth var + 49 image vars + 50 aux vars then 8 output vars
                solved = result[101:109]
                solved_101_108 = ''.join(['1' if int(ele) > 0 else '0' for ele in solved])
                       
                return solved_101_108

    
    
    
if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main(args, device)