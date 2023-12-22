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
import itertools
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
    parser.add_argument("--verify_num", type=int, default=200)

    # specify trained model
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--threshold', type=float, default=0.5)

    return parser.parse_args()

def _get_binary_configuration(args, model, image, num_patches):
     # Generate output
    pred = model(image)
    pred = torch.argmax(pred, dim=-1)

    # ----------- get the predicted configurations -----------

    # (batch_size, (784/stride**2)*bit_size)
    bit = model.hidden_act
    # (batch_size, (784/stride**2), bit_size)
    bit = bit.reshape(bit.size(0), num_patches, -1)

    threshold = args.threshold
    binary_bit = np.where(bit.cpu() < threshold, 0,
                        (np.where(bit.cpu() > 1-threshold, 1, None)))
    return binary_bit[0].reshape(128), pred.cpu()

def main(args, device):
    debug_dir = os.path.join(os.curdir, 'mnist_rule','debug',f'layer2_results_{args.threshold}')
    os.makedirs(debug_dir, exist_ok=True)

    # Load the model with corresponding weights (from the model directory!)
    model, dataloader = generate_dist._get_model_dataset(args=args, device=device)
    model.eval()

    num_patches = generate_dist._get_num_patches(args.stride)
    # print("num of patches: ", num_patches)

    # Input one image (16 patch)
    # Run evaluation
    # print("Genearte bits for testing images...")
    loader = tqdm(dataloader)
    counter = 0
    counter_align_gt = 0
    counter_align_pred = 0

    for image, label in loader:

        if counter > args.verify_num:
            break

        image = image.to(device)
        label = label.to(device)
        print("The label of this image is: ", label)

        # Generate output
        round1, pred1 = _get_binary_configuration(args, model, image, num_patches)
        round2, pred2 = _get_binary_configuration(args, model, image, num_patches)
        round3, pred3 = _get_binary_configuration(args, model, image, num_patches)

        # Check array validity
        if (None in round1) or (None in round2) or (None in round3):
            # print("One of the three rounds have None.")
            continue

        if np.all(round1 != round2) or np.all(round2 != round3) or np.all(round3 != round1):
            # print("They are not aligned.")
            continue

        print(f"Collect img{counter}.")

        # save the image
        plt.imshow(image.cpu().reshape(28,28), cmap='gray')
        plt.savefig(os.path.join(debug_dir, f"image{counter}.png"))

        # Write to puzzle file
        fpuzzle = os.path.join(debug_dir, f"img{counter}_config.txt")
        exp_name = os.path.join(args.save_dir, args.dataset,
                                    f'model_{args.model}_m_{args.m}_aux_{args.aux}_dim_{args.hidden_dim}_stride_{args.stride}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}')
        common_w_l2 = os.path.join(exp_name, 'solver_inference', 'common_w2.txt')


        with open(fpuzzle, 'w') as fpuz:

            #------construct one-hot constraints--------#

            # outputs = [n for n in range(180, 190)]
            # pairwise_combinations = list(itertools.combinations(outputs, 2))
            # extra_start = 3000
            # one_hot_weight = 1000

            # for pair in pairwise_combinations:

            #     fpuz.write(f'h {pair[0]} {pair[1]} {extra_start} 0\n')
            #     fpuz.write(f'h -{pair[0]} -{pair[1]} {extra_start} 0\n')
            #     fpuz.write(f'{one_hot_weight} -{extra_start} 0\n')
            #     extra_start += 1

            #--------construct the inputs-----------#
            for j in range(2, len(round1)+2):
                if round1[j-2] > 0:
                    fpuz.write( f'h {j} 0\n' )
                else:
                    fpuz.write( f'h -{j} 0\n' )

        opt_sols = []

        for i in range(10):
            fpuzzle_i = os.path.join(debug_dir, f'img{counter}_p{i}.txt')
            with open(fpuzzle_i, 'w') as file:
                for j in range(10):
                    if i==j:
                        file.write(f"h {180+j} 0\n")
                    else:
                        file.write(f"h -{180+j} 0\n")

            os.system(f"cat {fpuzzle} >> {fpuzzle_i}")
            os.system(f'cat {common_w_l2} >> {fpuzzle_i}')

            fsol_i = os.path.join(debug_dir, f'img{counter}_p{i}.sol')
            if args.solve:
                os.system(f"./mnist_rule/cashwmaxsatcoreplus -m {fpuzzle_i} > {fsol_i}")

            with open(fsol_i, 'r') as sol_file:
                opt_sol = float('inf')
                for l in sol_file.read().splitlines():
                    if l.startswith('o'):
                        opt_sol = int(l.split(' ')[1])
                opt_sols.append(opt_sol)

        print(f"optimal solutions for img{counter}: ", opt_sols)
        print(f"label form image: ", label.item())

        align_w_gt = np.argmin(opt_sols) == label.item()
        if align_w_gt:
            counter_align_gt += 1

        align_w_pred = np.argmin(opt_sols) == pred1.item()
        if align_w_pred:
            counter_align_pred += 1

        print("solved label aligns with the prediction: ", align_w_pred)
        print("solved label is correct: ", align_w_gt)
        print("alignment with pred so far: ", counter_align_pred/(counter+1) * 100)
        print("accuracy so far: ", counter_align_gt/(counter+1) * 100)
        counter += 1


if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    main(args, device)