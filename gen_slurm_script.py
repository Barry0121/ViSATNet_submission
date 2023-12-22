# generates a slurm job script

import argparse
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    # original:
    # parser.add_argument('--model', type=str, default='hrgen')
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--stride', type=int, default=7)
    return parser.parse_args()


def main():
    args = get_args()
    # get current date yyyy_mm_dd
    date = datetime.today().strftime('%Y_%m_%d')
    # get all model parameters
    
    job_content = f"#!/bin/bash\n\
\n\
#SBATCH --job-name={args.dataset}_{args.model}_{date}_dim_{args.hidden_dim}_stride_{args.stride}\n\
#SBATCH --partition=biggpunodes\n\
#SBATCH --mem=4G\n\
#SBATCH --time=72:00:00\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks=1\n\
#SBATCH --cpus-per-task=2\n\
#SBATCH --output={date}_{args.dataset}_{args.model}_m_{args.m}_aux_{args.aux}_dim_{args.hidden_dim}_stride_{args.stride}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}.txt\n\
#SBATCH --gres=gpu:1\n\
\n"
    
    with open(f"{date}_{args.dataset}_satnet2_hd_{args.hidden_dim}_stride_{args.stride}_aux_{args.aux}.sh","w") as fout:
        fout.write(job_content)
        fout.write("source /u/sissij/environments/image_completion/bin/activate\n")
        fout.write("cd /u/sissij/projects/satnet_image_completion\n")
        fout.write(f"python3 mnist_rule/train_eval.py --loss {args.loss} --model {args.model} --hidden_dim {args.hidden_dim} --stride {args.stride} --aux {args.aux} --epochs {args.epochs} --dataset {args.dataset}\n")
        fout.write("deactivate\n")
        
        
        
if __name__ == "__main__":
    main()