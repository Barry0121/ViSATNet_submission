'''
This is a file that converts a given C matrix to weighted maxsat constraints
i.e. "common constraints"
'''

import argparse
import torch
import numpy as np
import os

INF_W = 2 ** 30

name_to_ids = {}

class Lit:
    def __init__(self, pos, name):
        self.pos = pos
        self.name = name
    
    def __repr__(self) -> str:
        return self.name if self.pos else f'(not {self.name})'

class Clause:
    def __init__(self, lits=[]):
        self.lits = lits

    def add_lit(self, pos, name):
        self.lits.append( (pos, name))
    
    def __repr__(self) -> str:
        return f'{self.lits}'

class SATInstance:
    def __init__(self, cls=[]):
        self.cls = cls
    def add_clause(self, w, c):
        self.cls.append( (w, c) )
    def __repr__(self) -> str:
        return "SatInstance with {0} clauses:\n{1}".format(len(self.cls),'\n'.join([f'{w} {c}' for w,c in self.cls]) )

    def save_as_maxsat(self, fpath):
        global name_to_ids
        v = len(name_to_ids)
        for _, c in self.cls:
            for lit in c.lits:
                if lit.name not in name_to_ids:
                    v += 1
                    name_to_ids[lit.name] = v

        with open(fpath, 'w') as fout:
            # output the name to id mapping (optional)
            # for x in name_to_ids:
            #     fout.write(f'c {x} ==> {name_to_ids[x]}\n')

            # output weighted clauses

            # truth vector must be positive
            fout.write("h 1 0\n")
            for w, c in self.cls:
                fout.write( 'h' if w == INF_W else f'{w}')
                for lit in c.lits:
                    fout.write( f' {"" if lit.pos else "-"}{name_to_ids[lit.name]}')
                fout.write(' 0\n')

def encode_eq(s, vi, vj, w):
    eq_name = f'Eq_{vi}_{vj}'

    if w == 0:
        pass
    elif w > 0:
        s.add_clause(INF_W, Clause([Lit(False, vi), Lit(True, vj), Lit(False, eq_name)]))
        s.add_clause(INF_W, Clause([Lit(True, vi), Lit(False, vj), Lit(False, eq_name)]))
        s.add_clause(w, Clause([Lit(True, eq_name)]))
        pass
    elif w < 0:
        s.add_clause(INF_W, Clause([Lit(False, vi), Lit(False, vj), Lit(True, eq_name)]))
        s.add_clause(INF_W, Clause([Lit(True, vi), Lit(True, vj), Lit(True, eq_name)]))
        s.add_clause(-w, Clause([Lit(False, eq_name)]))


def interpret_C_matrix(cmatrix, aux, multiplier):    
    s = (cmatrix.transpose(0, 1) - cmatrix).sum()
    if s > 1e-2:
        print('cmatrix should be symmetric')
        exit()

    n = cmatrix.size(0)
    names = [ f'v{i}' for i in range(n - aux) ]
    names.extend([f'a{i}' for i in range(aux)])

    v = len(name_to_ids)
    for x in names:
        v += 1
        name_to_ids[x] = v

    sat = SATInstance()

    for i in range(n):
        for j in range(i):
            w = -1 * cmatrix[i][j].item()
            
            # for C_0426.pt
            # w = int (0.01 * w ) 
            # for C_sparse
            # w = int (100*w)
            
            # for mnist layer 1
            # w = int (0.05 * w)
            # for mnist layer 2
            w = int(multiplier * w)
            encode_eq(sat, names[i], names[j], w)

    return sat
 

def get_args():
    parser = argparse.ArgumentParser()
    # specification
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--stack_i', type=str, default='1')
    # parser.add_argument('--fout', type=str, default='common_w1.txt')

    # specify trained model
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--multiplier', type=float, default=1.0)

    return parser.parse_args()


def _get_model_path(args):
    """Return the target model weight path given arguments.
    If the path doesn't exist, make it exist."""
    exp_name = os.path.join(args.save_dir, args.dataset,
                            (f"model_{args.model}_m_{args.m}_aux_{args.aux}"
                             f"_dim_{args.hidden_dim}_stride_{args.stride}"
                             f"_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}"))
    os.makedirs(exp_name, exist_ok=True)
    return exp_name

def main():

    args = get_args()
    if args.stack_i not in ['1', '2']:
        print("!!!Warning!!! stack_i must be 1 or 2!")
        return
    
    exp_path = _get_model_path(args)
    weight_file = os.path.join(exp_path, "model/c_matrices.pt")
    
    
    # cmatrix = torch.from_numpy(np.load(args.cmatrix))
    weights = torch.load(weight_file)
    cmatrix = weights[f'conv{args.stack_i}.sat.C']
    print(f"cmatrix loaded from {weight_file}\nconverting the weights of layer {args.stack_i} ............")
    print(f"cmatrix first line \n {cmatrix[0]}")
    print("shape of the matrix: ", cmatrix.shape)
          
    # convert conv_i weights to weighted maxsat
    sat = interpret_C_matrix(cmatrix, args.aux, args.multiplier)
    
    save_path = os.path.join(exp_path, "solver_inference")
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, f"common_w{args.stack_i}.txt")
    
    sat.save_as_maxsat(save_file)
    print(f"weighted maxsat file saved at {save_file}")
    


if __name__ == '__main__':
    main()