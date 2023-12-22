'''
This file turns a weighted maxsat constraint file into
a pbo file (to be solved by gurobi)
'''
import argparse

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int)
    parser.add_argument('--patch_id', type=int, default=0)

    args = parser.parse_args()
    return args

def R(fpath):
    with open(fpath, "r") as fin:
        return fin.read().splitlines()

def print_pbo(pbo_cons,g,filename):
    goal = "min: "

    for ele in g:
        (w, var) = ele
        goal = goal + ('+' if w > 0 else '') +str(w) + ' ' + var + ' '
    
    goal = goal + ';'

    with open(filename, 'w') as fout: 
        fout.write(goal+'\n')

        for p in pbo_cons:
            ls, w = p
            str_reps = [('+' if w > 0 else '') + str(w) + ' ' + name for (w,name) in ls]
            fout.write(f"{' '.join(str_reps) } >= {w};"+'\n')

    

def max_to_pbo(maxsat_file, pbo_file):
    # turn maxsat into pbo, but with partial assignment

    hard_cons = []
    soft_cons = []

    for l in R(maxsat_file):
        if l.startswith('h'):
            vs = list(map(int, l[1:-1].split()))
            hard_cons.append(vs)
        else:
            vs = list(map(int, l[:-1].split()))
            soft_cons.append( (vs[0], vs[1:]) )


    pbo_cons = []
    g =[]
    
    ''' hard constraint example:
    h -1 -2 66 0 
    => 
    -1 x1 -1 x2 +1 66 >= -1'''

    for h in hard_cons:
        
        ct = len(list(filter(lambda x : x < 0, h)))
        xs = [f"x{abs(x)}" for x in h]
        ks = [abs(x)//x for x in h]
        res = list(zip(ks, xs))

        pbo_cons.append((res, 1-ct))


    ''' soft constraint example:
    12622 -66 0
    =>
    let z0 == -66 
    then in the objective, we put + 12622 z0
    instead of 12622*(-x66)'''

    for i, v in enumerate(soft_cons):
        (w, s) = v

        ct = len(list(filter(lambda x : x < 0, s)))
        xs = [f"x{abs(x)}" for x in s]
        ks = [abs(x)//x * w * -1 for x in s]
        res = list(zip(ks, xs))
        # we have +12622 x66 here (to be minimized in the objective)
        g.append(res[0])


    # fix the partial assignment bits
    # p_sol = construct_maxsat_assignment(pos)
    # p_mask = construct_lowlevel_mask(mask)

    # for x in p_sol:
    #     v = abs(x)
    #     if p_mask[v-2]:
    #         if x > 0:
    #             pbo_cons.append( ([(1, f'x{v}')], 1) )
    #         else:
    #             pbo_cons.append( ([(-1, f'x{v}')], 0) )


    print_pbo(pbo_cons,g,pbo_file)
    
    
def main():
    args = get_args()
    if args.layer == 1:
        
        max_to_pbo(f"patch_{args.patch_id}.txt", f"patch_{args.patch_id}.opb")
    elif args.layer == 2:
        max_to_pbo('test_image_puzzle.txt', 'test_image_puzzle.opb')
    else:
        print("layer must be either 1 or 2!")
    
    
    
if __name__ == "__main__":
    main()

