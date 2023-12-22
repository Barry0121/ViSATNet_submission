import os

def construct_config_constraints(config, f):
    with open(f, 'w') as fout:
        start = 101
        for i, ele in enumerate(config):
            if int(ele) == 0:
                fout.write(f'h -{start+i} 0\n')
            else:
                fout.write(f'h {start+i} 0\n')


def main():
    debug_dir = '/u/sissij/projects/satnet_image_completion/mnist_rule/debug_results'
    configs = ['00000000','00000001', '00000010', '00000011', '00000101', '00001000', '00010000', '00010010',
                '00011000', '00100000', '00100001', '01000000', '01000001', '01000010', '01000100', '01000101',
                '01100000', '01111000', '10000000', '10100000', '10100010', '11000000', '11100000']
    for config in configs:
        f = os.path.join(debug_dir, f'patch_0_{config}.txt')
        construct_config_constraints(config, f)
        os.system(f'cat /u/sissij/projects/satnet_image_completion/mnist_rule/debug/debug_results/patch0.txt >> {f}')

        fsol = os.path.join(debug_dir, f'patch_0_{config}.sol')
        
        if os.path.exists(fsol):

            os.system(f'rm {fsol}')

        os.system(f'./mnist_rule/cashwmaxsatcoreplus -m {f} -cpu-lim=500 >> {fsol}')
    
if __name__ == "__main__":
    main()
