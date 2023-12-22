import random
from tqdm import tqdm

from random_mask import *
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--digit', type=str)

    return parser.parse_args()


def set_up(mask_ratio):
    labels = [str(i) for i in range(10)]
    for label in labels:
        setup_one_digit(mask_ratio, label)


def setup_one_digit(mask_ratio, digit):
    # generate origin image
    print(f"Generating training and testing samples for digit {digit}")
    data_dir = 'data/mnist'
    image_dir = os.path.join(data_dir, digit)
    output_dir = 'data/multiplebit'

    output_digit_dir = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}')
    os.makedirs(output_digit_dir, exist_ok=True)

    img_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]
    train_cnt = 0
    test_cnt = 0
    train_repeat = 2
    test_repeat = 2
    train_base = 3000
    test_base = 500
    img_path_id = list(range(len(img_paths)))

    # training samples
    random.shuffle(img_path_id)
    for i in tqdm(range(train_base)):
        idx = img_path_id[i]
        generate_from_one_image(img_paths[idx], output_digit_dir, mask_ratio, train_cnt, train_repeat)
        train_cnt += train_repeat
    print('Training sample generation completed')

    # testing samples
    random.shuffle(img_path_id)
    for i in tqdm(range(test_base)):
        idx = img_path_id[i]
        generate_from_one_image(img_paths[idx], output_digit_dir, mask_ratio, test_cnt, test_repeat, training=False)
        test_cnt += test_repeat
    print('Testing sample generation completed')


if __name__ == '__main__':
    args = get_args()
    if args.digit is None:
        set_up(args.mask_ratio)
    else:
        setup_one_digit(args.mask_ratio, args.digit)