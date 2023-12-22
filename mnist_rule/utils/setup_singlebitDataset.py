import os
import shutil
from tqdm import tqdm
import torch
import torchvision

from utils.random_mask import generate_one
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float)

    return parser.parse_args()


def set_up(mask_ratio):
    labels = [str(i) for i in range(10)]
    for label in labels:
        setup_one_digit(mask_ratio, label)


def setup_one_digit(mask_ratio, digit):
    # generate origin image
    data_dir = 'data/mnist'
    image_dir = os.path.join(data_dir, 'image')
    label_dir = os.path.join(data_dir, 'label')
    output_dir = 'data/singlebit'

    output_digit_dir = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}')
    os.makedirs(output_digit_dir, exist_ok=True)

    txt_paths = [os.path.join(label_dir, txt_name) for txt_name in os.listdir(label_dir) if txt_name.endswith('.txt')]

    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            txt_content = f.read()
        if digit in txt_content:
            image_name = os.path.splitext(os.path.basename(txt_path))[0] + '.png'
            image_path = os.path.join(image_dir, image_name)

            shutil.copy(image_path, os.path.join(output_digit_dir, 'origin.png'))

            break

    # generate samples
    img_path = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}', 'origin.png')
    save_train_path = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}', 'train')
    save_test_path = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}', 'test')
    save_train_mask_path = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}', 'train_mask')
    save_test_mask_path = os.path.join(output_dir, f'digit_{digit}_mask_{mask_ratio * 100}', 'test_mask')
    train_size, test_size = 5000, 1000
    os.makedirs(save_train_path, exist_ok=True)
    os.makedirs(save_test_path, exist_ok=True)
    os.makedirs(save_train_mask_path, exist_ok=True)
    os.makedirs(save_test_mask_path, exist_ok=True)

    pbar = tqdm(range(train_size))
    for i in pbar:
        sample, mask = generate_one(img_path, mask_ratio=mask_ratio)
        torchvision.utils.save_image(sample, save_train_path+f'/{i}.png')
        torch.save(mask, save_train_mask_path+f'/{i}.pth')
        pbar.set_description(f"Generating training samples for digit {digit}")

    pbar = tqdm(range(test_size))
    for i in pbar:
        sample, mask = generate_one(img_path, mask_ratio=mask_ratio)
        torchvision.utils.save_image(sample, save_test_path+f'/{i}.png')
        torch.save(mask, save_test_mask_path + f'/{i}.pth')
        pbar.set_description(f"Generating testing samples for digit {digit}")


if __name__ == '__main__':
    args = get_args()
    set_up(args.mask_ratio)