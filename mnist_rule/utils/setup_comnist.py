import os
import random
from shutil import copyfile
from tqdm import tqdm
import argparse
from random_mask import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float)
    return parser.parse_args()

def main():
    """
    set up comnist dataset
    ori_image: 0-1 binary image
    """
    args = get_args()
    random.seed(42)

    data_dir = "./data"
    image_dir = os.path.join(data_dir, 'mnist', "image")
    label_dir = os.path.join(data_dir, 'mnist', "label")

    train_dir = os.path.join(data_dir, 'comnist', "train")
    test_dir = os.path.join(data_dir, 'comnist', "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # shuffle images and labels
    indices = list(range(len(os.listdir(label_dir))))
    random.shuffle(indices)
    train_indices = indices[:3000*10]  # 3000 samples per class, 10 classes in total
    test_indices = indices[-500*10:]  # 500 samples per class, 10 classes in total

    # copy train images and labels
    train_ori_image_dir = os.path.join(train_dir, "ori_image")
    train_label_dir = os.path.join(train_dir, "label")
    train_mask_image_dir = os.path.join(train_dir, 'mask_image')
    train_mask_dir = os.path.join(train_dir, 'mask')

    os.makedirs(train_ori_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(train_mask_image_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)

    for i, index in enumerate(tqdm(train_indices)):
        src_image_path = os.path.join(image_dir, f"{index}.png")
        src_label_path = os.path.join(label_dir, f"{index}.txt")
        dst_image_path = os.path.join(train_ori_image_dir, f"{i}.png")
        dst_label_path = os.path.join(train_label_dir, f"{i}.txt")
        dst_mask_path = os.path.join(train_mask_dir, f'{i}.pth')
        dst_mask_image_path = os.path.join(train_mask_image_dir, f'{i}.png')
        mask_image, mask, ori_image = generate_one(src_image_path, mask_ratio=args.mask_ratio, return_origin=True)
        copyfile(src_label_path, dst_label_path)
        torchvision.utils.save_image(mask_image,  dst_mask_image_path)
        torchvision.utils.save_image(ori_image, dst_image_path)
        torch.save(mask, dst_mask_path)

    # copy test images and labels
    test_ori_image_dir = os.path.join(test_dir, "ori_image")
    test_label_dir = os.path.join(test_dir, "label")
    test_mask_image_dir = os.path.join(test_dir, 'mask_image')
    test_mask_dir = os.path.join(test_dir, 'mask')

    os.makedirs(test_ori_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(test_mask_image_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)


    for i, index in enumerate(tqdm(test_indices)):
        src_image_path = os.path.join(image_dir, f"{index}.png")
        src_label_path = os.path.join(label_dir, f"{index}.txt")
        dst_image_path = os.path.join(test_ori_image_dir, f"{i}.png")
        dst_label_path = os.path.join(test_label_dir, f"{i}.txt")
        dst_mask_path = os.path.join(test_mask_dir, f'{i}.pth')
        dst_mask_image_path = os.path.join(test_mask_image_dir, f'{i}.png')
        mask_image, mask, ori_image = generate_one(src_image_path, mask_ratio=args.mask_ratio, return_origin=True)
        copyfile(src_label_path, dst_label_path)
        torchvision.utils.save_image(mask_image,  dst_mask_image_path)
        torchvision.utils.save_image(ori_image, dst_image_path)
        torch.save(mask, dst_mask_path)

main()