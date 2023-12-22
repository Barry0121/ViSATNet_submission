import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image


def generate_batches(image, mask_ratio=0.5, return_origin=False):
    mask = torch.rand_like(image)
    mask = torch.where(mask > mask_ratio, 1, 0).int()

    if return_origin is True:
        return mask * image, mask, image
    else:
        return mask * image, mask


def generate_one(path, mask_ratio=0.5, threshold=0.5, return_origin=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    image = Image.open(path)
    image = transform(image)
    # image = torch.where(image > threshold, 1., 0.)
    mask = torch.rand_like(image)
    mask = torch.where(mask > mask_ratio, 1, 0).int()

    if return_origin is True:
        return mask * image, mask, image
    else:
        return mask * image, mask


def generate_from_one_image(ori_path, output_digit_dir, mask_ratio, cnt, repeat,
                                training=True):
    if training is True:
        path = os.path.join(output_digit_dir, 'train')
    else:
        path = os.path.join(output_digit_dir, 'test')

    image_path = os.path.join(path, 'image')
    mask_path = os.path.join(path, 'mask')
    label_path = os.path.join(path, 'label')

    os.makedirs(image_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for i in range(repeat):
        sample, mask, label = generate_one(ori_path, mask_ratio=mask_ratio, return_origin=True)
        torchvision.utils.save_image(sample, image_path + f'/{cnt + i}.png')
        torchvision.utils.save_image(label, label_path + f'/{cnt + i}.png')
        torch.save(mask, mask_path + f'/{cnt + i}.pth')