import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CoMNIST(Dataset):
    def __init__(self, root_dir, training=True, device='cuda'):
        super(CoMNIST, self).__init__()
        if training is True:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
        self.ori_image_folder = os.path.join(self.root_dir, 'ori_image')
        self.mask_image_folder = os.path.join(self.root_dir, 'mask_image')
        self.mask_folder = os.path.join(self.root_dir, 'mask')
        self.label_folder = os.path.join(self.root_dir, 'label')

        self.device = device
        self.ori_image_filenames = os.listdir(self.ori_image_folder)
        self.mask_image_filenames = os.listdir(self.mask_image_folder)
        self.mask_filenames = os.listdir(self.mask_folder)
        self.label_filenames = os.listdir(self.label_folder)
        # self.labels = []
        # if training is True:
        #     print('Initializing training dataset...')
        # else:
        #     print('Initializing testing dataset...')
        # for file_name in self.label_filenames:
        #     path = os.path.join(self.label_folder, file_name)
        #     with open(path) as f:
        #         label = torch.tensor(int(f.readlines()[0]))
        #         self.labels.append(label)

    def __len__(self):
        return len(self.ori_image_filenames)

    def __getitem__(self, idx):
        ori_img_path = os.path.join(self.ori_image_folder, self.ori_image_filenames[idx])
        mask_img_path = os.path.join(self.mask_image_folder, self.mask_image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])
        label_path = os.path.join(self.label_folder, self.label_filenames[idx])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])

        ori_image = transform(Image.open(ori_img_path)).to(self.device)
        mask_image = transform(Image.open(mask_img_path)).to(self.device)
        # label = self.labels[idx].to(self.device)
        with open(label_path) as f:
            label = torch.tensor(int(f.readlines()[0])).to(self.device)
        mask = torch.load(mask_path).to(self.device)

        return ori_image, mask_image, mask, label