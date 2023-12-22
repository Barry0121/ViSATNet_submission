import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MnistMultiplebit(Dataset):
    def __init__(self, root_dir, training=True, device='cuda'):
        super(MnistMultiplebit, self).__init__()
        if training is True:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'test')
        self.image_folder = os.path.join(self.root_dir, 'image')
        self.mask_folder = os.path.join(self.root_dir, 'mask')
        self.label_folder = os.path.join(self.root_dir, 'label')

        self.device = device
        self.image_filenames = os.listdir(self.image_folder)
        self.mask_filenames = os.listdir(self.mask_folder)
        self.label_filenames = os.listdir(self.label_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        label_path = os.path.join(self.label_folder, self.label_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])

        # image = torch.load(img_path).to(self.device)
        # label = torch.load(label_path).to(self.device)
        image = transform(Image.open(img_path)).to(self.device)
        label = transform(Image.open(label_path)).to(self.device)
        mask = torch.load(mask_path).to(self.device)

        return image, label, mask