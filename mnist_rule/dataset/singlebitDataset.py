import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MnistSinglebit(Dataset):
    def __init__(self, root_dir, training=True, device='cuda'):
        super(MnistSinglebit, self).__init__()
        self.root_dir = root_dir
        self.device = device
        if training is True:
            self.image_folder = os.path.join(root_dir, 'train')
            self.mask_folder = os.path.join(root_dir, 'train_mask')
        else:
            self.image_folder = os.path.join(root_dir, 'test')
            self.mask_folder = os.path.join(root_dir, 'test_mask')
        self.image_filenames = os.listdir(self.image_folder)
        self.mask_filenames = os.listdir(self.mask_folder)
        assert len(self.image_filenames) == len(self.mask_filenames)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        image = transform(Image.open(img_path)).to(self.device)
        mask = torch.load(mask_path).to(self.device)

        label = transform(Image.open(os.path.join(self.root_dir, 'origin.png'))).to(self.device)
        label = torch.where(label > 0, 1., 0.)
        return image, label, mask