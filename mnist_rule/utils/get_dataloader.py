from torch.utils.data import DataLoader
from dataset.singlebitDataset import MnistSinglebit
from dataset.multiplebitDataset import MnistMultiplebit
from dataset.comnistDataset import CoMNIST
from torchvision import transforms, datasets

import torch
import torch.nn.functional as F

# Custom transformation: merging k*k grids using average pooling
class Merge:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, img):
        # Ensure the input is a PyTorch tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        # Apply average pooling
        img = F.avg_pool2d(img.unsqueeze(0), kernel_size=self.kernel_size, stride=self.stride).squeeze(0)

        return img
    
def merge_transform(p):
    return transforms.Compose([
        Merge(kernel_size=(p,p), stride=(p,p)),
        transforms.ToPILImage(),  # Convert the tensor back to a PIL image
        transforms.ToTensor()
        ])

def get_tarin_loader(dataset, path, batch_size, device):
    if dataset == 'singlebit':
        dataset = MnistSinglebit(path, device=device)
    elif dataset == 'multiplebit':
        dataset = MnistMultiplebit(path, device=device)
    elif dataset == 'comnist':
        dataset = CoMNIST(path, device=device)
    elif dataset == 'mnist':
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        
        l = 28
        k = 28
        p = l//k
        
        dataset = datasets.MNIST(path, train=True, download=True, transform=merge_transform(p))
    elif dataset == 'fashionmnist': 
        l = 28
        k = 28
        p = l//k
        dataset = datasets.FashionMNIST(path, train=True, download=True, transform=merge_transform(p))
    else:
        raise 'Unknwon dataset'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_test_loader(dataset, path, batch_size, device):
    if dataset == 'singlebit':
        dataset = MnistSinglebit(path, device=device, training=False)
    elif dataset == 'multiplebit':
        dataset = MnistMultiplebit(path, device=device, training=False)
    elif dataset == 'comnist':
        dataset = CoMNIST(path, device=device, training=False)
    elif dataset == 'mnist':
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        
        l = 28
        k = 28
        p = l//k
        
        dataset = datasets.MNIST(path, train=False, download=True, transform=merge_transform(p))
    elif dataset == 'fashionmnist': 
        l = 28
        k = 28
        p = l//k
        dataset = datasets.FashionMNIST(path, train=False, download=True, transform=merge_transform(p))
    else:
        raise 'Unknwon dataset'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
