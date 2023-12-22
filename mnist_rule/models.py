import torch
from torch import nn
import torch.nn.functional as F
from satnet import SATNet
import torch.nn.parallel as parallel


class Generator(nn.Module):
    # completion based on training images only (without label given)
    def __init__(self, n=784, aux=40, m=100):
        super(Generator, self).__init__()
        self.sat = SATNet(n, m, aux)

    def forward(self, x_in, mask):
        out = self.sat(x_in, mask)
        del x_in, mask
        return out


class Classifier(nn.Module):
    def __init__(self, n=14*14, num_class=10, aux=40, m=100):
        super(Classifier, self).__init__()
        self.sat = SATNet(n + num_class, m, aux)
        self.num_class = num_class

    def forward(self, x_in):
        device = x_in.device
        mask = torch.ones_like(x_in, device=device, dtype=torch.int)
        mask = torch.cat((mask, torch.zeros(
            (x_in.shape[0], self.num_class), device=device, dtype=torch.int)), dim=-1)
        input = torch.cat((x_in, torch.zeros(
            (x_in.shape[0], self.num_class), device=device, dtype=torch.int)), dim=-1)
        out = self.sat(input, mask)
        del x_in, mask, input
        return out[..., -self.num_class:]


class CoClassifier(nn.Module):
    def __init__(self, n=784, num_class=10, aux=40, m=100):
        super(CoClassifier, self).__init__()
        self.sat = SATNet(n + num_class, m, aux)
        self.num_class = num_class

    def forward(self, x_in, mask):
        device = x_in.device
        mask = torch.cat((mask, torch.zeros(
            (x_in.shape[0], self.num_class), device=device, dtype=torch.int)), dim=-1)
        input = torch.cat((x_in, torch.zeros(
            (x_in.shape[0], self.num_class), device=device, dtype=torch.int)), dim=-1)
        out = self.sat(input, mask)
        del x_in, mask, input

        # return 1. completed image, 2. predicted label
        return out[..., :-self.num_class], out[..., -self.num_class:]


class CoGenerator(nn.Module):
    # complete image with the true label given

    def __init__(self, n=784, num_class=10, m=100, aux=40):
        super(CoGenerator, self).__init__()
        self.sat = SATNet(n + num_class, m, aux)
        self.num_class = num_class

    def forward(self, x_in, mask, class_label):
        device = x_in.device
        mask = torch.cat((mask, torch.ones(
            (x_in.shape[0], self.num_class), device=device, dtype=torch.int)), dim=-1)
        class_var = F.one_hot(
            class_label, num_classes=self.num_class)  # b, num_class
        input = torch.cat((x_in, class_var), dim=-1)
        out = self.sat(input, mask)
        return out[..., :-self.num_class]


class SATConv(nn.Module):
    # take 3D input (image, where first dimension =1) or 2D input
    # x_in: b x in_n
    # x_out: b x out_n

    def __init__(self, in_n, out_n, m=200, aux=0):
        super(SATConv, self).__init__()
        self.sat = SATNet(in_n + out_n, m, aux)
        self.out_n = out_n

    def forward(self, x_in):
        if x_in.dim() == 3 and x_in.shape[0] == 1:
            x_in = x_in.squeeze(0)
        device = x_in.device
        mask = torch.cat((torch.ones_like(x_in, device=device, dtype=torch.int),
                          torch.zeros((x_in.shape[0], self.out_n), device=device, dtype=torch.int)), dim=-1)
        x_in = torch.cat((x_in, torch.zeros(
            (x_in.shape[0], self.out_n), device=device, dtype=torch.float)), dim=-1)
        out = self.sat(x_in, mask)
        return out[:, -self.out_n:]


class SATConvTranspose(nn.Module):
    def __init__(self, in_n, out_n, m=200, aux=50):
        super(SATConvTranspose, self).__init__()
        self.sat = SATNet(in_n + out_n, m, aux)
        self.out_n = out_n

    def forward(self, x_in, mask_img=None, mask=None):
        device = x_in.device
        if mask is None:
            mask = torch.zeros(
                (x_in.shape[0], self.out_n), device=device, dtype=torch.int)
        mask = torch.cat((mask, torch.ones_like(
            x_in, device=device, dtype=torch.int)), dim=-1)
        if mask_img is None:
            x_in = torch.cat((torch.zeros(
                (x_in.shape[0], self.out_n), device=device, dtype=torch.float), x_in), dim=-1)
        else:
            x_in = torch.cat((mask_img, x_in), dim=-1)
        out = self.sat(x_in, mask)
        return out[:, :self.out_n]


class HierarchicalClassifier(nn.Module):
    """
    :param num_class: int, number of target classes
    :param m:
    :param aux:
    :param hidden_dim: int, number of bits in the resulting encoding
    :param stride: int, size of the patch
    """

    def __init__(self, num_class=10, m=200, aux=50, hidden_dim=10, stride=7):
        super(HierarchicalClassifier, self).__init__()
        # Variables
        self.stride, self.hidden_dim = stride, hidden_dim
        self.num_patches = 784 // (self.stride ** 2)
        self.m, self.aux = m, aux

        # Save layer 1 output bits
        self.hidden_act = None

        # First stack of SATNet
        self.conv1 = SATConv(stride**2, hidden_dim, m=m, aux=aux)

        # Second stack of SATNet (16 patches)
        self.conv2 = SATConv(((28 - stride) // stride + 1)
                             ** 2 * hidden_dim, num_class, m=m, aux=aux)

        # Unfolding layer to extract patches
        self.unfold = nn.Unfold(kernel_size=(self.stride, self.stride),
                                stride=(self.stride, self.stride))

    def forward(self, x):
        """
        # Example code for 2 imgs with size=(16,16), stride=4
        # imgs.shape == [2, 16, 16]
        # imgs.unfold(-1, 4, 16).unfold(1, 4, 4).squeeze().permute(0, 1, 3, 2)
        Note: .unfold([where you want to apply the unfolding], [size of the step (kernel size)], [stride of the operation])
        Reference: https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/11

        # or

        # unfold = nn.Unfold(kernel_size=(4, 4), dilation=1, padding=0, stride=(4, 4))
        # unfold(imgs).view(2, 16, 16).permute(0, 2, 1)
        Note: unfold(imgs).view([batchsize], [total patch size], [number of patches])
        """

        # Create patches, shape=(num_patches, batchsize, patch_size**2)
        x = self.unfold(x).view(x.shape[0], self.stride**2, self.num_patches).permute(2,0,1)

        # Forward pass over first SATNet by iterating over patches, output shape=(num_patches, b, hidden_dim)
        y = torch.empty(
            (self.num_patches, x.shape[1], self.hidden_dim), device=x.device)
        for i in range(x.shape[0]):
            y[i] = self.conv1(x[i])

        # Return output back to batch-first format, shape=(batchsize, num_patches * hidden_dim)
        y = y.permute(1, 0, 2).reshape(x.shape[1], -1)

        # Save hidden bits by patches
        self.hidden_act = y

        # Final prediction output, shape=(batchsize, num_class)
        y = self.conv2(y)
        return y


class ThreeLayersHierarchicalClassifier(nn.Module):
    def __init__(self, num_class=10, m=200, aux=50, hidden_dim=(20, 10), stride=7):
        super(ThreeLayersHierarchicalClassifier, self).__init__()
        # first stack of SATNet
        self.conv1 = SATConv(49, hidden_dim[0], m=m, aux=aux)
        # second stack of SATNet (16 patches)
        # conv. nn. calculate count of patch
        self.conv2 = SATConv(((28 - 7) // stride + 1) **
                             2 * hidden_dim[0], hidden_dim[1], m=m, aux=aux)
        self.conv3 = SATConv(hidden_dim[1], num_class, m=m, aux=aux)
        # self.conv2 = SATConv(36 * hidden_dim, num_class)
        self.stride, self.hidden_dim = stride, hidden_dim

    def forward(self, x):
        b = x.shape[0]
        if x.dim() == 3:
            x = x.squeeze(1)
        # num_patches = ((x.shape[1] - 7) // self.stride + 1) ** 2
        # fix num_patches to 16
        num_patches = 16
        # x = x.unfold(1, 7, self.stride).unfold(2, 7, self.stride).reshape(x.shape[0], num_patches, 49).permute(1, 0, 2) # 16, b, 49

        # gray scale, change 28*28 to 16 patches of 7*7
        x = x.reshape([b, 784])
        x = x.unfold(-1, 49, 49).squeeze(1).permute(1, 0, 2)

        # 16, b, hidden_dim
        y = torch.empty(
            (num_patches, x.shape[1], self.hidden_dim[0]), device=x.device)
        for i in range(x.shape[0]):
            y[i] = self.conv1(x[i])

        # x = list(torch.split(x, split_size_or_sections=1, dim=0)) # list(b, 49) * 16
        # func_list = [self.conv1] * len(x)
        # y = parallel.parallel_apply(func_list, x)
        # y = torch.stack(y, dim=0)
        # print(y.shape)
        y = y.permute(1, 0, 2).reshape(b, -1)  # b, 16*hidden_dim
        y = self.conv2(y)  # b, 10
        # print(y.shape)

        y = self.conv3(y)  # b, num_class
        # print(y.shape)
        return y


class HierarchicalGenerator(nn.Module):
    # without true label given

    def __init__(self, num_class=10, num_patches=16, m=200, aux=50, hidden_dim=30):
        super(HierarchicalGenerator, self).__init__()
        self.transconv1 = SATConvTranspose(
            num_class + 16, hidden_dim, m=m, aux=aux)
        self.transconv2 = SATConvTranspose(hidden_dim, 49, m=m, aux=aux)
        # self.transconv = SATConvTranspose(num_class + 16, 49, m=m, aux=aux)
        self.hidden_dim, self.num_class, self.num_patches = hidden_dim, num_class, num_patches

    def forward(self, img, mask, label):
        b = label.shape[0]
        num_patches = self.num_patches
        label = F.one_hot(label, num_classes=self.num_class)
        img, mask = img.squeeze(1), mask.squeeze(1)

        img = img.unfold(1, 7, 7).unfold(2, 7, 7).reshape(
            img.shape[0], num_patches, 49).permute(1, 0, 2)  # 16, b, 49
        mask = mask.unfold(1, 7, 7).unfold(2, 7, 7).reshape(
            mask.shape[0], num_patches, 49).permute(1, 0, 2)
        y = torch.empty((num_patches, b, 49), device=label.device)
        for i in range(num_patches):
            pos_embedding = torch.zeros(
                (b, 16), device=label.device, dtype=torch.int)
            pos_embedding[:, i] = 1
            intermediate = self.transconv1(
                torch.cat((label, pos_embedding), dim=-1))  # b, hidden_dim
            y[i] = self.transconv2(intermediate, img[i], mask[i])  # b, 49
            # y[i] = self.transconv(torch.cat((label, pos_embedding), dim=-1), img[i], mask[i])
        y = y.permute(1, 0, 2).reshape(b, int(num_patches ** 0.5),
                                       int(num_patches ** 0.5), 7, 7).transpose(2, 3).reshape(b, 28, 28)

        return y.unsqueeze(1)
