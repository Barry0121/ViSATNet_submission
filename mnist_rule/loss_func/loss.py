import torch


def dice_loss(pred, target):
    smooth = 1e-7
    intersection = torch.sum(pred * target, dim=1)
    union = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
    loss = 1. - (2. * intersection + smooth) / (union + smooth)
    return loss.mean()