import os
import matplotlib.pyplot as plt
import torch
from train_eval import get_args
from models import *
from utils.random_mask import generate_one
import random


def get_one(model, img_path, mask_ratio, class_label=None, device='cpu'):
    image, mask, label = generate_one(img_path, mask_ratio=mask_ratio, return_origin=True)
    mask_image = image
    image, mask, label = image.to(device), mask.to(device), label.to(device)
    if class_label is None:
        image = model(image.view(1, -1), mask.view(1, -1)).view(mask.shape)
    else:
        class_label = class_label.to(device)
        image = model(image.view(1, -1), mask.view(1, -1), class_label.view(1)).view(mask.shape)
    return image.cpu().detach().numpy(), mask_image.cpu().detach().numpy(), label.cpu().detach().numpy()


def draw_image(label):
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.dataset == 'multiplebit':
        model = Generator(m=args.m, aux=args.aux).to(device)

        path = os.path.join(args.save_dir, args.dataset, f'digit_{label}_mask_{args.mask_ratio * 100}',
                            f'm_{args.m}_aux_{args.aux}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}', 'model',
                            'best_model.pt')
        model.load_state_dict(torch.load(path))
    elif args.dataset == 'comnist':
        model = CoGenerator(m=args.m, aux=args.aux).to(device)
        path = os.path.join(args.save_dir, args.dataset,
                            f'model_{args.model}_m_{args.m}_aux_{args.aux}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}', 'model',
                            'best_model.pt')
        model.load_state_dict(torch.load(path))
    else:
        raise 'Unknown dataset'
    # img_path = os.path.join(args.root_dir, args.dataset, f'digit_{label}_mask_{args.mask_ratio * 100}', 'test',
    #                         'label')
    img_path = os.path.join(args.root_dir, 'multiplebit', f'digit_{label}_mask_{args.mask_ratio * 100}', 'test',
                                                    'label')
    img_name = random.choice(os.listdir(img_path))
    img_path = os.path.join(img_path, img_name)
    if args.dataset == 'comnist':
        image, mask, ori = get_one(model, img_path, args.mask_ratio, torch.tensor(int(label)), device)
    else:
        image, mask, ori = get_one(model, img_path, args.mask_ratio, device=device)

    axes[0].imshow(ori.reshape(ori.shape[1], ori.shape[2], -1))
    axes[1].imshow(mask.reshape(mask.shape[1], mask.shape[2], -1))
    axes[2].imshow(image.reshape(image.shape[1], image.shape[2], -1))

    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    axes[0].set_title('origin')
    axes[1].set_title('mask')
    axes[2].set_title('prediction')
    plt.savefig(f'digit{label}.png', bbox_inches='tight')


if __name__ == '__main__':
    fig, axes = plt.subplots(ncols=3)
    for label in range(10):
        draw_image(label)