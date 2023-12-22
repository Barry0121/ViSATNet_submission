import numpy as np
from models import *
from utils.get_dataloader import get_tarin_loader, get_test_loader
from utils.random_mask import generate_batches
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import argparse
from loss_func.loss import dice_loss
from time import time
from utils.setup_singlebitDataset import setup_one_digit

# import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--digit', type=str)
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    # original:
    # parser.add_argument('--model', type=str, default='hrgen')
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--hidden_dim2', type=int, default=10)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--load_checkpoint', type=bool,
                        default=True)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args, device):
    # initialize dataset
    if args.dataset in ['multiplebit', 'singlebit']:
        sub_dir = f'digit_{args.digit}_mask_{args.mask_ratio * 100}'
        if not os.path.exists(os.path.join(args.root_dir, args.dataset, sub_dir)):
            setup_one_digit(args.mask_ratio, args.digit)
        train_loader = get_tarin_loader(args.dataset, os.path.join(args.root_dir, args.dataset, sub_dir), args.batch_size,
                                        device=device)
        test_loader = get_test_loader(args.dataset, os.path.join(
            args.root_dir, args.dataset, sub_dir), args.batch_size, device=device)

        exp_name = os.path.join(args.save_dir, args.dataset, sub_dir,
                                f'm_{args.m}_aux_{args.aux}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}')

        os.makedirs(exp_name, exist_ok=True)
    elif args.dataset in ['mnist', 'comnist', 'fashionmnist']:
        train_loader = get_tarin_loader(args.dataset, os.path.join(args.root_dir, args.dataset),
                                        args.batch_size,
                                        device=device)
        test_loader = get_test_loader(args.dataset, os.path.join(args.root_dir, args.dataset), args.batch_size,
                                      device=device)

        exp_name = os.path.join(args.save_dir, args.dataset,
                                f'model_{args.model}_m_{args.m}_aux_{args.aux}_dim_{args.hidden_dim}_stride_{args.stride}_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}')
        os.makedirs(exp_name, exist_ok=True)
    # initialize model
    if args.dataset in ['multiplebit', 'singlebit']:
        model = Generator(m=args.m, aux=args.aux).to(device)
    elif args.dataset == 'mnist' or args.dataset == 'fashionmnist':

        if args.model == 'clf':
            model = Classifier(m=args.m, aux=args.aux).to(device)
        elif args.model == 'hrclf':
            model = HierarchicalClassifier(
                m=args.m, aux=args.aux, stride=args.stride, hidden_dim=args.hidden_dim).to(device)
        elif args.model == 'hrclf3':
            if args.hidden_dim2 is None:
                raise AttributeError(
                    "The second hidden dimension isn't defined. ")
            model = ThreeLayersHierarchicalClassifier(m=args.m, aux=args.aux, stride=args.stride, hidden_dim=(
                args.hidden_dim, args.hidden_dim2)).to(device)
        elif args.model == 'hrgen':
            model = HierarchicalGenerator(
                m=args.m, aux=args.aux, hidden_dim=args.hidden_dim).to(device)
    elif args.dataset == 'comnist':
        if args.model == 'coclf':
            model = CoClassifier(m=args.m, aux=args.aux).to(device)
        elif args.model == 'cogen':
            model = CoGenerator(m=args.m, aux=args.aux).to(device)
    print("Model loaded!")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.5)

    # Load checkpoint weights
    best_model_path = os.path.join(exp_name, 'model', 'best_model.pt')
    checkpoint_exist = bool(args.load_checkpoint and (
        os.path.exists(best_model_path)))
    if checkpoint_exist:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    begin = time()
    model.train()

    if args.dataset in ['multiplebit', 'singlebit']:
        print(f'Training on digit {args.digit}')
    best_acc = 0

    # Start from 0 when no checkpoint is loaded.
    if not checkpoint_exist:
        epoch = 0
    else:
        epoch = checkpoint['epoch']

    while epoch < args.epochs:
        train_loss = 0
        num_train = 0
        pbar = tqdm(train_loader)
        for samples in pbar:
            if args.dataset in ['multiplebit', 'singlebit']:
                image, label, mask = samples
                image, label, mask = image.view(
                    image.shape[0], -1), label.view(label.shape[0], -1), mask.view(mask.shape[0], -1)
                pred = model(image, mask)
            elif args.dataset == 'mnist' or args.dataset == 'fashionmnist':
                image, label = samples
                if args.model == 'clf':
                    image, label = image.view(
                        image.shape[0], -1).to(device), label.to(device)
                    pred = model(image)
                elif args.model == 'hrclf':
                    image, label = image.to(device), label.to(device)
                    pred = model(image)
                elif args.model == 'hrclf3':
                    image, label = image.to(device), label.to(device)
                    pred = model(image)
                elif args.model == 'hrgen':
                    image, label = image.to(device), label.to(device)
                    mask_img, mask = generate_batches(image, args.mask_ratio)
                    pred = model(mask_img, mask, label)

                    complement_mask = 1 - mask
                    complement_mask_img = image * complement_mask
                    complement_pred = model(
                        complement_mask_img, complement_mask, label)
            elif args.dataset == 'comnist':
                ori_image, mask_image, mask, label = samples
                ori_image, mask_image, mask = ori_image.view(ori_image.shape[0], -1), mask_image.view(
                    mask_image.shape[0], -1), mask.view(mask.shape[0], -1)
                if args.model == 'coclf':
                    pred_image, pred_label = model(mask_image, mask)
                elif args.model == 'cogen':
                    pred = model(mask_image, mask, label)
                    label = ori_image
            else:
                raise 'Unknown dataset.'
            if args.loss == 'bce':
                if args.model == 'hrgen':
                    loss = F.binary_cross_entropy(pred, image)
                    complement_loss = F.binary_cross_entropy(
                        complement_pred, image)
                    loss = loss + complement_loss
                else:
                    label = torch.nn.functional.one_hot(
                        label, 10).to(torch.float32)
                    loss = F.binary_cross_entropy(pred, label)
            elif args.loss == 'dice':
                dsc_loss = dice_loss(pred_image, ori_image)
            elif args.loss == 'ce':
                label = torch.nn.functional.one_hot(
                    label, 10).to(torch.float32)
                loss = F.cross_entropy(pred, label)
            elif args.loss == 'dice+ce':
                ce_loss = F.cross_entropy(pred_label, label)
                dsc_loss = dice_loss(pred_image, ori_image)
                loss = ce_loss + dsc_loss
            else:
                raise 'Unknown loss function.'
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach() * label.shape[0]
            num_train += label.shape[0]
            pbar.set_description(
                f"Epoch {epoch}/{args.epochs}, loss={loss.item():.4f}")

        train_loss /= num_train
        print("avg_loss=", train_loss)

        # evaluation
        model.eval()
        acc = 0
        aux_acc = 0  # for comnist
        num_test = 0
        print('Start evaluation...')
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for samples in pbar:
                if args.dataset in ['multiplebit', 'singlebit']:
                    image, label, mask = samples
                    image, label, mask = image.view(image.shape[0], -1), label.view(label.shape[0], -1), mask.view(
                        mask.shape[0], -1)
                    pred = model(image, mask)
                    acc += (1 - dice_loss(pred, label)) * pred.shape[0]
                elif args.dataset == 'mnist' or args.dataset == 'fashionmnist':
                    image, label = samples
                    if args.model == 'clf':
                        image, label = image.view(
                            image.shape[0], -1).to(device), label.to(device)
                        pred = model(image)
                        pred = torch.argmax(pred, dim=-1)
                        acc += torch.where(pred == label, 1., 0.).sum()
                    elif args.model == 'hrclf' or args.model == 'hrclf3':
                        image, label = image.to(device), label.to(device)
                        pred = model(image)
                        pred = torch.argmax(pred, dim=-1)
                        acc += torch.where(pred == label, 1., 0.).sum()
                    elif args.model == 'hrgen':
                        image, label = image.to(device), label.to(device)
                        mask_img, mask = generate_batches(
                            image, args.mask_ratio)
                        pred = model(mask_img, mask, label)
                        acc += (1 - dice_loss(pred, image).detach()) * \
                            pred.shape[0]

                        complement_mask = 1 - mask
                        complement_mask_img = image * complement_mask
                        complement_pred = model(
                            complement_mask_img, complement_mask, label)
                        aux_acc += (1 - dice_loss(complement_pred,
                                    image).detach()) * pred.shape[0]
                elif args.dataset == 'comnist':
                    ori_image, mask_image, mask, label = samples
                    ori_image, mask_image, mask = ori_image.view(ori_image.shape[0], -1), mask_image.view(
                        mask_image.shape[0], -1), mask.view(mask.shape[0], -1)
                    if args.model == 'cogen':
                        pred = model(mask_image, mask, label)
                        label = ori_image
                        acc += (1 - dice_loss(pred, label)) * pred.shape[0]
                    elif args.model == 'coclf':
                        pred_image, pred_label = model(mask_image, mask)
                        pred_label = torch.argmax(pred_label, dim=-1)
                        aux_acc += (1 - dice_loss(pred_image, ori_image)
                                    ) * label.shape[0]
                        acc += torch.where(pred_label == label,
                                           1., 0.).sum() / label.shape[0]
                else:
                    raise 'Unknown dataset.'

                num_test += label.shape[0]
        acc /= num_test
        aux_acc /= num_test
        if aux_acc != 0:
            print(f'accuracy: {acc}, auxiliary accuracy: {aux_acc}')
        else:
            print(f'accuracy: {acc}')
        if acc > best_acc:
            os.makedirs(os.path.join(exp_name, 'model'), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
                os.path.join(exp_name, 'model', 'best_model.pt'))
            best_acc = acc
        with open(os.path.join(exp_name, 'log.txt'), 'a') as f:
            if aux_acc != 0:
                f.write(
                    f'Epoch: {epoch}/{args.epochs}, loss={train_loss.item():.4f}, acc={acc.item():.4f}, aux_acc={aux_acc.item():.4f}\n')
            else:
                f.write(
                    f'Epoch: {epoch}/{args.epochs}, loss={train_loss.item():.4f}, acc={acc.item():.4f}\n')
            scheduler.step()
        epoch += 1

    end = time()
    with open(os.path.join(exp_name, 'log.txt'), 'a') as f:
        f.write(
            f'Training completed in {end - begin}s. Best accuracy: {best_acc}\n')


if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    set_seed(args.seed)
    train(args, device)
