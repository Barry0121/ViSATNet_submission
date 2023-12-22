import argparse
import os
from tqdm import tqdm
import time

import numpy as np
import torch
import torch.nn.functional as F

# from satnet module
from models import *
from loss_func.loss import dice_loss
from utils.setup_singlebitDataset import setup_one_digit
from utils.get_dataloader import get_tarin_loader, get_test_loader
from utils.random_mask import generate_batches

# adversarial example generations
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hrclf')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--root_dir', type=str, default='data',
                        help="Specify root directory for data files.")
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--load_weights', type=bool, default=True,
                        help="Flag to load existing model weight. "
                             "Assume weights are trained with clean example.")

    # Adversarial training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help="Training epochs (if no loading weights are provided).")
    parser.add_argument('--eps', type=float, default=0.3,
                        help="Total epsilon for FGM and PGD attacks.")
    parser.add_argument('--adversarial_training', type=bool, default=False,
                        help="Flag for adding adversarial training examples "
                             "with projected gradient method.")

    # for logging information
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--aux', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=1)
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--lr', type=float, default=2.e-3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--loss', type=str, default='ce')
    return parser.parse_args()

def _get_model_path(args):
    """Return the target model/loss path given arguments.
    If the path doesn't exist, make it exist."""
    exp_name = os.path.join(args.save_dir, args.dataset,
                        (f"model_{args.model}_m_{args.m}_aux_{args.aux}"
                         f"_dim_{args.hidden_dim}_stride_{args.stride}"
                         f"_loss_{args.loss}_lr_{args.lr}_seed_{args.seed}"))
    os.makedirs(exp_name, exist_ok=True)
    return exp_name

def load_data(args, device):
    """
    dataset options: mnist, comnist, and fashionmnist
    """
    train_loader = get_tarin_loader(args.dataset,
                                    os.path.join(args.root_dir, args.dataset),
                                    args.batch_size,
                                    device=device)
    test_loader = get_test_loader(args.dataset,
                                  os.path.join(args.root_dir, args.dataset),
                                  args.batch_size,
                                  device=device)
    return train_loader, test_loader

def load_model(args, device):
    """
    model options: clf, hrclf (2 satnet layers), 3hrclf (3 satnet layers)
    """
    if args.model == 'clf':
        model = Classifier(m=args.m,
                           aux=args.aux)
    elif args.model == 'hrclf':
        model = HierarchicalClassifier(m=args.m,
                                       aux=args.aux,
                                       stride=args.stride,
                                       hidden_dim=args.hidden_dim)

    else: # if we are experimenting with multiple SATNet layers
        model = HierarchicalClassifier(m=args.m,
                                       aux=args.aux,
                                       stride=args.stride,
                                       hidden_dim=args.hidden_dim)
        # model = ThreeLayersHierarchicalClassifier(m=args.m,
        #                                           aux=args.aux,
        #                                           stride=args.stride,
        #                                           hidden_dim=(args.hidden_dim, args.hidden_dim2)
        #                                           ).to(device)
    exp_name = _get_model_path(args)
    if args.load_weights:
        if args.adversarial_training:
            model_weight_path = os.path.join(exp_name, "model", "best_model_adv.pt")
        else:
            model_weight_path = os.path.join(exp_name, "model", "best_model.pt")
        checkpoint = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    print(f"{args.model} model loaded!\n")
    return model.to(device)

def main(args, device):
    # get export path
    exp_path = _get_model_path(args)
    log_file = open(os.path.join(exp_path, "adversarial_log.txt"), 'a')

    # Load training and test data
    train_loader, test_loader = load_data(args, device)

    # Get model
    model = load_model(args, device)

    # Training utilities
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train vanilla model
    if not args.load_weights:
        print("Training...\n")
        best_acc = 0.0
        model.train()
        for epoch in range(1, args.epochs+1):
            train_loss = 0.0
            train_acc = 0.0
            num_train = 0
            for x, y in tqdm(train_loader):
                x, y = x.to(device), y.to(device)
                if args.adversarial_training:
                    # Replace clean example with adversarial example for adversarial training
                    x = projected_gradient_descent(model, x, args.eps, 0.01, 40, np.inf)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(F.one_hot(pred, 10).to(torch.float32), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += torch.where(torch.argmax(pred, dim=-1) == y, 1., 0.).sum()
                num_train += y.shape[0]

                # print and save loss statistics
                loss_report = "epoch: {}/{}, train loss: {:.3f}, train acc: {:.3f}".format(
                    epoch, args.epochs, train_loss/num_train, train_acc/num_train)
                print(loss_report)
                log_file.write(loss_report+'\n')


            # Save model weights
            if train_acc > best_acc:
                os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
                if args.adversarial_training:
                    torch.save({
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(exp_path, 'model', 'best_model_adv.pt'))
                else:
                    torch.save({
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }, os.path.join(exp_path, 'model', 'best_model.pt'))
                best_acc = train_acc



    # Evaluate on clean and adversarial data
    print("Evaluation...\n")
    model.eval()
    report = dict(number_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in tqdm(test_loader):
        x, y = x.to(device), y.to(device)
        # Transform original test input into adversarial examples with two techniques
        x_fgm = fast_gradient_method(model, x, args.eps, np.inf)
        x_pgd = projected_gradient_descent(model, x, args.eps, 0.01, 40, np.inf)
        _, y_pred = model(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = model(x_fgm).max(
            1
        )  # model prediction on FGM adversarial examples
        _, y_pred_pgd = model(x_pgd).max(
            1
        )  # model prediction on PGD adversarial examples
        report["number_test"] += y.size(0)
        report["correct"] += y_pred.eq(y).sum().item()
        report["correct_fgm"] += y_pred_fgm.eq(y).sum().item()
        report["correct_pgd"] += y_pred_pgd.eq(y).sum().item()

    # Report Results
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report["correct"] / report["number_test"] * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report["correct_fgm"] / report["number_test"] * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report["correct_pgd"] / report["number_test"] * 100.0
        )
    )

    # Save evaluation results
    print(f"Save evaluation results at {exp_path} as adversarial_log.txt.\n")
    log_file.write("clean examples acc: {}, "
                   "FGM examples acc: {}, "
                   "PGD examples acc: {}. \n".format(
        report["correct"] / report["number_test"] * 100.0,
        report["correct_fgm"] / report["number_test"] * 100.0,
        report["correct_pgd"] / report["number_test"] * 100.0
    ))
    log_file.close()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    main(args, device)
