import sched
import sys
import os
import argparse
import time
import numpy as np
import glob

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import re

def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss):
    '''Train for 1 epoch, return the batch mean loss'''
    t = time.time()
    model.train()
    loss_accumulator = []

    # instructions: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer; otherwise you’ll accumulate the gradients from multiple passes
        output = model(data) # forward pass: model.forward(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target) # calc batch loss
        loss.backward() # computes gradients (x.grad += dloss/dx) for every x that has requires_grad=True
        optimizer.step() # updates the value of x using the gradient x.grad (ex: x += -lr * x.grad)
        loss_accumulator.append(loss.item()) # item() method extracts the loss’s value as a Python floa

        print(
            "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                epoch,
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                loss.item(),
                time.time() - t,
            ),
            end="" if batch_idx + 1 < len(train_loader) else "\n",
        )

    return np.mean(loss_accumulator)


@torch.no_grad()
def test(model, device, test_loader, epoch, Dice_loss, BCE_loss, perf_measure):
    '''calc val_loss for 1 epoch'''
    t = time.time()
    model.eval()
    loss_accumulator = []
    perf_accumulator = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target) # calc batch loss
        loss_accumulator.append(loss.item())
        perf_accumulator.append(perf_measure(output, target).item())
        
        print(
            "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tPerformance: {:.6f}\tTime: {:.6f}".format(
                epoch,
                batch_idx + 1,
                len(test_loader),
                100.0 * (batch_idx + 1) / len(test_loader),
                np.mean(loss_accumulator), # **
                np.mean(perf_accumulator), # **
                time.time() - t,
            ),
            end = "" if batch_idx + 1 < len(test_loader) else "\n",
        )

    return np.mean(loss_accumulator), np.mean(perf_accumulator)


def build(args):
    '''Prepare data (train + val), model, optimizer, loss, metric in under the form of functions'''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    img_path = args.train_set + "/images/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = args.train_set + "/masks/*"
    target_paths = sorted(glob.glob(depth_path))

    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size, is_train=True
    )

    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    perf = performance_metrics.DiceScore()
    model = models.FCBFormer()
    checkpoint = None

    if args.resume:
        print(f"...Loading model, optimizer from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        if args.mgpu == "true":
            model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        print("[INFO] lr before checkpoint:", optimizer.param_groups[0]['lr'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        if args.mgpu == "true":
            model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (device, train_dataloader, val_dataloader,
            Dice_loss, BCE_loss,
            perf, model, optimizer, checkpoint)

def file_weight_cnt(weight_name):
    '''đếm file trùng tên, VD: [mix.pt, mix_1.pt]. Note: nếu xoá file best_weight thì fải xoá luôn last.pt thì hàm này chạy mới đúng'''
    file_cnt = 0
    # prepare path to save weight
    if not os.path.exists("./trained_weights"):
        os.makedirs("./trained_weights")
    else:
        existing_weights = os.listdir("./trained_weights")
        patt = f"{weight_name}(_\d)?.pt" 
        for weight_file in existing_weights: # VD: [mix.pt, mix_1.pt, CIM.pt]
            if re.match(patt, weight_file): 
                file_cnt += 1

    file_cnt = '' if file_cnt == 0 else f'_{file_cnt}'
    return file_cnt

def train(args):
    (
        device,
        train_dataloader, val_dataloader,
        Dice_loss, BCE_loss,
        perf, # DiceScore
        model,
        optimizer,
        checkpoint # if any, else: None
    ) = build(args)

    # keep track of file weight, avoid overriding
    file_cnt = file_weight_cnt(args.name) 

    # nếu có dùng learning rate scheduler
    if args.lrs == "true":
        if args.lrs_min > 0: # nếu có dùng min_lr trong scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    
    start_epoch = 1
    min_val_loss = 1e6 # the lowest err on val_set so far
    patience = 0
    max_patience = args.patience
    if checkpoint is not None:
        print(f"...Loading no. epoch, min_val_loss from {args.resume}")
        start_epoch = checkpoint['epoch'] + 1 
        min_val_loss = checkpoint['val_loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("[INFO] lr after checkpoint:", optimizer.param_groups[0]['lr'])
        patience = checkpoint['patience']
        print(f"[INFO] patience: {patience}/{max_patience}")
        if patience > max_patience:
            print(f"[INFO] Training ended due to early stopping with max_patience={max_patience}")
            return
    else:
        print(f"[INFO] max_patience = {max_patience}")

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            train_loss = train_epoch(model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss) # loss of each epoch
            val_loss, val_dice = test(model, device, val_dataloader, epoch, Dice_loss, BCE_loss, perf)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            sys.exit(0)

        # nếu có dùng learning rate scheduler -> update lr according to the scheduling scheme
        if args.lrs == "true": 
            scheduler.step(val_dice)

        # save best current epoch (if any)
        if val_loss < min_val_loss: 
            patience = 0
            min_val_loss = val_loss
            print(f"[INFO] Saving best weights to trained_weights/{args.name}{file_cnt}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss, # val loss of THIS epoch
                    "scheduler_state_dict": scheduler.state_dict(),
                    "patience": patience
                },
                f"trained_weights/{args.name}{file_cnt}.pt",
            )

        # save current epoch
        old_name = f"trained_weights/{args.name}-epoch_{epoch-1}.pt"
        print(f"[INFO] Saving epoch {epoch} to trained_weights/{args.name}-epoch_{epoch}.pt")
        torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss, 
                    "val_loss": min_val_loss, # the lowest val loss
                    "scheduler_state_dict": scheduler.state_dict(),
                    "patience": patience + 1 if val_loss > min_val_loss else patience
                },
                old_name, # ghi đè
        )
        os.rename(old_name, f"trained_weights/{args.name}-epoch_{epoch}.pt")
        
        if val_loss > min_val_loss: # err does not improve
            patience += 1
            if patience > max_patience:
                print(f"[INFO] Training ended due to early stopping with max_patience={max_patience}")
                return


def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--name", type=str, required=True, help="Đặt tên cho file best_weight.pt")
    parser.add_argument("--train-set", type=str, required=True, help="Đường dẫn tới thư mục tập train")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr", help="lr0: steps start out large, which makes quick progress and escape local minima") 
    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs", help="True nếu có dùng lr scheduler") 
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min")
    parser.add_argument("--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"])
    parser.add_argument('--resume', type=str, help='resume most recent training from the specified path')
    parser.add_argument('--patience', type=int, default=15, help='max patience for early stopping')

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
