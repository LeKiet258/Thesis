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
    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = Dice_loss(output, target) + BCE_loss(torch.sigmoid(output), target)
        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())

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
def test(model, device, test_loader, epoch, perf_measure):
    t = time.time()
    model.eval()
    perf_accumulator = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        
        print(
            "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                epoch,
                batch_idx + 1,
                len(test_loader),
                100.0 * (batch_idx + 1) / len(test_loader),
                np.mean(perf_accumulator),
                time.time() - t,
            ),
            end = "" if batch_idx + 1 < len(test_loader) else "\n",
        )

    return np.mean(perf_accumulator), np.std(perf_accumulator)


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
        train_dataloader,
        val_dataloader,
        Dice_loss,
        BCE_loss,
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
    prev_best_test = None
    loss_epoch = None 
    
    if checkpoint is not None:
        print(f"...Loading STT epoch, prev_best_test from {args.resume}")
        start_epoch = checkpoint['epoch'] + 1 
        prev_best_test = checkpoint['test_measure_mean']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        loss_epoch = checkpoint['loss_epoch']
    else:
        loss_epoch = []

    for epoch in range(start_epoch, args.epochs + 1):
        try:
            loss = train_epoch(model, device, train_dataloader, optimizer, epoch, Dice_loss, BCE_loss)
            loss_epoch.append(loss)
            test_measure_mean, test_measure_std = test(model, device, val_dataloader, epoch, perf)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            sys.exit(0)
        if args.lrs == "true": # nếu có dùng learning rate scheduler -> update lr according to the scheduling scheme
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test: # save current best
            print(f"[INFO] Saving best weights to trained_weights/{args.name}{file_cnt}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                    "scheduler_state_dict": scheduler.state_dict()
                },
                f"trained_weights/{args.name}{file_cnt}.pt",
            )
            prev_best_test = test_measure_mean
        
        # remove prev epoch 
        # if os.path.exists(f"trained_weights/{args.name}-epoch_{epoch-1}.pt"):
        #     os.remove(f"trained_weights/{args.name}-epoch_{epoch-1}.pt")
        # save last.pt
        old_name = f"trained_weights/{args.name}-epoch_{epoch-1}.pt"
        print(f"[INFO] Saving epoch {epoch} to trained_weights/{args.name}-epoch_{epoch}.pt")
        torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": prev_best_test, # current best, not this epoch's dice
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss_epoch": loss_epoch
                },
                old_name, # ghi đè
            )
        os.rename(old_name, f"trained_weights/{args.name}-epoch_{epoch}.pt")
    
    # lưu loss_epoch
    with open(f'loss_tracking_{args.name}.txt', 'w') as f:
        for item in loss_epoch:
            f.write(str(item) + '\n')


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

    return parser.parse_args()


def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()