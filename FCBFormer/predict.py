import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics


def build(args):
    '''pre-preperation for prediction, return:
        - device: cpu or cuda
        - test_dataloader
        - perf: performance metric function
        - model: FCBFormer()
        - target_paths: where the segmentation images lie
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # configure path to test dataset
    data_root = '/content/drive/MyDrive/Thesis/data/'
    img_path = data_root + args.test_dataset + "/images/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = data_root + args.test_dataset + "/masks/*"
    target_paths = sorted(glob.glob(depth_path))
    # get test_dataloader
    _, test_dataloader, _ = dataloaders.get_dataloaders(input_paths, target_paths, batch_size=1, is_generalisability=args.generalisability)
    _, test_indices, _ = (_, np.arange(len(target_paths)), _) if args.generalisability else dataloaders.split_ids(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))]
    # print(target_paths)
    perf = performance_metrics.DiceScore()
    model = models.FCBFormer()

    state_dict = torch.load(f"./trained_weights/FCBFormer_{args.train_dataset}.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)

    return device, test_dataloader, perf, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)
    full_or_not = " full " if args.generalisability else " "
    cnt = 0

    if not os.path.exists("./Predictions"):
        os.makedirs("./Predictions")
    if not os.path.exists("./Predictions/Trained on {}".format(args.train_dataset)):
        os.makedirs(f"./Predictions/Trained on {args.train_dataset}")
    if not os.path.exists(f"./Predictions/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}"):
        os.makedirs(f"./Predictions/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}")
    else:
        if not args.exist_ok: # nếu ko ghi đè thì tạo folder mới
            test_on_folders = os.listdir(f"./Predictions/Trained on {args.train_dataset}")
            test_on_kvasir = [fold for fold in test_on_folders if 'Kvasir' in fold]
            test_on_cvc = [fold for fold in test_on_folders if 'CVC' in fold]
            cnt = " " + str(len(test_on_kvasir) if args.test_dataset == "Kvasir" else len(test_on_cvc)) # " 2", " 3"
            os.makedirs(f"./Predictions/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}{cnt}")
    
    file_cnt = cnt if cnt != 0 else ""
    t = time.time()
    model.eval()
    perf_accumulator = []
    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        perf_accumulator.append(perf_measure(output, target).item())
        predicted_map = np.array(output.cpu())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite(
            "./Predictions/Trained on {}/Tested on{}{}{}/{}".format(
                args.train_dataset, full_or_not, args.test_dataset, file_cnt, os.path.basename(target_paths[i])
            ),
            predicted_map * 255,
        )

        print("\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                i + 1,
                len(test_dataloader),
                100.0 * (i + 1) / len(test_dataloader),
                np.mean(perf_accumulator),
                time.time() - t,
            ),
            end="" if i + 1 < len(test_dataloader) else "\n",
        )


def get_args():
    parser = argparse.ArgumentParser(description="Make predictions on specified dataset")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, required=True)
    # parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--exist-ok", action='store_true', help='allow override prediction folder? default: create new folder')
    parser.add_argument("--generalisability", action='store_true', help="conduct generalisability test?")
    
    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()