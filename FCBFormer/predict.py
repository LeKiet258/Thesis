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
import shutil

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
    # data_root = '/content/drive/MyDrive/Thesis/data/'
    img_path =  args.test_set + "/images/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = args.test_set + "/masks/*"
    target_paths = sorted(glob.glob(depth_path))
    # get test_dataloader
    _, test_dataloader, _ = dataloaders.get_dataloaders(input_paths, target_paths, batch_size=1, is_train=False)
    test_indices = np.arange(len(target_paths))
    target_paths = [target_paths[test_indices[i]] for i in range(len(test_indices))] # ['./data/ASU_polyp/masks/0.png', './data/ASU_polyp/masks/1.png', ...]

    perf = performance_metrics.DiceScore()
    model = models.FCBFormer()

    state_dict = torch.load(args.weight, map_location=torch.device('cpu')) #f"./trained_weights/best.pt"
    model.load_state_dict(state_dict["model_state_dict"])
    model.to(device)

    return device, test_dataloader, perf, model, target_paths

def check_generalisability(test_set):
    '''test_set: đường dẫn tới thư mục test_set. VD: data/Kvasir+CVC/TestDataset'''
    test_name = None
    if "TestDataset" in test_set:
        folders = test_set.split('/')
        dataset_name, test_set_name = folders[-2], folders[-1]
        test_name = f"{dataset_name}'s {test_set_name}"
    else:
        test_name = f"full {test_set.split('/')[-1]}"
    return test_name

@torch.no_grad()
def predict(args):
    device, test_dataloader, perf_measure, model, target_paths = build(args)
    test_set_name = check_generalisability(args.test_set) # full or not

    weight_name = os.path.basename(args.weight).split('.')[0]
    if not os.path.exists(f"./Predictions/Train on {weight_name}/Test on {test_set_name}"):
        os.makedirs(f"./Predictions/Train on {weight_name}/Test on {test_set_name}")
    else: # nếu tồn tại đường dẫn trên thì ghi đè
        dir = f"./Predictions/Train on {weight_name}/Test on {test_set_name}"
        shutil.rmtree(dir) # remove
        os.makedirs(dir) # create new
    print(f"[INFO] Save to ./Predictions/Train on {weight_name}/Test on {test_set_name}")

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
            "./Predictions/Train on {}/Test on {}/{}".format(
                weight_name, test_set_name, os.path.basename(target_paths[i])
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
    parser.add_argument("--weight", type=str, required=True, help="đường dẫn tới best weight")
    parser.add_argument("--test-set", type=str, required=True, help="đường dẫn tới thư mục test_set. VD: data/Kvasir+CVC/TestDataset")
    # parser.add_argument("--data-root", type=str, required=True, dest="root")
    # parser.add_argument("--exist-ok", action='store_true', help='allow override prediction folder? default: create new folder')
    # parser.add_argument("--generalisability", action='store_true', help="conduct generalisability test?")
    
    return parser.parse_args()


def main():
    args = get_args()
    predict(args)


if __name__ == "__main__":
    main()