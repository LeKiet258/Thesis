import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
from skimage.io import imread
from skimage.transform import resize
from Data.dataloaders import split_ids

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import shutil

def eval(args):
    # determine path to prediction folder
    full_or_not = " full " if args.generalisability else " "
    if f"Tested on{full_or_not}{args.test_dataset}" not in args.pred_dir: # dẫn sai đường dẫn tới folder khác test_dataset
        raise Exception("'test-dataset' & 'pred-dir' are incompatible. Consider modifying 'test-dataset' or 'pred-dir'")
    prediction_files = sorted(glob.glob(args.pred_dir)) 
    if not prediction_files:
        raise Exception("prediction folder is empty")

    # determine path to GT
    data_root = '/content/drive/MyDrive/Thesis/data/'
    depth_path = data_root + args.test_dataset + "/masks/*"
    target_paths = sorted(glob.glob(depth_path)) 

    _, test_indices, _ = ('_', np.arange(len(target_paths)), '_') if args.generalisability else split_ids(len(target_paths))
    test_files = sorted([target_paths[test_indices[i]] for i in range(len(test_indices))]) # [img1.png, img2.png]

    dice = []
    IoU = []
    precision = []
    recall = []

    for i in range(len(test_files)):
        # print(prediction_files[i], '||', test_files[i])
        pred = np.mean(cv2.imread(prediction_files[i]) / 255, axis=2) > 0.5 # shape: (352,352)
        pred = np.ndarray.flatten(pred) # shape: 123904 
        gt = (resize(cv2.imread(test_files[i]), (int(352), int(352)), anti_aliasing=False) > 0.5)

        if len(gt.shape) == 3:
            gt = np.mean(gt, axis=2)
        gt = np.ndarray.flatten(gt)

        dice.append(f1_score(gt, pred))
        IoU.append(jaccard_score(gt, pred)) 
        precision.append(precision_score(gt, pred))
        recall.append(recall_score(gt, pred))

        print(
            "\rTest: [{}/{} ({:.1f}%)]\tModel scores: Dice={:.6f}, mIoU={:.6f}, precision={:.6f}, recall={:.6f}".format(
                i + 1,
                len(test_files),
                100.0 * (i + 1) / len(test_files),
                np.mean(dice),
                np.mean(IoU),
                np.mean(precision),
                np.mean(recall),
            ),
            end="" if i + 1 < len(test_files) else "\n",
        )
    
    # print worst predictions
    top_lows = {i:item for i, item in enumerate(dice)}
    top_lows = dict(sorted(top_lows.items(), key=lambda item: item[1])) # sort dict based on vals
    top_lows_ix = list(top_lows.keys())[:args.top_low]

    if not os.path.exists("./Worst cases"): 
        os.makedirs("./Worst cases")  
    if not os.path.exists("./Worst cases/Trained on {}".format(args.train_dataset)):
        os.makedirs(f"./Worst cases/Trained on {args.train_dataset}")
    if os.path.exists(f"./Worst cases/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}"): # remove all files of the folde
        old_files = glob.glob(f"./Worst cases/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}/*") 
        for f in old_files:
            os.remove(f)
    else: 
        os.makedirs(f"./Worst cases/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}")     

    extension = os.listdir(data_root + args.test_dataset + f"/images")[0].split('.')[-1]
    for i, ix in enumerate(top_lows_ix):
        img_targ = Image.open(test_files[ix]).resize((352, 352)).convert('1')
        img_pred = Image.open(prediction_files[ix]).resize((352, 352)).convert('1')
        name_saved = test_files[ix].split('/')[-1].split('.')[0]
        img_rgb = Image.open(data_root + args.test_dataset + f"/images/{name_saved}.{extension}").resize((352, 352))
        fig, ax = plt.subplots(1,3)
        fig.suptitle(f'{i+1}-th lowest dice', fontsize=16)
        ax[0].imshow(img_rgb)
        ax[0].axis('off')
        ax[0].set_title('rgb')
        ax[1].imshow(img_targ)
        ax[1].axis('off')
        ax[1].set_title('target')
        ax[2].imshow(img_pred)
        ax[2].axis('off')
        ax[2].set_title('pred')
        plt.savefig(f"./Worst cases/Trained on {args.train_dataset}/Tested on{full_or_not}{args.test_dataset}/{name_saved}.png")


def get_args():
    parser = argparse.ArgumentParser(description="Make predictions on specified dataset")
    parser.add_argument("--train-dataset", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, required=True, help="follow the format: x/y/*")
    parser.add_argument("--generalisability", action='store_true', help="used when conduct generalisability test")
    parser.add_argument("--top-low", type=int, default=10) # mặc định override nếu execute nhiều lần

    return parser.parse_args()


def main():
    args = get_args()
    eval(args)


if __name__ == "__main__":
    main()

