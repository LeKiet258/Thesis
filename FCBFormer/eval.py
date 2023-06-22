import os
import glob
import argparse
import numpy as np

from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score
# from skimage.io import imread
from skimage.transform import resize
from Data.dataloaders import split_ids

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import shutil

def eval(args):
    # checking
    which_dataset = os.path.basename(args.test_set)
    if which_dataset not in args.pred_dir.split('/') [-2]:
        raise Exception("--test-set & --pred-dir do not refer to the same test dataset")
    
    # determine path to prediction folder
    prediction_files = sorted(glob.glob(args.pred_dir))
    if not prediction_files:
        raise Exception("Prediction folder is empty")

    # determine path to GT
    depth_path = args.test_set + "/masks/*"
    target_paths = sorted(glob.glob(depth_path)) 

    test_indices = np.arange(len(target_paths))
    test_files = sorted([target_paths[test_indices[i]] for i in range(len(test_indices))]) # [img1.png, img2.png]
    print(f"[INFO] Test trên {len(test_indices)} ảnh")

    # checking
    test_files_reduce = [os.path.basename(f) for f in test_files]
    prediction_files_reduce = [os.path.basename(f) for f in prediction_files]
    if test_files_reduce != prediction_files_reduce:
        print(set(test_files_reduce).difference(set(prediction_files_reduce)))
        raise Exception("All files in masks & pred_dir/* are not the same")

    dice = []
    IoU = []
    precision = []
    recall = []

    for i in range(len(test_files)):
        if 'dice.csv' in prediction_files[i]:
            continue
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
    
    # lưu tất cả dice theo thứ tự tăng dần vào csv
    top_lows = dict(zip(prediction_files, dice))
    top_lows = dict(sorted(top_lows.items(), key=lambda item: item[1]))
    with open(f'{args.pred_dir[:-2]}/dice.csv', 'w') as f:
        for file_name, d in top_lows.items(): # d: dice
            file_name = os.path.basename(file_name) 
            f.write(f"{file_name},{d}\n")
    print(f"[INFO] Saving dice.csv to {args.pred_dir[:-2]}")

    # print worst predictions
    top_lows = {i:item for i, item in enumerate(dice)}
    top_lows = dict(sorted(top_lows.items(), key=lambda item: item[1])) # sort dict based on vals
    top_lows_ix = list(top_lows.keys())[:args.top_low]

    wc_dir = "./Worst cases/" + '/'.join(args.pred_dir.split('/')[-3:-1])
    print(f"[INFO] Saving {args.top_low} worst predictions to {wc_dir}")

    if not os.path.exists(wc_dir): # nếu chưa có thì tạo mới
        os.makedirs(wc_dir)
    else: # nếu đã tồn tại thì ghi đè
        shutil.rmtree(wc_dir) # remove
        os.makedirs(wc_dir) # create new

    extension = os.listdir(f"{args.test_set}/images")[0].split('.')[-1]
    for i, ix in enumerate(top_lows_ix):
        img_targ = Image.open(test_files[ix]).resize((352, 352)).convert('1')
        img_pred = Image.open(prediction_files[ix]).resize((352, 352)).convert('1')
        name_saved = os.path.basename(test_files[ix]).split('.')[0]
        img_rgb = Image.open(f"{args.test_set}/images/{name_saved}.{extension}").resize((352, 352))
       
        fig, ax = plt.subplots(1,3)
        fig.suptitle(f'{i+1}-th lowest dice (dice = {top_lows[ix]})', fontsize=16)
        for j, (title, img) in enumerate(zip(['rgb', 'target', 'pred'], [img_rgb, img_targ, img_pred])):
            ax[j].imshow(img)
            ax[j].axis('off')
            ax[j].set_title(title)
        plt.savefig(f"{wc_dir}/{name_saved}.png") # save fig


def get_args():
    parser = argparse.ArgumentParser(description="Make predictions on specified dataset")
    parser.add_argument("--test-set", type=str, required=True, help="đường dẫn tới thư mục test_set. VD: data/Kvasir+CVC/TestDataset")
    parser.add_argument("--pred-dir", type=str, required=True, help="đường dẫn tới thư mục Predictions. VD: Predictions/Train on CIM/Test on ETIS")
    parser.add_argument("--top-low", type=int, default=10) # mặc định override nếu execute nhiều lần

    return parser.parse_args()


def main():
    args = get_args()
    eval(args)


if __name__ == "__main__":
    main()

