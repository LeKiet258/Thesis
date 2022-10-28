import numpy as np
import random
import multiprocessing

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

from Data.dataset import SegDataset


def split_ids(len_ids, is_train):
    '''Arguments:
    - len_ids: SL file ảnh input truyền vào. VD: len(Dataset/TrainDataset/*) or len(Dataset/TestDataset/*)
    - is_train: True nếu trong pha training (split dataset into train/val (90:10) numpy array of indices), False nếu ko phải'''
    train_indices, test_indices, val_indices = None, None, None # init

    if is_train:
        valid_size = int(round((10 / 100) * len_ids))
         
        train_indices, val_indices = train_test_split(
            np.linspace(0, len_ids - 1, len_ids).astype(int),
            test_size=valid_size,
            random_state=42,
        )

        # for print
        n_train, n_val = len(train_indices), len(val_indices)
        print("[INFO] Train:val = {}:{} ảnh = {:.2f}%:{:.2f}%".format(
            n_train, n_val, 
            n_train/(n_train+n_val)*100, n_val/(n_train+n_val)*100
        ))
    else:
        test_indices = np.arange(len_ids)
        print(f"[INFO] Test trên {len(test_indices)} ảnh")

    return train_indices, test_indices, val_indices # np arrays


def get_dataloaders(input_paths, target_paths, batch_size, is_train):
    '''return train/test/val dataloaders. Arguments:
        - is_train: True nếu là pha train (tức nạp vào TrainDataset và chia thành train/val_set trong quá trình train(); False nếu là pha test(tức nạp vào TestDataset thì chia mỗi test_set trong quá trình test)
        - is_generalisability = False: có thực hiện bài generalisability test ko
        - input_paths: a list of rgb images from Dataset/images
        - target_paths: a list of masks from Dataset/masks'''
    train_dataloader, test_dataloader, val_dataloader = None, None, None # init
    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )
    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_indices, test_indices, val_indices = split_ids(len(input_paths), is_train)

    if is_train:
        # transform trước khi đưa ảnh input vào model
        transform_input4train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((352, 352), antialias=True),
                transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
                ),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        
        # thiết lập các option đầu vào cho tập train/test/val (tạm thời ignore input_paths & target_paths)
        train_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            transform_input=transform_input4train,
            transform_target=transform_target,
            hflip=True,
            vflip=True,
            affine=True,
        )
        val_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
        # convert 2 tập train/val thành dạng dataloader (batch iterable)
        train_dataset = data.Subset(train_dataset, train_indices)
        val_dataset = data.Subset(val_dataset, val_indices)

        train_dataloader = data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last= False, # bỏ đi batch cuối vì dataset is not divisible by your batch_size; False nếu mún giữ lại
            num_workers=multiprocessing.Pool()._processes,
        )

        val_dataloader = data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=multiprocessing.Pool()._processes,
        )
    else:
        test_dataset = SegDataset(
            input_paths=input_paths,
            target_paths=target_paths,
            transform_input=transform_input4test,
            transform_target=transform_target,
        )
        test_dataset = data.Subset(test_dataset, test_indices)

        test_dataloader = data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=multiprocessing.Pool()._processes,
        )
    
    return train_dataloader, test_dataloader, val_dataloader
