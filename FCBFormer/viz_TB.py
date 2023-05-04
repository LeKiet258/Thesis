from Models import models
from Data.dataset import SegDataset
import argparse
import torch
import torch.nn as nn
# from torchsummary import summary
from PIL import Image
from torchvision import transforms
from torch.utils import data
import multiprocessing
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

activation = {}
input_ca = None
def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def build(args):
    # ==============load ảnh test==============
    # chuẩn bị đường dẫn
    img_path =  args.test_set + "/images/*"
    input_paths = sorted(glob.glob(img_path))
    depth_path = args.test_set + "/masks/*"
    target_paths = sorted(glob.glob(depth_path))

    # thực hiện transformation
    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )
    test_indices = [0] # temp: viz_dataset
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

    # ==============chuẩn bị model + load weight==============
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.FCBFormer()

    state_dict = torch.load(args.weight, map_location=torch.device('cpu')) #f"./trained_weights/best.pt"
    model.load_state_dict(state_dict["model_state_dict"]) # key là tên của layer và value là parameter (gồm weight và bias) của layer đó
    model.to(device)
    tb = model.TB
    # print(tb)
    # shit
    
    # register forward hooks on the layers of choice
    hooks = []
    ## feature maps by PVT
    for i, ix in enumerate([2,5,8,-1], 1):
        hooks.append(tb.backbone[ix].register_forward_hook(getActivation(f'F{i}')))
    ## LE block
    for i in range(len(tb.LE)):
        hooks.append(tb.LE[i].register_forward_hook(getActivation(f'F{i+1}_LE')))
    ## CIM
    hooks.append(tb.ca.register_forward_hook(getActivation('ca'))) 
    hooks.append(tb.sa.register_forward_hook(getActivation('sa'))) 
    ## SFA
    for i in range(len(tb.SFA)):
        hooks.append(tb.SFA[i].register_forward_hook(getActivation(f'sfa_{i}'))) # sfa_2,1,0 = 3 lần aggregate from top to bot

    # ==============chạy forward pass==============
    for i, (img, target) in enumerate(test_dataloader):
        img, target = img.to(device), target.to(device)
        output = tb(img)
    
    for k in activation:
        print(f"{k}: {activation[k].shape}")
    # shit

    # detach the hooks
    for hook in hooks:
        hook.remove()
    
def viz_fm(args):
    w = os.path.basename(args.weight).split(".pt")[0]

    # create folder containing fmap
    if not os.path.exists("feature_maps"):
        os.mkdir("feature_maps")
    par = f"feature_maps/{w}_feature_maps"

    if not os.path.exists(f"./{par}"):
        os.makedirs(f"./{par}")
    else: # nếu tồn tại đường dẫn trên thì xoá đi tạo mới
        dir = f"./{par}"
        import shutil
        shutil.rmtree(dir) # remove
        os.makedirs(dir) # create new
    
    # viz ca 
    weight_ca = activation['ca'] # (1, 64, 1, 1)
    # print(weight_ca)
    mat = torch.squeeze(weight_ca).reshape(8,8)
    fig, ax = plt.subplots()
    ax.imshow(mat.cpu(), cmap="gray")
    ax.axis("off")
    ax.set_title("2d representation of 1d channel attention")
    plt.savefig(f"./{par}/ca.png")

    # viz sa
    weight_sa = activation['sa']
    fig, ax = plt.subplots()
    weight_sa = weight_sa[0].permute(1,2,0) # 88,88,1
    ax.imshow(weight_sa[:,:,0].cpu(), cmap='gray')
    ax.axis("off")
    ax.set_title("Spatial Attention Map")
    plt.savefig(f"./{par}/sa.png")

    deps = [64,128,320,512]
    spats = [88,44,22,11]
    weight_f1_le = None
    for i, (fm, weight) in enumerate(activation.items()):
        if fm == "sa" or fm == "ca": # CIM
            continue

        if fm == "F1_LE":
            weight_f1_le = weight
            # np.savetxt('F1_LE.csv', weight.cpu().reshape(-1))

        weight = torch.squeeze(weight) # squeeze: remove axis "1"
        if 'LE' in fm or 'sfa' in fm:
            weight = weight.permute(1,2,0)
        else:
            weight = weight.reshape(spats[i], spats[i], deps[i]) 
        weight = weight[:, :, :64] # tạm thời chỉ lấy 64 fm ở mỗi stage để viz
        fig, axes = plt.subplots(8,8, figsize=(30,30))
        axes = axes.ravel()

        for j in range(64):
            axes[j].imshow(weight[:, :, j].cpu(), cmap='gray')
            axes[j].axis("off")
        plt.savefig(f"./{par}/{fm}.png")

    # save elem-wise result of cim
    weighted_fmap = weight_ca * weight_f1_le # (1, 64, 88, 88)
    weight = torch.squeeze(weighted_fmap) # squeeze: remove axis "1"
    weight = weight.permute(1,2,0)
    fig, axes = plt.subplots(8,8, figsize=(30,30))
    axes = axes.ravel()
    for j in range(64):
        axes[j].imshow(weight[:, :, j].cpu(), cmap='gray')
        axes[j].axis("off")
    plt.savefig(f"./{par}/weighted_fmap.png")

    # save the final result of cim (sa's output * sa's input)
    cim_res = weighted_fmap * activation['sa'] # (1, 64, 88, 88)
    weight = torch.squeeze(cim_res) # squeeze: remove axis "1"
    weight = weight.permute(1,2,0)
    _, axes = plt.subplots(8,8, figsize=(30,30))
    axes = axes.ravel()
    for j in range(64):
        axes[j].imshow(weight[:, :, j].cpu(), cmap='gray')
        axes[j].axis("off")
    plt.savefig(f"./{par}/cim_output.png")


def get_args():
    parser = argparse.ArgumentParser(description="Print feature map of 1 image")
    parser.add_argument("--weight", type=str, required=True, help="đường dẫn tới best weight")
    parser.add_argument("--test-set", type=str, required=True, help="đường dẫn tới file hình muốn vẽ feature map")
    
    return parser.parse_args()

def main():
    args = get_args()
    build(args)
    viz_fm(args)

if __name__ == "__main__":
    main()