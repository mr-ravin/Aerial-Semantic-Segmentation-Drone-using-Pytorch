import glob
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils.graphics import generate_group_mask

class CustomImageDataset(Dataset):
    def __init__(self, root_dir=".", total_channels=5, individual_class_count=24, transform=None, mode="train", device="cpu"):
        self.root_dir = root_dir
        self.mode = mode
        self.device = device
        self.path = root_dir+"/dataset/"+self.mode+"/images/"
        self.net_data = glob.glob(self.path+"*.jpg")
        self.transform = transform
        self.total_channels = total_channels
        self.individual_class_count = individual_class_count

    def __len__(self):
        return len(self.net_data)

    def __getitem__(self, idx):
        img_path = self.net_data[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_shape = image.shape
        tmp_name = img_path.split("/")[-1]
        mask_name = tmp_name.replace(".jpg",".png")
        mask_path = self.root_dir+"/dataset/masks/"+mask_name
        raw_mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = generate_group_mask(raw_mask_np=raw_mask_np, individual_class_count=self.individual_class_count)
        masks_list= []
        net_mask = None
        for idx in range(self.total_channels):
            tmp_mask = (mask == idx)
            tmp_mask = tmp_mask*1
            masks_list.append(tmp_mask.copy())
        if self.transform:
            transformed = self.transform(image=image,masks=masks_list)
            image, net_mask = transformed["image"], transformed["masks"]
            net_mask = torch.stack(net_mask,axis=0)
        return image, net_mask, img_path