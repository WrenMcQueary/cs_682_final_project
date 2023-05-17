from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pdb
from datetime import datetime
from plyer import notification
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import time
import copy
import random
import glob as gb
import cv2
import tifffile as tiff
from torch.utils.data import Dataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


#torch.use_deterministic_algorithms(True)    # Enforces reproducibility elsewhere as long as I seed

def seed_all(s):
    random.seed(s)
    torch.manual_seed(s)
    np.random.seed(s)
    
chosen_seed = 0
seed_all(chosen_seed)


def rle_to_mask(rle: str, shape: tuple) -> np.ndarray:
    """
    Borrows heavily from https://www.kaggle.com/code/hyunwoo2/training-with-thickness-and-staining-augmentation
    :param rle: run-length-encoded string starting on black
    :param shape: tuple with length 2, representing (height, width) of tensor to output
    :return: an ndarray of 0s and 1s, where 1s are detections.
    """
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, imsize=512):
        """
        :param transform: is applied to features
        :param target_transform: is applied to labels
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.iloc[idx, 0]}.tiff")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 7]
        image_original_height = np.shape(image)[0]
        image_original_width = np.shape(image)[1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = image.movedim(0, -1)    # Change dimensionality from (3, H, W) to (H, W, 3)
        label = rle_to_mask(label, (image_original_height, image_original_width))
        label = cv2.resize(label, (self.imsize, self.imsize), interpolation=cv2.INTER_AREA)
        return image, label