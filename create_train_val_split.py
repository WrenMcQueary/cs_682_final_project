"""Requires first running combine_datasets.py to merge previous.csv and extended_metadata_with_rle_gt_masks.csv.
Split off a percentage of the files in train_images, and the corresponding files in train_annotations, into separate
folders val_images and val_annotations.  Also split off a corresponding portion of train.csv into a new file, val.csv.
"""


import os
import shutil
import random
import glob as gb
from math import floor
import pandas as pd


random.seed(0)


DATA_PATH = "data/"
val_images_folder_name = "val_images/"
val_annotations_folder_name = "val_annotations/"
val_csv_file_name = "val.csv"


# Handle errors
# There's already a validation folder
for folder in val_images_folder_name, val_annotations_folder_name:
    if os.path.exists(os.path.join(DATA_PATH, folder)):
        raise RuntimeError(f"folder {os.path.join(DATA_PATH, folder)} already exists.  This implies that your training images and annoations have already been shaven down from their original size of 351 samples")
# There's already a train .csv file
if os.path.exists(os.path.join(DATA_PATH, "train.csv")):
    raise RuntimeError(f"file {os.path.join(DATA_PATH, 'train.csv')} already exists.  This implies that your training images and annotations have already been shaven down from their original size of 351 samples")
# There's already a validation .csv file
if os.path.exists(os.path.join(DATA_PATH, val_csv_file_name)):
    raise RuntimeError(f"file {os.path.join(DATA_PATH, val_csv_file_name)} already exists.  This implies that your training images and annotations have already been shaven down from their original size of 351 samples")


# Turn all .tif files in train_images/ into .tiff
for filename in os.listdir(os.path.join(DATA_PATH, "train_images")):
    if filename.endswith(".tif"):
        os.rename(os.path.join(os.path.join(DATA_PATH, "train_images"), filename), os.path.join(os.path.join(DATA_PATH, "train_images"), filename[:-3] + "tiff"))


# Split
original_csv_dataframe = pd.read_csv(os.path.join(DATA_PATH, "original.csv"))
train_csv_dataframe = pd.DataFrame(columns=["id", "organ", "data_source", "img_height", "img_width", "pixel_size", "tissue_thickness", "rle", "age", "sex"])
val_csv_dataframe = pd.DataFrame(columns=["id", "organ", "data_source", "img_height", "img_width", "pixel_size", "tissue_thickness", "rle", "age", "sex"])
train_image_paths = sorted(gb.glob(os.path.join(DATA_PATH, 'train_images/*tiff')))
#num_val_images = floor(len(train_image_paths) * ratio)
num_val_images = 207
val_image_paths = random.sample(train_image_paths, k=num_val_images)

# Write new .csv files
for train_image_path in train_image_paths:
    filename = train_image_path.replace("data/train_images\\", "").replace(".tiff", "")
    row = original_csv_dataframe.loc[original_csv_dataframe["id"] == filename]
    ID = row["id"].item()
    ORGAN = row["organ"].item()
    DATA_SOURCE = row["data_source"].item()
    IMG_HEIGHT = row["img_height"].item()
    IMG_WIDTH = row["img_width"].item()
    PIXEL_SIZE = row["pixel_size"].item()
    TISSUE_THICKNESS = row["tissue_thickness"].item()
    RLE = row["rle"].item()
    AGE = row["age"].item()
    SEX = row["sex"].item()
    new_row = [ID, ORGAN, DATA_SOURCE, IMG_HEIGHT, IMG_WIDTH, PIXEL_SIZE, TISSUE_THICKNESS, RLE, AGE, SEX]
    if train_image_path in val_image_paths:
        # Write to the val csv
        val_csv_dataframe.loc[len(val_csv_dataframe.index)] = new_row
    else:
        # Write to the train csv
        train_csv_dataframe.loc[len(train_csv_dataframe.index)] = new_row
train_csv_dataframe.to_csv(os.path.join(DATA_PATH, "train.csv"), sep=",", index=False)
val_csv_dataframe.to_csv(os.path.join(DATA_PATH, val_csv_file_name), sep=",", index=False)

# Move images and annotations
for folder in val_images_folder_name, val_annotations_folder_name:
    os.makedirs(os.path.join(DATA_PATH, folder))
for val_image_path in val_image_paths:
    # Move the image
    shutil.move(val_image_path, val_image_path.replace("/train_images", "/val_images"))
    # Move the annotation
    filename = val_image_path.replace("data/train_images\\", "").replace(".tiff", "")
    json_filename = f"{filename}.json"
    shutil.move(os.path.join(DATA_PATH, f"train_annotations/{json_filename}"), os.path.join(DATA_PATH, val_annotations_folder_name))

# Delete a .json file for an image that wasn't in one of the .csv files
os.remove(os.path.join(DATA_PATH, "train_annotations/38829_81799_A_2_4_lung.json"))
