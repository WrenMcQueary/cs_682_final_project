"""Create data/original.csv by combining extended_metadata_with_rle_gt_masks.csv and previous.csv, keeping the column
conventions of previous.csv
"""


import pandas as pd
import numpy as np
import os


DATA_PATH = "../data/"


# Read both .csv files as dataframes
df_previous = pd.read_csv("previous.csv")
df_extended = pd.read_csv("extended_metadata_with_rle_gt_masks.csv")


# Create and fill an out dataframe
df_out = df_previous.copy()
for ii, row in df_extended.iterrows():
    ID = row.filename
    ORGAN = row.tissue_name
    DATA_SOURCE = row.data_type
    IMG_HEIGHT = row.image_dims[1:-1].split(", ")[0]
    IMG_WIDTH = row.image_dims[1:-1].split(", ")[1]
    PIXEL_SIZE = row.pixel_size
    TISSUE_THICKNESS = row.tissue_thickness
    RLE = row.rle
    AGE = row.age
    SEX = row.sex
    new_row = [ID, ORGAN, DATA_SOURCE, IMG_HEIGHT, IMG_WIDTH, PIXEL_SIZE, TISSUE_THICKNESS, RLE, AGE, SEX]
    df_out.loc[len(df_out.index)] = new_row


# Write to .csv
df_out.to_csv(os.path.join(DATA_PATH, "original.csv"), sep=",", index=False)
