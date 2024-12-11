import os
import sys
import time

# import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import skimage.transform as skTrans

# import skimage.transform as skTrans


def normalize_img(img_array):
    maxes = np.quantile(img_array, 0.995, axis=(0, 1, 2))
    # print("Max value for each modality", maxes)
    return img_array / maxes


def process_dicom_dir(directory, meta, meta_all, img_id):
    """Process a directory of DICOM files."""
    dicom_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")
    ]
    dicom_files.sort(key=lambda x: float(pydicom.dcmread(x).SliceLocation))
    slices = [pydicom.dcmread(f).pixel_array for f in dicom_files]
    im = np.stack(slices, axis=-1)  # Create 3D volume from slices
    n_i, n_j, n_k = im.shape
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2
    im1 = skTrans.resize(im[center_i, :, :], (72, 72), order=1, preserve_range=True)
    im2 = skTrans.resize(im[:, center_j, :], (72, 72), order=1, preserve_range=True)
    im3 = skTrans.resize(im[:, :, center_k], (72, 72), order=1, preserve_range=True)
    im = np.array([im1, im2, im3]).T
    idx = meta[meta["Image Data ID"] == img_id].index[0]
    label = meta.at[idx, "Group"]
    subject = meta.at[idx, "Subject"]
    norm_im = normalize_img(im)
    new_row = pd.DataFrame([{"img_array": norm_im, "label": label, "subject": subject}])
    meta_all = pd.concat([meta_all, new_row], ignore_index=True)
    return meta_all


def create_dataset(meta, meta_all, path_to_datadir):
    # Traverse the directory structure
    for root, dirs, files in os.walk(path_to_datadir):
        for file in files:
            if file.endswith(".dcm"):
                # Handle DICOM files (assume one directory contains a set of slices)
                directory = os.path.dirname(os.path.join(root, file))
                img_id = os.path.basename(directory).split("_")[
                    -1
                ]  # Adjust naming convention if needed
                print(f"Processing {directory}, Image ID: {img_id}")
                meta_all = process_dicom_dir(directory, meta, meta_all, img_id)
                break  # Avoid processing the same directory multiple times

    meta_all.to_pickle("./data/processed/mri/mri_meta.pkl")


def main():

    path_to_meta = "./data/imaging/mri/mri_adnibaseline_12_09_2024.csv"
    path_to_datadir = "./data/imaging/mri/ADNI"

    meta = pd.read_csv(path_to_meta)
    print(f"Opened meta at {path_to_meta}, Length: {len(meta)}")

    # get rid of not needed columns
    meta = meta[["Image Data ID", "Group", "Subject"]]  # MCI = 0, CN =1, AD = 2
    meta["Group"] = pd.factorize(meta["Group"])[0]
    # initialize new dataset where arrays will go
    meta_all = pd.DataFrame(columns=["img_array", "label", "subject"])
    create_dataset(meta, meta_all, path_to_datadir)


if __name__ == "__main__":
    main()
