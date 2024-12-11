"""
This file extracts a 3D MRI volume into axial, coronal, and sagittal slices and visualizes them.
"""

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Step 1: Load DICOM files
dicom_dir = "./data/imaging/mri/ADNI/011_S_0002/MPRAGE/2005-08-26_08_45_00.0/I7025"
dicom_files = sorted(glob(f"{dicom_dir}/*.dcm"))  # Sort to ensure correct order

# Step 2: Assemble the 3D volume
slices = [pydicom.dcmread(f) for f in dicom_files]
slices.sort(key=lambda x: float(x.SliceLocation))  # Sort based on SliceLocation
volume = np.stack([s.pixel_array for s in slices], axis=-1)

# Step 3: Compute center indices
center_axial = volume.shape[2] // 2  # Axial slice (XY plane)
center_coronal = volume.shape[1] // 2  # Coronal slice (XZ plane)
center_sagittal = volume.shape[0] // 2  # Sagittal slice (YZ plane)

# Step 4: Extract slices
axial_slice = volume[:, :, center_axial]
coronal_slice = volume[:, center_coronal, :]
sagittal_slice = volume[center_sagittal, :, :]

# Step 5: Visualize the slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(axial_slice, cmap="gray")
axes[0].set_title("Axial Slice")
axes[1].imshow(coronal_slice, cmap="gray")
axes[1].set_title("Coronal Slice")
axes[2].imshow(sagittal_slice, cmap="gray")
axes[2].set_title("Sagittal Slice")
plt.show()
