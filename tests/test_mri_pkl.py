import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the .pkl file
pkl_file_path = "./data/processed/mri/mri_meta.pkl"
meta_all = pd.read_pickle(pkl_file_path)

# Inspect the contents
print(f"DataFrame shape: {meta_all.shape}")
print("Column names:", meta_all.columns)
print(meta_all.head())

# Print all unique labels
unique_labels = meta_all["label"].value_counts()
print(f"Unique labels (image_label): {unique_labels}")

# Visualize an example MRI's slices
# Pick the first MRI entry
example = meta_all.iloc[0]  # Adjust index if needed
img_array = example["img_array"]  # Extract the image array
label = example["label"]  # Extract the label
subject = example["subject"]  # Extract the subject ID

# Ensure the array has the expected shape
print(f"Image array shape: {img_array.shape}")

# Plot the three slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_array[:, :, 0], cmap="gray")
axes[0].set_title("Axial Slice")
axes[1].imshow(img_array[:, :, 1], cmap="gray")
axes[1].set_title("Coronal Slice")
axes[2].imshow(img_array[:, :, 2], cmap="gray")
axes[2].set_title("Sagittal Slice")

# Add additional details
plt.suptitle(f"Subject: {subject}, Label: {label}")
plt.show()
