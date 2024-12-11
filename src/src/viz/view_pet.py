import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Path to your .dcm file
file_path = "data/imaging/pet/ADNI/003_S_10073/ADNI4_florbetapir__AC_/2024-06-04_15_17_02.0/I10667630/ADNI_003_S_10073_PT_ADNI4_florbetapir_(AC)_br_raw_20240619183607354_44.dcm"

# Load the DICOM file
dicom_data = pydicom.dcmread(file_path)

# Extract pixel data
pixel_array = dicom_data.pixel_array

# Display the image
plt.imshow(pixel_array, cmap="gray")
plt.title("DICOM PET Image")
plt.colorbar()
plt.show()

# Step 3: Print DICOM metadata
print("\nDICOM Metadata:\n")
for elem in dicom_data:
    if elem.VR != "SQ":  # Skip sequence data for simplicity
        print(f"{elem.name}: {elem.value}")

