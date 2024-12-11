import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm


# Step 1: Custom Dataset for DICOM Files
class DICOMDataset(Dataset):
    def __init__(self, root_dir, labels_dict, transform=None):
        """
        Args:
            root_dir (str): Path to the root directory with DICOM files.
            labels_dict (dict): Mapping of folder names to class labels.
            transform (callable, optional): Transformations to apply to images.
        """
        self.root_dir = root_dir
        self.labels_dict = labels_dict
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Scan all DICOM files
        for folder, label in labels_dict.items():
            folder_path = os.path.join(root_dir, folder)
            dicom_files = glob(os.path.join(folder_path, "**/*.dcm"), recursive=True)
            self.image_paths.extend(dicom_files)
            self.labels.extend([label] * len(dicom_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load DICOM file
        dicom_path = self.image_paths[idx]
        label = self.labels[idx]
        dicom_data = pydicom.dcmread(dicom_path)
        img = dicom_data.pixel_array.astype(np.float32)

        # Normalize image
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Add channel dimension for PyTorch
        img = np.expand_dims(img, axis=0)

        # Apply transformations if any
        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


# Step 2: Prepare Data
def prepare_labels(adni_merge_path, dicom_root_dir):
    # Load ADNIMERGE dataset
    adnimerge = pd.read_csv(adni_merge_path)

    # Filter relevant columns
    adnimerge = adnimerge[["PTID", "DX", "DX_bl"]]

    # Map missing DX to DX_bl
    adnimerge["diagnosis"] = adnimerge["DX"].fillna(adnimerge["DX_bl"])

    # Map diagnosis to CN, MCI, AD
    label_mapping = {"CN": "CN", "MCI": "MCI", "Dementia": "AD"}
    adnimerge["diagnosis"] = adnimerge["diagnosis"].map(label_mapping)

    # Filter for valid diagnoses
    adnimerge = adnimerge.dropna(subset=["diagnosis"])

    # Build label dictionary for folder mapping
    label_dict = {}
    for folder in os.listdir(dicom_root_dir):
        patient_id = folder.split("_")[2]  # Extract PTID from folder name
        diagnosis = adnimerge.loc[adnimerge["PTID"] == patient_id, "diagnosis"]
        if not diagnosis.empty:
            label_dict[folder] = {"CN": 0, "MCI": 1, "AD": 2}[diagnosis.values[0]]

    return label_dict


# Step 3: Train-Test Split
def split_data(dataset, test_size=0.2):
    dataset_size = len(dataset)
    test_size = int(test_size * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


# Step 4: Define Model
class PetClassifier(torch.nn.Module):
    def __init__(self):
        super(PetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 3)

    def forward(self, x):
        return self.model(x)


# Step 5: Train Function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return running_loss / len(dataloader), accuracy


# Step 6: Evaluate Function
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred


# Step 7: Main Training and Evaluation Loop
def main():
    dicom_root_dir = "data/imaging/pet/ADNI"
    adnimerge_path = "data/study_files/ADNIMERGE_10Oct2024.csv"

    # Prepare labels and dataset
    labels_dict = prepare_labels(adnimerge_path, dicom_root_dir)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DICOMDataset(dicom_root_dir, labels_dict, transform=transform)

    # Split into train and test sets
    train_dataset, test_dataset = split_data(dataset)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize model, criterion, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PetClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(
            model, train_loader, criterion, optimizer, device
        )
        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}"
        )

    # Evaluate the model
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # Save model
    torch.save(model.state_dict(), "pet_classifier.pth")


if __name__ == "__main__":
    main()
