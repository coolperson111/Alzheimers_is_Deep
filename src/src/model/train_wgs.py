import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, 32)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return self.softmax(x)

def save_confusion_matrix(y_true, y_pred, file_path, class_names):
    """Save a confusion matrix plot to a file."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()

def main():
    # Load data
    X_train = pd.read_pickle("X_train_vcf.pkl").values
    y_train = pd.read_pickle("y_train_vcf.pkl").values
    X_test = pd.read_pickle("X_test_vcf.pkl").values
    y_test = pd.read_pickle("y_test_vcf.pkl").values

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Prepare DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_split = int(0.1 * len(train_dataset))
    train_split = len(train_dataset) - val_split
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_split, val_split])
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Hyperparameters
    input_dim = X_train.shape[1]
    num_classes = 3
    num_epochs = 50
    learning_rate = 0.001

    # Initialize model
    model = MLPModel(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += y_val.size(0)
                val_correct += (predicted == y_val).sum().item()

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Accuracy: {val_correct/val_total}')

    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    all_true_labels = []
    all_predicted_labels = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            test_total += y_test.size(0)
            test_correct += (predicted == y_test).sum().item()
            all_true_labels.extend(y_test.numpy())
            all_predicted_labels.extend(predicted.numpy())

    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc}')

    # Save confusion matrix
    confusion_matrix_path = "confusion_matrix.png"
    class_names = ['Class 0', 'Class 1', 'Class 2']  # Replace with your class names
    save_confusion_matrix(all_true_labels, all_predicted_labels, confusion_matrix_path, class_names)
    print(f'Confusion matrix saved to {confusion_matrix_path}')

if __name__ == '__main__':
    main()