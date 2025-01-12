import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader, Subset
from torch.utils.data import random_split

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

# 7 conv layers
class ReflectanceCNN(nn.Module):
    def __init__(self, input_channels, input_length):
        super(ReflectanceCNN, self).__init__()
        kernel_size = [150, 100, 75, 50, 15, 5, 3] # 7 conv layers at most
        channels = [64, 64, 64, 64, 64, 64, 64] # num of filters at each conv layer
        # Conv1D + BatchNorm + MaxPool
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=channels[0], kernel_size=kernel_size[0], stride=1, padding=(kernel_size[0] - 1) // 2) # 'same' padding
        self.bn1 = nn.BatchNorm1d(num_features=channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size[1], stride=1, padding=(kernel_size[1] - 1) // 2)
        self.bn2 = nn.BatchNorm1d(num_features=channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size[2], stride=1, padding=(kernel_size[2] - 1) // 2)
        self.bn3 = nn.BatchNorm1d(num_features=channels[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernel_size[3], stride=1, padding=(kernel_size[3] - 1) // 2)
        self.bn4 = nn.BatchNorm1d(num_features=channels[3])
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv5 = nn.Conv1d(in_channels=channels[3], out_channels=channels[4], kernel_size=kernel_size[4], stride=1, padding=(kernel_size[4] - 1) // 2)
        self.bn5 = nn.BatchNorm1d(num_features=channels[4])
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv6 = nn.Conv1d(in_channels=channels[4], out_channels=channels[5], kernel_size=kernel_size[5], stride=1, padding=(kernel_size[5] - 1) // 2)
        self.bn6 = nn.BatchNorm1d(num_features=channels[5])
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv7 = nn.Conv1d(in_channels=channels[5], out_channels=channels[6], kernel_size=kernel_size[6], stride=1, padding=(kernel_size[6] - 1) // 2)
        self.bn7 = nn.BatchNorm1d(num_features=channels[6])
        self.pool7 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc_size = self._get_conv_output(input_channels, input_length)

        # Fully Connected Layers + Dropout
        self.fc1 = nn.Linear(self.fc_size, 3000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(3000, 1200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1200, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 6)

    def _get_conv_output(self, input_channels, input_length):
        # Dummy pass to get the output size
        input = torch.autograd.Variable(torch.rand(1, input_channels, input_length))
        output = self._forward_features(input)
        n_size = output.data.reshape(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        # x = self.pool7(F.relu(self.bn7(self.conv7(x)))) # comment out to just use 6 feature extraction layers
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3((self.fc3(x)))
        x = self.fc4(x)
        return x
    
# Dataset
class ReflectanceDataset(Dataset):
    def __init__(self, features_file, labels_file):
        self.features = np.loadtxt(features_file, delimiter=',')
        print("Features loaded, shape:", self.features.shape)  # Debug print
        self.labels = np.loadtxt(labels_file, delimiter=',')
        print("Labels loaded, shape:", self.labels.shape)  # Debug print

        self.features = self.features.reshape(-1, 1, self.features.shape[-1])
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        return features_tensor, self.labels[idx]


# Early Stopping Setup
class EarlyStopping:
    def __init__(self, patience, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when the validation loss decrease.'''
        print(f'Validation loss decreased ({self.best_score} --> {val_loss}).  Saving model ...')
        torch.save(model, self.path)


def train_model_k_fold(model, dataset, criterion, optimizer, n_epochs, device, k_folds=5, model_path='model.pth'):
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Storing fold results for plotting
    all_train_losses = []
    all_val_losses = []
    all_train_r2 = []
    all_val_r2 = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold+1}')
        print('--------------------------------')

        train_subsampler = Subset(dataset, train_ids)
        val_subsampler = Subset(dataset, val_ids)

        train_loader = DataLoader(train_subsampler, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_subsampler, batch_size=100)

        early_stopping = EarlyStopping(patience=10, path=model_path)  # Path updated for each fold if needed

        fold_train_losses = []
        fold_val_losses = []
        fold_train_r2_scores = []
        fold_val_r2_scores = []

        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            all_labels = []
            all_outputs = []

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                all_labels.append(labels.cpu().detach().numpy())
                all_outputs.append(outputs.cpu().detach().numpy())

            train_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_outputs))
            fold_train_r2_scores.append(train_r2)
            fold_train_losses.append(running_loss / len(train_loader))

            model.eval()
            val_loss = 0.0
            val_labels = []
            val_outputs = []

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    val_labels.append(labels.cpu().numpy())
                    val_outputs.append(outputs.cpu().numpy())

            val_r2 = r2_score(np.concatenate(val_labels), np.concatenate(val_outputs))
            fold_val_r2_scores.append(val_r2)
            fold_val_losses.append(val_loss / len(val_loader))

            early_stopping(val_loss / len(val_loader), model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        all_train_losses.append(fold_train_losses)
        all_val_losses.append(fold_val_losses)
        all_train_r2.append(fold_train_r2_scores)
        all_val_r2.append(fold_val_r2_scores)

        model = torch.load(model_path)  # Optionally load the best model saved by early stopping

    # Plotting
    plot_k_fold_results(all_train_losses, all_val_losses, all_train_r2, all_val_r2)

def plot_k_fold_results(all_train_losses, all_val_losses, all_train_r2, all_val_r2):
    mean_train_losses = np.mean(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    std_train_losses = np.std(all_train_losses, axis=0)
    std_val_losses = np.std(all_val_losses, axis=0)

    mean_train_r2 = np.mean(all_train_r2, axis=0)
    mean_val_r2 = np.mean(all_val_r2, axis=0)
    std_train_r2 = np.std(all_train_r2, axis=0)
    std_val_r2 = np.std(all_val_r2, axis=0)

    epochs = range(1, len(mean_train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.errorbar(epochs, mean_train_losses, yerr=std_train_losses, label='Train Loss', fmt='-o')
    plt.errorbar(epochs, mean_val_losses, yerr=std_val_losses, label='Validation Loss', fmt='-o')
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.errorbar(epochs, mean_train_r2, yerr=std_train_r2, label='Train R²', fmt='-o')
    plt.errorbar(epochs, mean_val_r2, yerr=std_val_r2, label='Validation R²', fmt='-o')
    plt.title('Training and Validation R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('train_val_loss.png', dpi=300)
    plt.show()


# Function to train model with plotting at the end
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs, device, model_path):
    early_stopping = EarlyStopping(patience=10, path=model_path)  # Early stopping utility

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        all_labels = []
        all_outputs = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            # Collect labels and outputs for R² calculation
            all_labels.append(labels.cpu().detach().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

        # Compute training R² score
        train_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_outputs))
        train_r2_scores.append(train_r2)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_labels = []
        val_outputs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Collect labels and outputs for R² calculation
                val_labels.append(labels.cpu().numpy())
                val_outputs.append(outputs.cpu().numpy())

        # Compute validation R² score
        val_r2 = r2_score(np.concatenate(val_labels), np.concatenate(val_outputs))
        val_r2_scores.append(val_r2)

        # Update training and validation losses
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Validation Loss = {val_losses[-1]:.4f}, Train R² = {train_r2:.4f}, Validation R² = {val_r2:.4f}")

        # Early stopping check
        early_stopping(val_losses[-1], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Plotting after training completion
    plt.figure(figsize=(12, 6))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation R²
    plt.subplot(1, 2, 2)
    plt.plot(train_r2_scores, label='Train R²')
    plt.plot(val_r2_scores, label='Validation R²')
    plt.title('Training and Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('train_val_loss.png', dpi=300)
    plt.show()

    # Optionally load the best model if early stopping was used
    model = torch.load(model_path)
    return model



# Define a evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev_score = explained_variance_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Explained Variance Score: {ev_score:.4f}")

# Adjust these paths to where you saved your split data
model_number = "thesis"
model_name = "model_N50000_d200_6_conv_layers_lr1e-4"
model_path = f'/content/drive/MyDrive/model_{model_number}/{model_name}.pth'
suffix = 'csv'
# Load the full dataset
full_dataset = ReflectanceDataset(f"/content/drive/MyDrive/model_{model_number}/X_train_N50000_d200.{suffix}",
                                  f"/content/drive/MyDrive/model_{model_number}/Y_train_N50000_d200.{suffix}")

# Set a seed for the random number generator
torch.manual_seed(0)

# # Determine the lengths of train/validation splits
# train_len = int(len(full_dataset) * 0.8)  # 80% for training
# val_len = len(full_dataset) - train_len  # 20% for validation

# # Split the dataset
# train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
# print('Number of samples in train_dataset:', len(train_dataset))
# print('Number of samples in val_dataset:', len(val_dataset))
# sample = train_dataset[0]
# print('Shape of a sample in train_dataset:', sample[0].shape)
# print(sample[1].shape)

# Create data loaders
batch_size = 100
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_length = full_dataset.features.shape[2]
input_channels = 1


model = ReflectanceCNN(input_channels=input_channels, input_length=input_length).to(device)
model = model.float()

# criterion = ReflectanceLoss(wavelengths=wavelengths, static_thicknesses=static_thicknesses, param_ranges=param_ranges, layers=layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
n_epochs = 200

# train the model
train_model_k_fold(model, full_dataset, criterion, optimizer, n_epochs=n_epochs, device=device, model_path=model_path)
# # Evaluate the model
# evaluate_model(model, test_loader, device)

# Evaluate the model
evaluate_model(model, test_loader, device)