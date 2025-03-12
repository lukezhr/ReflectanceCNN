import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
config = {
    "model_name": "model_N50000_d200_6_conv_layers_lr1e-4",
    "data_folder": os.path.join("data", "synthetic"),
    "batch_size": 32,
    "n_epochs": 5,
    "learning_rate": 1e-4,
    "k_folds": 5,
    "patience": 10,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
}

# Define paths
config["model_path"] = os.path.join("training", "models", "saved_models", f"{config['model_name']}.pth")
config["train_features_path"] = os.path.join(config["data_folder"], "X_train.csv")
config["train_labels_path"] = os.path.join(config["data_folder"], "Y_train.csv")
config["val_features_path"] = os.path.join(config["data_folder"], "X_val.csv")
config["val_labels_path"] = os.path.join(config["data_folder"], "Y_val.csv")

# Ensure directories exist
os.makedirs(os.path.dirname(config["model_path"]), exist_ok=True)


# Model Definition
class ReflectanceCNN(nn.Module):
    def __init__(self, input_channels, input_length):
        super(ReflectanceCNN, self).__init__()
        kernel_size = [150, 100, 75, 50, 15, 5] # six layers
        channels = [64, 64, 64, 64, 64, 64]

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        for i in range(len(kernel_size)):
            conv = nn.Conv1d(
                in_channels=input_channels if i == 0 else channels[i - 1],
                out_channels=channels[i],
                kernel_size=kernel_size[i],
                stride=1,
                padding=(kernel_size[i] - 1) // 2,
            )
            bn = nn.BatchNorm1d(channels[i])
            pool = nn.MaxPool1d(kernel_size=3 if i < 2 else 2, stride=2, padding=1 if i < 2 else 0)

            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            self.pool_layers.append(pool)

        self.flatten = nn.Flatten()
        self.fc_size = self._get_conv_output(input_channels, input_length)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_size, 3000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3000, 1200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1200, 300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, 6),
        )

    def _get_conv_output(self, input_channels, input_length):
        """
        Get the output size of the convolutional layers.
        """
        dummy_input = torch.rand(1, input_channels, input_length)
        output = self._forward_features(dummy_input)
        return output.numel()

    def _forward_features(self, x):
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = pool(F.relu(bn(conv(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


# Dataset
class ReflectanceDataset(Dataset):
    def __init__(self, features_file, labels_file, val_features_file=None, val_labels_file=None):
        self.features = np.loadtxt(features_file, delimiter=",")
        self.labels = np.loadtxt(labels_file, delimiter=",")
        self.features = self.features.reshape(-1, 1, self.features.shape[-1])
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        
        if val_features_file and val_labels_file:
            self.val_features = np.loadtxt(val_features_file, delimiter=",")
            self.val_labels = np.loadtxt(val_labels_file, delimiter=",")
            self.val_features = self.val_features.reshape(-1, 1, self.val_features.shape[-1])
            self.val_labels = torch.tensor(self.val_labels, dtype=torch.float32)
        else:
            self.val_features = None
            self.val_labels = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), self.labels[idx]


# Early Stopping
class EarlyStopping:
    def __init__(self, patience, delta=0, path="checkpoint.pth"):
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
            logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        logging.info(f"Validation loss decreased. Saving model to {self.path}")
        torch.save(model.state_dict(), self.path)


# Training Function
def train_model_with_validation(model, dataset, criterion, optimizer, n_epochs, device, model_path="model.pth"):
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    early_stopping = EarlyStopping(patience=config["patience"], path=model_path)

    train_losses, val_losses, train_r2, val_r2 = [], [], [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss, train_r2_score = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_r2_score = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2.append(train_r2_score)
        val_r2.append(val_r2_score)

        logging.info(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
            f"Train R² = {train_r2_score:.4f}, Val R² = {val_r2_score:.4f}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

    plot_training_results(train_losses, val_losses, train_r2, val_r2)


# Helper Functions
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_outputs = [], []

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        all_labels.append(labels.cpu().detach().numpy())
        all_outputs.append(outputs.cpu().detach().numpy())

    train_loss = running_loss / len(train_loader)
    train_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_outputs))
    return train_loss, train_r2


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_labels.append(labels.cpu().detach().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

    val_loss = val_loss / len(val_loader)
    val_r2 = r2_score(np.concatenate(all_labels), np.concatenate(all_outputs))
    return val_loss, val_r2


# Plotting Function
def plot_training_results(train_losses, val_losses, train_r2, val_r2):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_r2, label="Train R²")
    plt.plot(epochs, val_r2, label="Validation R²")
    plt.title("Training and Validation R² Score")
    plt.xlabel("Epoch")
    plt.ylabel("R² Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("train_val_loss.png", dpi=300)
    plt.show()


# Main Execution
if __name__ == "__main__":
    # At start of training
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)

    # Load dataset
    dataset = ReflectanceDataset(
        config["train_features_path"],
        config["train_labels_path"],
        config["val_features_path"],
        config["val_labels_path"]
    )

    # Initialize model
    model = ReflectanceCNN(input_channels=1, input_length=dataset.features.shape[2]).to(config["device"])
    model = model.float()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train the model
    train_model_with_validation(model, dataset, criterion, optimizer, config["n_epochs"], config["device"], config["model_path"])

    # Save the model
    torch.save(model.state_dict(), config["model_path"])
