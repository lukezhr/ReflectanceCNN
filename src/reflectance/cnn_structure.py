import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectanceCNN(nn.Module):
    def __init__(self, input_channels, input_length):
        super(ReflectanceCNN, self).__init__()
        kernel_size = [150, 100, 75, 50, 15, 5]  # 6 conv layers
        channels = [64, 64, 64, 64, 64, 64]
        
        # Conv1D + BatchNorm + MaxPool layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=channels[0],
                               kernel_size=kernel_size[0], stride=1, padding=(kernel_size[0] - 1) // 2)
        self.bn1 = nn.BatchNorm1d(num_features=channels[0])
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(in_channels=channels[0], out_channels=channels[1],
                               kernel_size=kernel_size[1], stride=1, padding=(kernel_size[1] - 1) // 2)
        self.bn2 = nn.BatchNorm1d(num_features=channels[1])
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(in_channels=channels[1], out_channels=channels[2],
                               kernel_size=kernel_size[2], stride=1, padding=(kernel_size[2] - 1) // 2)
        self.bn3 = nn.BatchNorm1d(num_features=channels[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        self.conv4 = nn.Conv1d(in_channels=channels[2], out_channels=channels[3],
                               kernel_size=kernel_size[3], stride=1, padding=(kernel_size[3] - 1) // 2)
        self.bn4 = nn.BatchNorm1d(num_features=channels[3])
        self.pool4 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv5 = nn.Conv1d(in_channels=channels[3], out_channels=channels[4],
                               kernel_size=kernel_size[4], stride=1, padding=(kernel_size[4] - 1) // 2)
        self.bn5 = nn.BatchNorm1d(num_features=channels[4])
        self.pool5 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.conv6 = nn.Conv1d(in_channels=channels[4], out_channels=channels[5],
                               kernel_size=kernel_size[5], stride=1, padding=(kernel_size[5] - 1) // 2)
        self.bn6 = nn.BatchNorm1d(num_features=channels[5])
        self.pool6 = nn.MaxPool1d(kernel_size=1, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc_size = self._get_conv_output(input_channels, input_length)

        # Fully connected layers with dropout
        self.fc1 = nn.Linear(self.fc_size, 3000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(3000, 1200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1200, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 6)

    def _get_conv_output(self, input_channels, input_length):
        # Dummy forward pass to calculate the flattened feature size.
        input_tensor = torch.autograd.Variable(torch.rand(1, input_channels, input_length))
        output = self._forward_features(input_tensor)
        n_size = output.data.reshape(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(self.fc3(x))
        x = self.fc4(x)
        return x