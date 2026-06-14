import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=1, confidence_scale=2.0):
        super(Detector, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features // 2)
        self.bn2 = nn.BatchNorm1d(hidden_features // 2)
        self.fc3 = nn.Linear(hidden_features // 2, out_features)
        self.sigmoid = nn.Sigmoid()
        self.confidence_scale = confidence_scale  # 用于增强输出信心

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        # 使用缩放因子增强信心
        x = self.sigmoid(x * self.confidence_scale)
        return x
