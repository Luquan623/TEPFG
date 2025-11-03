import torch
import numpy as np


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean.values
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std.values
        return (data * std) + mean


class Max_StandardScaler:
    """
    Standard the input
    """

    def __init__(self, _max, device):
        # self.max = _max
        self.max = torch.from_numpy(_max).float().to(device)

    def transform_max(self, data):
        return data / self.max

    def inverse_transform_max(self, data):
        return data * self.max