import numpy as np
import torch
from atp.metric.metric import MetricAbstract

def huber_loss(pred, true, mask_value=None, delta=1.0):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    residual = torch.abs(pred - true)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))
    # lo = torch.nn.SmoothL1Loss()
    # return lo(preds, labels)


class MAE(MetricAbstract):
    def __init__(self):
        self.bigger = False

    def __call__(self, groundtruth, pred) -> float:

        return np.mean(np.abs(pred - groundtruth)).item()


class MAE_E(MetricAbstract):
    def __init__(self):
        self.bigger = False

    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices
        mae_extreme = np.mean(np.abs(pred[extreme_indices] - groundtruth[extreme_indices])) if np.any(extreme_indices) else 0.0
        # mae_normal = np.mean(np.abs(pred[normal_indices] - groundtruth[normal_indices])) if np.any(normal_indices) else 0.0

        return mae_extreme

class MAE_N(MetricAbstract):
    def __init__(self):
        self.bigger = False
    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices
        # mae_extreme = np.mean(np.abs(pred[extreme_indices] - groundtruth[extreme_indices])) if np.any(extreme_indices) else 0.0
        mae_normal = np.mean(np.abs(pred[normal_indices] - groundtruth[normal_indices])) if np.any(normal_indices) else 0.0

        return mae_normal

class RMSE(MetricAbstract):
    def __init__(self):
        self.bigger = False
    def __call__(self, groundtruth, pred) -> float:
        return np.sqrt(np.mean((pred - groundtruth)**2)).item()
class RMSE_E(MetricAbstract):
    def __init__(self):
        self.bigger = False
    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices

        rmse_extreme = (
            np.sqrt(np.mean((pred[extreme_indices] - groundtruth[extreme_indices])**2))
            if np.any(extreme_indices)
            else 0.0
        )
        # rmse_normal = (
        #     np.sqrt(np.mean((pred[normal_indices] - groundtruth[normal_indices])**2))
        #     if np.any(normal_indices)
        #     else 0.0
        # )
        return rmse_extreme

class RMSE_N(MetricAbstract):
    def __init__(self):
        self.bigger = False
    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices
        rmse_normal = (
            np.sqrt(np.mean((pred[normal_indices] - groundtruth[normal_indices])**2))
            if np.any(normal_indices)
            else 0.0
        )
        return rmse_normal

class MSE(MetricAbstract):
    def __init__(self):
        self.bigger = False

    def __call__(self, groundtruth, pred) -> float:
        mask = np.where(groundtruth != (0), True, False)
        groundtruth = groundtruth[mask]
        pred = pred[mask]
        return np.mean((pred - groundtruth)**2).item()

class MAPE(MetricAbstract):
    def __init__(self):
        self.bigger = False

    def __call__(self, groundtruth, pred) -> float:
        mask = np.where(groundtruth != (0), True, False)
        groundtruth = groundtruth[mask]
        pred = pred[mask]
        return np.mean(np.abs((pred - groundtruth) / groundtruth)).item()


class Precision(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        TP = np.sum((pred == 1) & (groundtruth == 1))
        FP = np.sum((pred == 1) & (groundtruth == 0))

        precision = TP / (TP + FP + 1e-8)
        return precision.item()

class Recall(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        TP = np.sum((pred == 1) & (groundtruth == 1))
        FN = np.sum((pred == 0) & (groundtruth == 1))

        recall = TP / (TP + FN + 1e-8)
        return recall.item()

class F1Score(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        TP = np.sum((pred == 1) & (groundtruth == 1))
        FP = np.sum((pred == 1) & (groundtruth == 0))
        FN = np.sum((pred == 0) & (groundtruth == 1))

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1.item()

class Accuracy(MetricAbstract):
    def __init__(self):
        self.bigger = True

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        correct = np.sum(pred == groundtruth)
        total = groundtruth.size

        acc = correct / (total + 1e-8)
        return acc.item()
class MAE_W(MetricAbstract):
    def __init__(self):
        self.bigger = False

    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std
        extreme_mask = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        weights = np.where(extreme_mask, 10.0, 1.0)
        abs_error = np.abs(pred - groundtruth)
        weighted_mae = np.sum(weights * abs_error) / np.sum(weights)

        return weighted_mae

class RMSE_W(MetricAbstract):
    def __init__(self):
        self.bigger = False
    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std
        extreme_mask = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        weights = np.where(extreme_mask, 10.0, 1.0)
        sq_error = (pred - groundtruth) ** 2
        weighted_mse = np.sum(weights * sq_error) / np.sum(weights)
        weighted_rmse = np.sqrt(weighted_mse)

        return weighted_rmse




