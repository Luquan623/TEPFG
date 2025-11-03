'''
Always evaluate the model with MAE, RMSE, MAPE, RRSE, PNBI, and oPNBI.
Why add mask to MAE and RMSE?
    Filter the 0 that may be caused by error (such as loop sensor)
Why add mask to MAPE and MARE?
    Ignore very small values (e.g., 0.5/0.5=100%)
'''
import numpy as np
import torch
from atp.metric.metric import MetricAbstract

def MAE_torch(pred, true, mask_value=None):
    if mask_value != None: # mask_value 参数是一个可选参数，用于指定一个阈值。如果提供了该值，那么只有真实标签大于 mask_value 的部分才会被考虑进计算。
        mask = torch.gt(true, mask_value).to('cuda:0') # 这行代码生成一个布尔型的掩码张量（mask），它的值是 True 或 False，表示 true 中的每个元素是否大于 mask_value。
        pred = torch.masked_select(pred, mask) # 根据布尔型掩码 mask，从 pred 张量中选择相应位置的元素。
        true = torch.masked_select(true, mask) #  从 true 张量中选择与 mask 对应的元素，只保留真实值大于 mask_value 的部分。
        true_count = torch.sum(mask).item() # 计算掩码中 True 元素的总数，即真实标签 true 中大于 mask_value 的元素的个数。
    else:
        true_count = None
    mae_loss = torch.abs(true - pred) # 计算预测值 pred 和真实标签 true 之间的绝对误差（MAE），即每个位置的误差值为 |true - pred|。

    # print(mae_loss[mae_loss>3].shape, mae_loss[mae_loss<1].shape, mae_loss.shape)
    return torch.mean(mae_loss), true_count

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

def MSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        true_count = torch.sum(mask).item()
    else:
        true_count = None
    return torch.mean((pred - true) ** 2), true_count

def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        true_count = torch.sum(mask).item()
    else:
        true_count = None
    return torch.sqrt(torch.mean((pred - true) ** 2)), true_count

def RRSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))

def CORR_torch(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        pred = pred.transpose(1, 2).unsqueeze(dim=1)
        true = true.transpose(1, 2).unsqueeze(dim=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(2, 3)
        true = true.transpose(2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(dim=dims)
    true_mean = true.mean(dim=dims)
    pred_std = pred.std(dim=dims)
    true_std = true.std(dim=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(dim=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
        true_count = torch.sum(mask).item()
    else:
        true_count = None
        # print(true[true<1].shape, true[true<0.0001].shape, true[true==0].shape)
        # print(true)
    return torch.mean(torch.abs(torch.div((true - pred), true))), true_count

def PNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()

def oPNBI_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true+pred) / (2*true)
    return bias.mean()

def MARE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))

def SMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true-pred)/(torch.abs(true)+torch.abs(pred)))


def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def MSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MSE = np.mean(np.square(pred-true))
    return MSE

def RMSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

#Root Relative Squared Error
def RRSE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    mean = true.mean()
    return np.divide(np.sqrt(np.sum((pred-true) ** 2)), np.sqrt(np.sum((true-mean) ** 2)))

def MAPE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), true)))

def PNBI_np(pred, true, mask_value=None):
    #if PNBI=0, all pred are smaller than true
    #if PNBI=1, all pred are bigger than true
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = pred-true
    indicator = np.where(bias>0, True, False)
    return indicator.mean()

def oPNBI_np(pred, true, mask_value=None):
    #if oPNBI>1, pred are bigger than true
    #if oPNBI<1, pred are smaller than true
    #however, this metric is too sentive to small values. Not good!
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    bias = (true + pred) / (2 * true)
    return bias.mean()

def MARE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true> (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.divide(np.sum(np.absolute((true - pred))), np.sum(true))

def CORR_np(pred, true, mask_value=None):
    #input B, T, N, D or B, N, D or B, N
    if len(pred.shape) == 2:
        #B, N
        pred = pred.unsqueeze(dim=1).unsqueeze(dim=1)
        true = true.unsqueeze(dim=1).unsqueeze(dim=1)
    elif len(pred.shape) == 3:
        #np.transpose include permute, B, T, N
        pred = np.expand_dims(pred.transpose(0, 2, 1), axis=1)
        true = np.expand_dims(true.transpose(0, 2, 1), axis=1)
    elif len(pred.shape)  == 4:
        #B, T, N, D -> B, T, D, N
        pred = pred.transpose(0, 1, 2, 3)
        true = true.transpose(0, 1, 2, 3)
    else:
        raise ValueError
    dims = (0, 1, 2)
    pred_mean = pred.mean(axis=dims)
    true_mean = true.mean(axis=dims)
    pred_std = pred.std(axis=dims)
    true_std = true.std(axis=dims)
    correlation = ((pred - pred_mean)*(true - true_mean)).mean(axis=dims) / (pred_std*true_std)
    index = (true_std != 0)
    correlation = (correlation[index]).mean()
    return correlation

def All_Metrics(pred, true, mask1, mask2):
    #mask1 filter the very small value, mask2 filter the value lower than a defined threshold
    assert type(pred) == type(true)
    if type(pred) == np.ndarray:
        mae  = MAE_np(pred, true, mask1)
        rmse = RMSE_np(pred, true, mask1)
        mse = MSE_np(pred, true, mask1)
        mape = MAPE_np(pred, true, mask2)
        rrse = RRSE_np(pred, true, mask1)
        # corr = 0
        corr = CORR_np(pred, true, mask1)
        #pnbi = PNBI_np(pred, true, mask1)
        #opnbi = oPNBI_np(pred, true, mask2)
    elif type(pred) == torch.Tensor:
        mae, mae_count = MAE_torch(pred, true, mask1)
        rmse, rmse_count = RMSE_torch(pred, true, mask1)
        mse, mse_count = MSE_torch(pred, true, mask1)
        mape, mape_count = MAPE_torch(pred, true, mask2)
        rrse = RRSE_torch(pred, true, mask1)
        corr = CORR_torch(pred, true, mask1)
        #pnbi = PNBI_torch(pred, true, mask1)
        #opnbi = oPNBI_torch(pred, true, mask2)
    else:
        raise TypeError
    return mae, rmse, mape, mse, corr, mae_count, rmse_count, mse_count, mape_count

def SIGIR_Metrics(pred, true, mask1, mask2):
    rrse = RRSE_torch(pred, true, mask1)
    corr = CORR_torch(pred, true, 0)
    return rrse, corr

class MAE(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred) -> float:

        return np.mean(np.abs(pred - groundtruth)).item()


class MAE_E(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        """
        计算极端值和正常值的 MAE。

        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值

        返回：
        - (mae_extreme, mae_normal): tuple, 分别为极端值和正常值的 MAE
        """
        # # 计算真实值的均值和标准差
        # mean = np.mean(groundtruth)
        # std = np.std(groundtruth)

        # 定义极端值的范围
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 找出极端值和正常值的索引
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices

        # 分别计算极端值和正常值的 MAE
        mae_extreme = np.mean(np.abs(pred[extreme_indices] - groundtruth[extreme_indices])) if np.any(extreme_indices) else 0.0
        # mae_normal = np.mean(np.abs(pred[normal_indices] - groundtruth[normal_indices])) if np.any(normal_indices) else 0.0

        return mae_extreme

class MAE_N(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        """
        计算极端值和正常值的 MAE。

        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值

        返回：
        - (mae_extreme, mae_normal): tuple, 分别为极端值和正常值的 MAE
        """
        # # 计算真实值的均值和标准差
        # mean = np.mean(groundtruth)
        # std = np.std(groundtruth)

        # 定义极端值的范围
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 找出极端值和正常值的索引
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices

        # 分别计算极端值和正常值的 MAE
        # mae_extreme = np.mean(np.abs(pred[extreme_indices] - groundtruth[extreme_indices])) if np.any(extreme_indices) else 0.0
        mae_normal = np.mean(np.abs(pred[normal_indices] - groundtruth[normal_indices])) if np.any(normal_indices) else 0.0

        return mae_normal

class RMSE(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred) -> float:

        return np.sqrt(np.mean((pred - groundtruth)**2)).item()

class RMSE_E(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        """
        计算极端值和正常值的 RMSE。

        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值

        返回：
        - (rmse_extreme, rmse_normal): tuple, 分别为极端值和正常值的 RMSE
        """
        # 检查输入的形状是否一致
        if groundtruth.shape != pred.shape:
            raise ValueError("groundtruth 和 pred 的形状必须一致")

        # 计算真实值的均值和标准差
        # mean = np.mean(groundtruth)
        # std = np.std(groundtruth)

        # 定义极端值的范围（均值的 ±1.6 倍标准差之外为极端值）
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 找出极端值和正常值的索引
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices

        # 分别计算极端值和正常值的 RMSE
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
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred,extreme_max,extreme_min,mean,std) ->  float:
        """
        计算极端值和正常值的 RMSE。

        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值

        返回：
        - (rmse_extreme, rmse_normal): tuple, 分别为极端值和正常值的 RMSE
        """
        # 检查输入的形状是否一致
        if groundtruth.shape != pred.shape:
            raise ValueError("groundtruth 和 pred 的形状必须一致")

        # # 计算真实值的均值和标准差
        # mean = np.mean(groundtruth)
        # std = np.std(groundtruth)

        # 定义极端值的范围（均值的 ±1.6 倍标准差之外为极端值）
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 找出极端值和正常值的索引
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        normal_indices = ~extreme_indices

        # 分别计算极端值和正常值的 RMSE
        # rmse_extreme = (
        #     np.sqrt(np.mean((pred[extreme_indices] - groundtruth[extreme_indices])**2))
        #     if np.any(extreme_indices)
        #     else 0.0
        # )
        rmse_normal = (
            np.sqrt(np.mean((pred[normal_indices] - groundtruth[normal_indices])**2))
            if np.any(normal_indices)
            else 0.0
        )

        return rmse_normal

class MSE(MetricAbstract):          # ******
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred) -> float:
        mask = np.where(groundtruth != (0), True, False)
        groundtruth = groundtruth[mask]
        pred = pred[mask]
        return np.mean((pred - groundtruth)**2).item()


# class MAE(MetricAbstract):          # 会过滤掉真实值中的0值
#     def __init__(self):
#         self.bigger = False  # 指标越小越好
#
#     def __call__(self, groundtruth, pred) -> float:
#         mask = np.where(groundtruth != (0), True, False)
#         groundtruth = groundtruth[mask]
#         pred = pred[mask]
#         return np.mean(np.abs(pred - groundtruth)).item()


class MAPE(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred) -> float:
        # mask filter the value lower than a defined threshold
        mask = np.where(groundtruth != (0), True, False)
        groundtruth = groundtruth[mask]
        pred = pred[mask]
        return np.mean(np.abs((pred - groundtruth) / groundtruth)).item()

class MAPE_E(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        if groundtruth.shape != pred.shape:
            raise ValueError("groundtruth 和 pred 的形状必须一致")

        # 定义极端值范围
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 极端值索引
        extreme_indices = (groundtruth < lower_bound) | (groundtruth > upper_bound)

        # 有效值（groundtruth 不为 0）索引
        valid_mask = (groundtruth != 0)

        # 最终有效的极端索引
        final_mask = extreme_indices & valid_mask

        if np.any(final_mask):
            gt_filtered = groundtruth[final_mask]
            pred_filtered = pred[final_mask]
            return np.mean(np.abs((pred_filtered - gt_filtered) / gt_filtered)).item()
        else:
            return 0.0

class MAPE_N(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        if groundtruth.shape != pred.shape:
            raise ValueError("groundtruth 和 pred 的形状必须一致")

        # 定义极端值范围
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 正常值索引
        normal_indices = (groundtruth >= lower_bound) & (groundtruth <= upper_bound)

        # 有效值（groundtruth 不为 0）索引
        valid_mask = (groundtruth != 0)

        # 最终有效的正常值索引
        final_mask = normal_indices & valid_mask

        if np.any(final_mask):
            gt_filtered = groundtruth[final_mask]
            pred_filtered = pred[final_mask]
            return np.mean(np.abs((pred_filtered - gt_filtered) / gt_filtered)).item()
        else:
            return 0.0



class SMAPE(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred) -> float:
        # mask filter the value lower than a defined threshold
        mask = np.where(groundtruth != (0), True, False)
        groundtruth = groundtruth[mask]
        pred = pred[mask]
        return 2.0 * np.mean(np.abs(pred - groundtruth) / (np.abs(pred) + np.abs(groundtruth))) * 100

class Precision(MetricAbstract):
    def __init__(self):
        self.bigger = True  # 精度越大越好

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        TP = np.sum((pred == 1) & (groundtruth == 1))
        FP = np.sum((pred == 1) & (groundtruth == 0))

        precision = TP / (TP + FP + 1e-8)  # 避免除以0
        return precision.item()

class Recall(MetricAbstract):
    def __init__(self):
        self.bigger = True  # 召回率越大越好

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        TP = np.sum((pred == 1) & (groundtruth == 1))
        FN = np.sum((pred == 0) & (groundtruth == 1))

        recall = TP / (TP + FN + 1e-8)
        return recall.item()

class F1Score(MetricAbstract):
    def __init__(self):
        self.bigger = True  # F1 越大越好

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
        self.bigger = True  # 准确率越高越好

    def __call__(self, groundtruth, pred) -> float:
        pred = np.asarray(pred).astype(int)
        groundtruth = np.asarray(groundtruth).astype(int)

        correct = np.sum(pred == groundtruth)
        total = groundtruth.size

        acc = correct / (total + 1e-8)  # 避免除以0
        return acc.item()
class MAE_W(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好

    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        """
        计算极值加权的 MAE。

        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值
        - extreme_max/extreme_min: z-score 阈值
        - mean/std: 用于标准化还原的统计值

        返回：
        - 加权 MAE 值（极值误差更重要）
        """
        # 1. 定义极值范围（基于 z-score）
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 2. 构造权重：极值权重大（如 10.0），正常值为 1.0
        extreme_mask = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        weights = np.where(extreme_mask, 10.0, 1.0)  # 可调参数：极值权重

        # 3. 加权 MAE 计算
        abs_error = np.abs(pred - groundtruth)
        weighted_mae = np.sum(weights * abs_error) / np.sum(weights)

        return weighted_mae

class RMSE_W(MetricAbstract):
    def __init__(self):
        self.bigger = False  # 指标越小越好
    def __call__(self, groundtruth, pred, extreme_max, extreme_min, mean, std) -> float:
        """
        计算极值加权 RMSE。
        参数：
        - groundtruth: ndarray, 真实值
        - pred: ndarray, 预测值
        返回：
        - 加权 RMSE 值（极值部分误差加权放大）
        """
        # 形状检查
        if groundtruth.shape != pred.shape:
            raise ValueError("groundtruth 和 pred 的形状必须一致")

        # 定义极值范围
        lower_bound = mean - extreme_min * std
        upper_bound = mean + extreme_max * std

        # 构建极值 mask
        extreme_mask = (groundtruth < lower_bound) | (groundtruth > upper_bound)
        weights = np.where(extreme_mask, 10.0, 1.0)  # 极值权重设为 5.0，正常值为 1.0（可调）

        # 加权 RMSE 计算
        sq_error = (pred - groundtruth) ** 2
        weighted_mse = np.sum(weights * sq_error) / np.sum(weights)
        weighted_rmse = np.sqrt(weighted_mse)

        return weighted_rmse


if __name__ == '__main__':
    pred = torch.Tensor([1, 2, 3,4])
    true = torch.Tensor([2, 1, 4,5])
    print(All_Metrics(pred, true, None, None))

