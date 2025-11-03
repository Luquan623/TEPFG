import numpy as np
import pandas as pd
from ts_forecasting_traffic.dataset.ts_forecasting_traffic_staeformer import define_dataloder,define_dataloder_standard
from atp.metric.metric import MetricAbstract
from atp.model.model import ModelAbstract
import torch
import os
from torch.utils.data import  DataLoader
class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data): # 标准化函数
        return (data - self.mean) / self.std

    def inverse_transform(self, data): # 逆标准化函数
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray: #首先检查 data 是否是 PyTorch 的 Tensor 类型，并且 mean 是否是 NumPy 的数组 np.ndarray 类型
            # 使用 torch.from_numpy 将 NumPy 数组 self.std 和 self.mean 转换为 PyTorch 张量，确保它们与输入 data 的设备（data.device）和数据类型（data.dtype）一致。
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


def normalize_dataset(data, input_base_dim): # input_base_dim：输入数据的基本维度，用于区分原始数据与其他特征（如天数据和周数据）。
    # 切分数据
    data_ori = data[:, :, 0:input_base_dim]
    data_day = data[:, :, input_base_dim:input_base_dim+1]
    data_week = data[:, :, input_base_dim+1:input_base_dim+2]

    # 计算均值与标准差
    mean_data = data_ori.mean()
    std_data = data_ori.std()
    mean_day = data_day.mean()
    std_day = data_day.std()
    mean_week = data_week.mean()
    std_week = data_week.std()

    # 标准化器的创建
    scaler_data = StandardScaler(mean_data, std_data)
    scaler_day = StandardScaler(mean_day, std_day)
    scaler_week = StandardScaler(mean_week, std_week)
    print('Normalize the dataset by Standard Normalization')

    return scaler_data, scaler_day, scaler_week

class TSForecastingTVT:

    def __init__(self):
        self.progress = np.zeros(1, dtype=float)  # 保存各个进程的进度
        self.progress_train_proportion = 0.9  # 计算训练作为测试工作进度的比例


        self.add_time_stamp = False
        self.informer = False
        self.label_len = 0
        self.input_size = 12
        self.output_size = 12
        self.data = ''
        self.Data_path = ""
        self.root_path = ""
        self.inverse = False

    def test(self, algorithm: ModelAbstract,metrics, cb_progress=lambda x: None) :


        assert isinstance(metrics, (list, set, tuple)), "参数metrics必须是数组，且所有元素都必须是util.metric.MetricBase 的子类"
        assert np.all(isinstance(m, MetricAbstract) for m in metrics), "参数metrics的所有元素都必须是util.metric.MetricBase 的子类"
        # 数据加载器构建
        # 分别代表全部训练样本，正常训练样本，极端训练样本，全部验证样本，正常验证样本，极端验证样本，测试样本，标准化器
        if algorithm.pattern == 'train_alone':
            train_loader, val_loader, test_loader, scaler = define_dataloder_standard(
                algorithm)
            algorithm.scaler = scaler
            algorithm.train(train_loader, val_loader, test_loader, metrics[0],
                            lambda x: cb_progress(x * 0.8))  # 进度前80%用于训练
            pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        else:
            train_loader, train_loader_de, train_loader_ex, val_loader, val_loader_de, val_loader_ex, test_loader,scaler  = define_dataloder(algorithm)
            algorithm.scaler = scaler

        if algorithm.pattern == 'train_whole': # 完整流程,去极训练+微调+预测
           algorithm.train(train_loader_de, val_loader, test_loader,
                            metrics[0],
                           lambda x: cb_progress(x * 0.8))  # 进度前80%用于训练
           algorithm.finetune(train_loader_ex)
           pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'train': # 去极训练+预测
            algorithm.train(train_loader_de, val_loader, test_loader,
                             metrics[0],
                            lambda x: cb_progress(x * 0.8))  # 进度前80%用于训练
            pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'oversampling': # 过采样 训练+预测
            from torch.utils.data import ConcatDataset, DataLoader

            # 假设 train_loader 和 train_loader_ex 是 DataLoader 实例
            train_dataset = train_loader_de.dataset
            extreme_dataset = train_loader_ex.dataset

            # 设置过采样倍数，比如复制2次
            oversample_times = 2
            oversampled_extreme_dataset = ConcatDataset([extreme_dataset] * oversample_times)

            # 合并过采样后的极端样本和原始样本
            combined_dataset = ConcatDataset([train_dataset, oversampled_extreme_dataset])

            # 构造新的 DataLoader
            combined_train_loader = DataLoader(
                combined_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                drop_last=train_loader.drop_last
            )

            algorithm.train(combined_train_loader, val_loader, test_loader,
                             metrics[0],
                            lambda x: cb_progress(x * 0.8))  # 进度前80%用于训练
            pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'extreme_train': # 极值样本训练+预测
            algorithm.train(train_loader_ex,  val_loader_ex, test_loader,
                             metrics[0],
                            lambda x: cb_progress(x * 0.8))  # 进度前80%用于训练
            pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'finetune':  # 微调+预测
            algorithm.finetune(train_loader_ex)
            pred, ground_truth = algorithm.predict(test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'merge_test': # 直接加载双模型预测
            pred, ground_truth = algorithm.predict( test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        elif algorithm.pattern == 'test':
            pred, ground_truth = algorithm.predict( test_loader, lambda x: cb_progress(x * 0.2 + 80))  # 进度后20%用于测试
        # 计算评价指标
        results = [
            m(ground_truth, pred, algorithm.extreme_max, algorithm.extreme_min,algorithm.scaler.mean,algorithm.scaler.std)
            if any(name in m.__class__.__name__ for name in ['MAE_W','RMSE_W','MAPE_N', 'MAPE_E','MAE_N', 'MAE_E', 'RMSE_N', 'RMSE_E'])
            else m(ground_truth, pred)
            for m in metrics
        ]

        headers = [str(m) for m in metrics]

        return dict(zip(headers, results))


