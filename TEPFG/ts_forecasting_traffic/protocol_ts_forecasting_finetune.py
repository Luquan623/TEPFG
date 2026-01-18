import numpy as np
import pandas as pd
from ts_forecasting_traffic.dataset.ts_forecasting_traffic_TEPFG import define_dataloder, define_dataloder_standard
from atp.metric.metric import MetricAbstract
from atp.model.model import ModelAbstract
import torch
import os



class StandardScaler:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


def normalize_dataset(data, input_base_dim):
    data_ori = data[:, :, 0:input_base_dim]
    data_day = data[:, :, input_base_dim:input_base_dim + 1]
    data_week = data[:, :, input_base_dim + 1:input_base_dim + 2]

    mean_data = data_ori.mean()
    std_data = data_ori.std()
    mean_day = data_day.mean()
    std_day = data_day.std()
    mean_week = data_week.mean()
    std_week = data_week.std()

    scaler_data = StandardScaler(mean_data, std_data)
    scaler_day = StandardScaler(mean_day, std_day)
    scaler_week = StandardScaler(mean_week, std_week)

    print('Normalize the dataset by Standard Normalization')

    return scaler_data, scaler_day, scaler_week


class TSForecastingTVT:

    def __init__(self):
        self.progress = np.zeros(1, dtype=float)
        self.progress_train_proportion = 0.9
        self.add_time_stamp = False
        self.informer = False
        self.label_len = 0
        self.input_size = 12
        self.output_size = 12
        self.data = ''
        self.Data_path = ""
        self.root_path = ""
        self.inverse = False

    def test(self, algorithm: ModelAbstract, metrics, cb_progress=lambda x: None):

        assert isinstance(metrics, (list, set, tuple))
        assert np.all(isinstance(m, MetricAbstract) for m in metrics)

        if algorithm.pattern == 'train_alone':
            train_loader, val_loader, test_loader, scaler = define_dataloder_standard(
                algorithm)
            algorithm.scaler = scaler
            algorithm.train(
                train_loader,
                val_loader,
                test_loader,
                metrics[0],
                lambda x: cb_progress(x * 0.8)
            )
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        else:
            train_loader, train_loader_de, train_loader_ex, val_loader, val_loader_de, val_loader_ex, test_loader, scaler = define_dataloder(
                algorithm)
            algorithm.scaler = scaler

        if algorithm.pattern == 'train_whole':
            algorithm.train(
                train_loader_de,
                val_loader,
                test_loader,
                metrics[0],
                lambda x: cb_progress(x * 0.8)
            )
            algorithm.finetune(train_loader_ex)
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'train':
            algorithm.train(
                train_loader_de,
                val_loader,
                test_loader,
                metrics[0],
                lambda x: cb_progress(x * 0.8)
            )
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'oversampling':
            from torch.utils.data import ConcatDataset, DataLoader

            train_dataset = train_loader_de.dataset
            extreme_dataset = train_loader_ex.dataset

            oversample_times = 2
            oversampled_extreme_dataset = ConcatDataset(
                [extreme_dataset] * oversample_times)

            combined_dataset = ConcatDataset(
                [train_dataset, oversampled_extreme_dataset])

            combined_train_loader = DataLoader(
                combined_dataset,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                drop_last=train_loader.drop_last
            )

            algorithm.train(
                combined_train_loader,
                val_loader,
                test_loader,
                metrics[0],
                lambda x: cb_progress(x * 0.8)
            )
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'extreme_train':
            algorithm.train(
                train_loader_ex,
                val_loader_ex,
                test_loader,
                metrics[0],
                lambda x: cb_progress(x * 0.8)
            )
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'finetune':
            algorithm.finetune(train_loader_ex)
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'merge_test':
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )
        elif algorithm.pattern == 'test':
            pred, ground_truth = algorithm.predict(
                test_loader,
                lambda x: cb_progress(x * 0.2 + 80)
            )

        results = [
            m(
                ground_truth,
                pred,
                algorithm.extreme_max,
                algorithm.extreme_min,
                algorithm.scaler.mean,
                algorithm.scaler.std
            )
            if any(
                name in m.__class__.__name__
                for name in [
                    'MAE_W',
                    'RMSE_W',
                    'MAPE_N',
                    'MAPE_E',
                    'MAE_N',
                    'MAE_E',
                    'RMSE_N',
                    'RMSE_E'
                ]
            )
            else m(ground_truth, pred)
            for m in metrics
        ]

        headers = [str(m) for m in metrics]

        return dict(zip(headers, results))
