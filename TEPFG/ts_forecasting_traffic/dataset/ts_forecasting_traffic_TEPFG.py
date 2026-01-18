import torch
import random
import numpy as np
from ts_forecasting_traffic.util import StandardScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import pandas as pd
import torch.utils.data as Data
from sklearn.mixture import GaussianMixture


def add_extreme_labels(data, feature_index, extreme_max, extreme_min):
    flow = data[:, :, feature_index]
    labels = np.zeros_like(flow, dtype=np.int8)
    labels[(flow > extreme_max) | (flow < -extreme_min)] = 1
    labels = np.expand_dims(labels, axis=-1)
    new_data = np.concatenate((data, labels), axis=-1)
    return new_data


def time_add(data, week_start, interval=5):
    time_slot = 24 * 60 // interval
    day_data = np.zeros_like(data, dtype=float)
    week_data = np.zeros_like(data, dtype=int)
    for index in range(data.shape[0]):
        day_data[index, :] = (index % time_slot) / time_slot
        week_init = (week_start - 1 + (index // time_slot)) % 7
        week_data[index, :] = week_init
    return day_data, week_data


def load_st_dataset(dataset, algorithm):
    if dataset == 'PEMS04':
        data_path = os.path.join(f'ts_forecasting_traffic/data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 1
        interval = 5
        algorithm.interval = interval
        day_data, week_data = time_add(data, week_start, interval)

    elif dataset == 'PEMS08':
        data_path = os.path.join(f'ts_forecasting_traffic/data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 5
        interval = 5
        algorithm.interval = interval
        day_data, week_data = time_add(data, week_start, interval)

    data = np.expand_dims(data, axis=-1)
    day_data = np.expand_dims(day_data, axis=-1).astype(np.float32)
    week_data = np.expand_dims(week_data, axis=-1).astype(np.float32)

    data = np.concatenate([data, day_data, week_data], axis=-1)

    return data


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    if test_ratio == 0:
        test_data = data[:0]
        val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
        if val_ratio == 0:
            train_data = data
        else:
            train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    else:
        test_data = data[-int(data_len * test_ratio):]
        val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
        train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


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
def normalize_dataset(data, input_base_dim): # input_base_dim：输入数据的基本维度
    data_ori = data[:, :, 0:input_base_dim]
    mean_data = data_ori.mean()
    std_data = data_ori.std()
    scaler_data = StandardScaler(mean_data, std_data)
    return scaler_data
class TrafficDataset(Dataset):
    def __init__(self, data, batch_size, input_window=288, output_window=288, eval_only=False):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

        self.windows = [
            (data[i:i + input_window], data[i + input_window:i + input_window + output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]

        if eval_only == False:
            random.shuffle(self.windows)
            if len(self.windows) % batch_size != 0:
                self.windows = self.windows[:-(len(self.windows) % batch_size)]

        self.batches = [
            self.windows[i:i + batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_x, batch_y = zip(*self.batches[idx])
        return torch.from_numpy(np.stack(batch_x)).float(), torch.from_numpy(np.stack(batch_y)).float()

def define_dataloder(algorithm):
    dataset_name = algorithm.dataset_use[0]
    data = load_st_dataset(dataset_name, algorithm)
    data_train, data_val, data_test = split_data_by_ratio(data, algorithm.val_ratio, algorithm.test_ratio)
    num_train = len(data_train)
    num_test = len(data_val)
    num_vali = len(data_test)
    border1s = [0, num_train - algorithm.his, len(data) - num_test - algorithm.his]
    border2s = [num_train, num_train + num_vali, len(data)]
    scaler_data = normalize_dataset(data_train, algorithm.input_base_dim)
    print(data_train.shape, scaler_data.mean, scaler_data.std)
    data[..., :algorithm.input_base_dim] = scaler_data.transform(data[:, :, :algorithm.input_base_dim])

    if getattr(algorithm, "use_GMM", False):
        gmm_indicator = generate_gmm_indicator(
            data=data,
            feature_index=0,
            n_components=3,
            fit_source=data_train
        )
        data = np.concatenate([data, gmm_indicator], axis=-1)
        print("Data shape after adding GMM indicator:", data.shape)

    if getattr(algorithm, "extreme_labeling", False):
        data = add_extreme_labels(
            data,
            feature_index=0,
            extreme_max=algorithm.extreme_max,
            extreme_min=algorithm.extreme_min
        )
    if getattr(algorithm, "label", False):
        new_labels = np.load(algorithm.label_path)

        if new_labels.ndim == 3:
            new_labels = np.squeeze(new_labels, axis=1)

        T = data.shape[0]
        N = data.shape[1]
        test_len = int(T * algorithm.test_ratio)
        test_start = T - test_len
        test_end = T
        old_labels = data[test_start:test_end, :, 3]

        assert new_labels.shape == old_labels.shape, \
            f"Label dimensions do not match! New labels: {new_labels.shape}, Original: {old_labels.shape}, Range: [{test_start}, {test_end})"

        no = np.sum(old_labels == 0)
        ex = np.sum(old_labels == 1)
        total_diff = np.sum(old_labels != new_labels)
        count_0_to_1 = np.sum((old_labels == 0) & (new_labels == 1))
        count_1_to_0 = np.sum((old_labels == 1) & (new_labels == 0))

        print(f"Actual normal points: {no}")
        print(f"Actual extreme points: {ex}")
        print(f"Total different points: {total_diff}")
        print(f"From 0 to 1 (new extreme values): {count_0_to_1}")
        print(f"From 1 to 0 (removed extreme values): {count_1_to_0}")

        data[test_start:test_end, :, 3] = new_labels

        print("Shape of labels after replacement:", data[test_start:test_end, :, 3].shape)
        print("Is it consistent:", np.all(data[test_start:test_end, :, 3] == new_labels))

    algorithm.data = data
    train_loader, train_loader_de, train_loader_ex = get_datasets_all(algorithm, border1s[0], border2s[0], type='train')
    val_loader, val_loader_de, val_loader_ex = get_datasets_all(algorithm, border1s[1], border2s[1], type='val')
    test_loader = get_datasets_all(algorithm, border1s[2], border2s[2] - algorithm.pred - algorithm.his + 1, type='test')
    return train_loader, train_loader_de, train_loader_ex, val_loader, val_loader_de, val_loader_ex, test_loader, scaler_data


def define_dataloder_standard(algorithm):
    dataset_name = algorithm.dataset_use[0]
    data = load_st_dataset(dataset_name, algorithm)
    data_train, data_val, data_test = split_data_by_ratio(data, algorithm.val_ratio,
                                                              algorithm.test_ratio)

    num_train = len(data_train)
    num_test = len(data_val)
    num_vali = len(data_test)

    border1s = [0, num_train - algorithm.his, len(data) - num_test - algorithm.his]
    border2s = [num_train, num_train + num_vali, len(data)]
    scaler_data = normalize_dataset(data_train,algorithm.input_base_dim)
    print(data_train.shape, scaler_data.mean, scaler_data.std)

    data[..., :algorithm.input_base_dim] =  scaler_data.transform(data[:, :, :algorithm.input_base_dim])
    if getattr(algorithm, "extreme_labeling", False):
        data = add_extreme_labels(
            data,
            feature_index=0,
            extreme_max=algorithm.extreme_max,
            extreme_min=algorithm.extreme_min
        )
    algorithm.data = data
    train_loader = get_datasets_all(algorithm, border1s[0], border2s[0],type='train')
    val_loader = get_datasets_all(algorithm, border1s[1], border2s[1],type='val')
    test_loader= get_datasets_all(algorithm, border1s[2], border2s[
        2] - algorithm.pred - algorithm.his + 1,type='test')
    return train_loader,val_loader,test_loader,scaler_data

def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None
def get_datasets_all(algorithm, start, end, type='train'):

    if getattr(algorithm, "pattern", None) == "train_alone":
        return get_datasets_all_train_alone(algorithm, start, end, type)

    (seq_x_original,seq_y_original),(seq_x, seq_y), (seq_x_ex, seq_y_ex) = get_datasets_ex(algorithm, start, end, type)
    if type == 'train':
        data_loader_original = set_dataloader_value(seq_x_original, seq_y_original, algorithm.batch_size, True, 0,False)
        data_loader = set_dataloader_value(seq_x, seq_y, algorithm.batch_size, True, 0, False)
        loader_ex = select_data_ex(seq_x_ex, seq_y_ex, batch_size=algorithm.finetune_batch_size,shuffle_flag=True,
                                    type=type)
        return data_loader_original, data_loader, loader_ex
    if type == 'val':
        data_loader_original = set_dataloader_value(seq_x_original, seq_y_original, algorithm.batch_size, False, 0,False)
        data_loader = set_dataloader_value(seq_x, seq_y, algorithm.batch_size, False, 0, False)
        loader_ex = select_data_ex(seq_x_ex, seq_y_ex, batch_size=algorithm.finetune_batch_size,shuffle_flag=False,
                                    type=type)
        return data_loader_original,data_loader,loader_ex
    if type == 'test':
        data_loader_original = set_dataloader_value(seq_x_original, seq_y_original, algorithm.batch_size, False, 0,
                                                    False)
        return data_loader_original
def get_datasets_all_train_alone(algorithm, start, end, type='train'):
    seq_x_all, seq_y_all = [], []

    df_all = algorithm.data
    for curr in range(start, end):
        s_end = curr + algorithm.his
        r_end = s_end + algorithm.pred
        s_x_all = df_all[curr:s_end,   ...]
        s_y_all = df_all[s_end:r_end,  ...]
        seq_x_all.append(s_x_all)
        seq_y_all.append(s_y_all)

    shuffle_flag = (type == 'train')
    loader = set_dataloader_value(seq_x_all, seq_y_all, algorithm.batch_size, shuffle_flag, 0, False)

    return loader
def get_datasets_ex(algorithm, start, end, type):
    seq_x_original, seq_y_original = [], []
    seq_x, seq_y = [], []
    seq_x_ex, seq_y_ex = [], []
    count_ex, count_ge, tot_count = 0, 0, 0

    if type in ['train', 'val']:
        ex_all_index_lst = []
        ex_sep_index_lst = []
        ex_sum = []

    for curr in range(start, end):
        s_end = curr + algorithm.his
        r_end = s_end + algorithm.pred
        df_data = algorithm.data[..., 0]
        s_x = df_data[curr:s_end]
        s_y = df_data[s_end:r_end]
        s_x_all = algorithm.data[curr:s_end,... ]
        s_y_all = algorithm.data[s_end:r_end, ...]
        upper_bound = algorithm.extreme_max
        lower_bound = -algorithm.extreme_min
        extreme_values_sy = np.sum((s_y < lower_bound) | (s_y > upper_bound))
        extreme_values_sum = extreme_values_sy

        if extreme_values_sum > algorithm.extreme_sample_num :
            seq_x_ex.append(s_x_all)
            seq_y_ex.append(s_y_all)
            if type in ['train', 'val']:
                ex_all_index_lst.append(tot_count)
                ex_sep_index_lst.append(count_ex)
                ex_sum.append(extreme_values_sum)
                count_ex += 1
        else:
            count_ge += 1

        tot_count += 1
        seq_x.append(s_x_all)
        seq_y.append(s_y_all)

    seq_x_original = seq_x
    seq_y_original = seq_y
    if type in ['train', 'val']:
        if type == 'train':
            sample = algorithm.finetune_sample_num
        if type == 'val':
            sample = algorithm.detect_sample_num
        if sample != 0:
            select_num = min(len(ex_sep_index_lst), sample)
            indexed_extreme_values = list(zip(ex_sum, ex_sep_index_lst))
            sorted_samples = sorted(indexed_extreme_values, key=lambda x: x[0], reverse=True)
            top_samples = sorted_samples[:select_num]
            selected_indices = [index for _, index in top_samples]
            ex_sep_index_array = np.array(ex_sep_index_lst)
            ex_all_index_array = np.array(ex_all_index_lst)
            ex_sep_location = ex_sep_index_array[selected_indices]
            ex_all_location = ex_all_index_array[selected_indices]
            ex_all_location_set = set(ex_all_location)
            seq_x = [x for i, x in enumerate(seq_x) if i not in ex_all_location_set]
            seq_y = [x for i, x in enumerate(seq_y) if i not in ex_all_location_set]
            seq_x_ex = np.array(seq_x_ex)
            seq_y_ex = np.array(seq_y_ex)
            seq_x_ex = seq_x_ex[ex_sep_location].tolist()
            seq_y_ex = seq_y_ex[ex_sep_location].tolist()
        else:
            seq_x_ex, seq_y_ex = seq_x, seq_y
    else:
        seq_x_ex, seq_y_ex = seq_x, seq_y

    return (seq_x_original, seq_y_original), (seq_x, seq_y), (seq_x_ex, seq_y_ex)

def set_dataloader_value( seq_x, seq_y, batch_size, shuffle_flag,num_workers,drop_last):
    loader = Data.DataLoader(
        Data.TensorDataset(
            torch.from_numpy(np.array(seq_x)).float(),
            torch.from_numpy(np.array(seq_y)).float(),
        ),
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=False
    )
    return loader
def select_data_ex( seq_x, seq_y, batch_size=32, shuffle_flag=True, type=None):

    seq_x, seq_y = np.array(seq_x), np.array(seq_y)
    seq_x_selected = seq_x
    seq_y_selected = seq_y

    if len(seq_x_selected) == 0:
        print('cannot use this dataset')
        print(seq_x_selected.shape)
    else:
        print(type, len(seq_x_selected))

    loader = set_dataloader_value(seq_x_selected, seq_y_selected,
                                          batch_size, shuffle_flag, 0,False)
    return loader

def generate_gmm_indicator(data, feature_index=0, n_components=3, fit_source=None):
    T, N, F = data.shape
    if fit_source is None:
        fit_source = data
    X_all = data[:, :, feature_index].reshape(-1, 1)
    X_fit = fit_source[:, :, feature_index].reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_fit)
    max_idx = gmm.means_.argmax()
    probs = gmm.predict_proba(X_all)[:, max_idx]
    gmm_indicator = probs.reshape(T, N, 1)
    return gmm_indicator
