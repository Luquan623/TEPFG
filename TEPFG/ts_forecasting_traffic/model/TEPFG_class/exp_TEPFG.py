from ts_forecasting_traffic.model.trainer import TrainerBase
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from atp.model.torchmodel import TorchModel
import numpy as np
import torch
import os
from ts_forecasting_traffic.model.TEPFG_class.model.TEPFG import Model
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt

class Exp_TEPFG_class(TorchModel):
    def __init__(self):
        super(Exp_TEPFG_class,self).__init__()
        self.device = 'cuda'
        self.cuda = True
        self.resume_state = False
        self.in_steps = 12
        self.out_steps = 12
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.time_of_day = True
        self.day_of_week = True

        self.num_nodes = 307
        self.steps_per_day = 288
        self.input_dim = 3
        self.output_dim = 1
        self.input_embedding_dim = 24
        self.tod_embedding_dim = 24
        self.dow_embedding_dim = 24
        self.spatial_embedding_dim = 0
        self.adaptive_embedding_dim = 80
        self.feed_forward_dim = 256
        self.num_heads = 4
        self.num_layers = 3
        self.dropout = 0.1

        self.lr = 0.001
        self.weight_decay = 0.0005
        self.milestones = [15, 30, 50]
        self.lr_decay_rate = 0.1
        self.batch_size = 8
        self.epochs = 2
        self.use_cl = False
        self.cl_step_size = 2500

        self.pattern = 'train_alone'
        self.extreme_max = 1.6
        self.extreme_min = 1.6
        self.scaler = None
        self.dataset_use = ['PEMS04']
        self.Model = 'TEPFG'
        self.weather = False
        self.input_base_dim = 1
        self.extreme_labeling = False
        self.his = 12
        self.pred = 1

        self.extreme_ratio = 0.1
        self.finetune_batch_size = 8
        self.extreme_sample_num = 368
        self.finetune_epochs = 10
        self.finetune_sample_num = 1000
        self.loss_type_extreme = 'weighted_huber'
        self.detect_sample_num = 0
        self.dataset = 'PEMS04'
        self.root_path = './ts_forecasting_traffic/data'
        self.checkpoint_path_pntrain_model = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04/checkpoint.pth'
        self.checkpoint_path_standard_model = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04_standard/checkpoint.pth'
        self.finetune_path = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04/finetune_type/checkpoint.pth'
        self.label = False
        self.label_save = "result/PEMS04/label/predictions00.npy"
        self.best_threshold = 0.5
        self.val_loss_curve = []
        self.use_GMM = False
        self.save_possibility = False

    def _build_model(self):
        model = Model(
            self
        ).float()
        return model

    def opt_one_batch(self, inputs, targets=None):
        inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
        targets = targets[..., -self.output_dim:].float()

        out = self.model(inputs)
        self.optimizer.zero_grad()

        loss_pred = self.loss(out, targets)

        loss = loss_pred
        loss.backward()
        self.optimizer.step()

        Loss_dict = {}
        Loss_dict['loss'] = float(loss.data.cpu().numpy())
        return Loss_dict

    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None, valid_func=None, cb_progress=lambda x: None):
        self.model = self._build_model()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        ratio = 10
        pos_weight = torch.tensor([ratio], device=self.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        writer = SummaryWriter(self.tensorboard_path)
        trainer = TrainerBase(self.epochs, valid_on_train_set=True)

        trainer.train(self, train_dataloader, val_dataloader, test_dataloader, valid_func, writer)
        os.makedirs(os.path.dirname(self.checkpoint_path_pntrain_model), exist_ok=True)
        self.plot_val_metric_curve(ylabel=valid_func.__class__.__name__)
        torch.save(self.model.state_dict(), self.checkpoint_path_pntrain_model)

    def _predict(self, inputs, targets):
        output= self.model(inputs)
        probs = torch.sigmoid(output)
        if self.save_possibility:
            output = probs
        else:
            output = (probs > self.best_threshold).int()
        y_lbl = targets[..., -self.output_dim:].float()
        return output.detach().cpu().numpy(), y_lbl.detach().cpu().numpy()

    def eval_data(self, dataloader, metric, inbatch=None) -> float:
        self.model.eval()
        Y = []
        Pred = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets[..., -self.output_dim:].squeeze(0).float().to(self.device)
                out = self.model(inputs)
                probs = torch.sigmoid(out)
                out = (probs > self.best_threshold).int()
                Y.append(targets.detach().cpu().numpy())
                Pred.append(out.detach().cpu().numpy())
        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        self.model.train()
        score = metric(Y, Pred)
        self.val_loss_curve.append(score)
        return score

    def predict(self, test_dataloader, cb_progress=lambda x: None):
        if self.pattern == 'test':
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))
            self.model = self.model.to(self.device)
        Y = []
        Pred = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
                pred, targets = self._predict(inputs, targets)
                Y.append(targets)
                Pred.append(pred)
        save_dir = os.path.dirname(self.label_save)
        os.makedirs(save_dir, exist_ok=True)
        pred_array = np.concatenate(Pred, axis=0)
        if pred_array.shape[-1] == 1:
            pred_array = np.squeeze(pred_array, axis=-1)
        np.save(self.label_save, pred_array)
        csv_ready_array = pred_array.reshape(pred_array.shape[0], -1)
        csv_path = os.path.join(save_dir, "predictions00.csv")
        np.savetxt(csv_path, csv_ready_array, delimiter=",", fmt="%.4f")
        if not self.save_possibility:
            total_points = pred_array.size
            extreme_points = np.sum(pred_array == 1)
            extreme_ratio = extreme_points / total_points
            print(f"Total predicted points: {total_points}")
            print(f"Predicted extreme points (1): {extreme_points}")
            print(f"Extreme ratio: {extreme_ratio:.4%}")
        return np.squeeze(np.concatenate(Pred)), np.squeeze(np.concatenate(Y))
    def plot_val_metric_curve(self, ylabel="Validation Metric"):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.val_loss_curve) + 1), self.val_loss_curve, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.show()