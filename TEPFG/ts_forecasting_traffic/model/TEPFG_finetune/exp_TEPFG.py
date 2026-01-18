from ts_forecasting_traffic.model.trainer import TrainerBase
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from atp.model.torchmodel import TorchModel
import numpy as np
import torch
import os
from ts_forecasting_traffic.model.TEPFG_finetune.model.TEPFG import Model
from tqdm import tqdm
import copy

class Exp_TEPFG_finetune(TorchModel):
    def __init__(self):
        super(Exp_TEPFG_finetune,self).__init__()
        ## pretarin
        # parser
        self.device = 'cuda'
        self.cuda = True
        self.resume_state = False
        # data
        self.in_steps = 12
        self.out_steps = 12
        self.train_ratio = 0.6
        self.val_ratio = 0.2
        self.test_ratio = 0.2
        self.time_of_day = True
        self.day_of_week = True


        # model
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
        self.use_mixed_proj = True

        # train
        self.lr = 0.001
        self.weight_decay = 0.0005
        self.milestones = [15, 30, 50]
        self.lr_decay_rate = 0.1
        self.batch_size = 8
        self.epochs = 2
        self.use_cl = False
        self.cl_step_size = 2500


        # other
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
        self.pred = 12

        self.extreme_ratio = 0.1
        self.finetune_batch_size = 8
        self.extreme_sample_num = 368
        self.finetune_epochs = 10
        self.finetune_sample_num = 1000
        self.loss_type_extreme = 'weighted_huber'
        self.detect_sample_num = 0 #
        self.dataset = 'PEMS04'
        self.root_path = './ts_forecasting_traffic/data'
        self.checkpoint_path_pntrain_model = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04/checkpoint.pth'
        self.checkpoint_path_standard_model = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04_standard/checkpoint.pth'
        self.checkpoint_path_standard_extreme_model = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04_standard/checkpoint_0_100_1.6'
        self.finetune_path = 'ts_forecasting_traffic/checkpoints/TEPFG/PEMS04/finetune_type/checkpoint.pth'
        self.merge = False
        self.label = False
        self.use_GMM = False
        self.use_possibility= False
        self.normal_weight = 0
        self.extreme_weight = 1
        self.label_path = "result/PEMS04/label/predictions_rec.npy"

    def _build_model(self):
        model = Model(
            self
        ).float()
        return model

    def opt_one_batch(self, inputs, targets=None):
        inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
        targets = targets[..., :self.output_dim]
        out = self.model(inputs)
        self.optimizer.zero_grad()
        out = self.scaler.inverse_transform(out)
        targets = self.scaler.inverse_transform(targets)
        loss_pred = self.loss(out, targets)
        loss = loss_pred
        loss.backward()
        self.optimizer.step()
        Loss_dict = {}
        Loss_dict['loss'] = float( loss.data.cpu().numpy())
        return Loss_dict

    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None, valid_func=None, cb_progress=lambda x: None):


        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay,
            eps= 1e-8,
        )
        if self.pattern == 'extreme_train':
            self.loss = WeightedExtremeMSELoss(extreme_threshold=1.6, mean=self.scaler.mean, std=self.scaler.std,
                               normal_weight=self.normal_weight, extreme_weight=self.extreme_weight)
        else:
            self.loss = nn.HuberLoss()
        writer = SummaryWriter(self.tensorboard_path)

        #self.standard_scaler = train_ds.scaler

        trainer = TrainerBase(self.epochs, valid_on_train_set=True)
        trainer.train(self, train_dataloader, val_dataloader, test_dataloader, valid_func, writer)
        os.makedirs(os.path.dirname(self.checkpoint_path_pntrain_model), exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path_pntrain_model)
    def _predict(self, inputs, targets):
        is_extreme = is_extreme_data(targets, lower_threshold=-self.extreme_min, upper_threshold=self.extreme_max,
                                     extreme_ratio=self.extreme_ratio)
        if self.pattern in ['train_whole','finetune','merge_test']:
            if self.merge:
                # label = inputs[..., 3]
                label = targets[..., 3]
                label = label.unsqueeze(-1)
                output_no = self.model(inputs)
                output_ex = self.model_extreme(inputs)
                if self.use_possibility:
                    output = label * output_ex + (1 - label) * output_no
                else:
                    output = torch.where(label == 1, output_ex, output_no)

                targets = targets[..., :self.output_dim]
                output = self.scaler.inverse_transform(output)
                y_lbl = self.scaler.inverse_transform(targets)
                return output.detach().cpu().numpy(), y_lbl.detach().cpu().numpy(), is_extreme

                # output_no = self.model(inputs)
                # output_ex = self.model_extreme(inputs)
                # # extreme_mask = (output_no > self.extreme_max) | (output_no < -self.extreme_min)
                # extreme_mask = (output_ex > self.extreme_max) | (output_ex < -self.extreme_min)
                # output =  torch.where(extreme_mask, output_ex, output_no)
                # output = self.scaler.inverse_transform(output)
                # targets=targets[...,: self.output_dim]
                # y_lbl = self.scaler.inverse_transform(targets)
                # return output.detach().cpu().numpy(), y_lbl.detach().cpu().numpy(), is_extreme
            else:
                # if is_extreme == 1:
                #     output = self.model_extreme(inputs)
                # elif is_extreme == 0:
                #     output = self.model(inputs)
                # output = self.model_extreme(inputs)

                output = self.model(inputs)
                output = self.scaler.inverse_transform(output)
                targets = targets[..., :self.output_dim]
                y_lbl = self.scaler.inverse_transform(targets)
                return output.detach().cpu().numpy(),y_lbl.detach().cpu().numpy(), is_extreme

        elif self.pattern in ['train', 'train_alone','extreme_train','oversampling']:
            output = self.model(inputs)
            targets = targets[..., :self.output_dim]

            output = self.scaler.inverse_transform(output)
            y_lbl = self.scaler.inverse_transform(targets)


            return output.detach().cpu().numpy(),y_lbl.detach().cpu().numpy(),is_extreme

    def eval_data(self, dataloader, metric, inbatch=None) -> float:


        self.model.eval()

        Y = []
        Pred = []
        with torch.no_grad():
            for inputs, targets in  dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets[..., :self.output_dim].squeeze(0).to(self.device)
                out = self.model(inputs)
                out = self.scaler.inverse_transform(out)
                targets = self.scaler.inverse_transform(targets)
                loss_pred = self.loss(out, targets)
                Y.append(targets.detach().cpu().numpy())
                Pred.append(out.detach().cpu().numpy())
        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        self.model.train()
        return metric(Y, Pred)

    def predict(self, test_dataloader, cb_progress=lambda x: None):
        Y = []
        Pred = []
        e_num = 0
        n_num = 0

        if self.pattern == 'merge_test':
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))
            self.model = self.model.to(self.device)

            standard_extreme_model_path = self.checkpoint_path_standard_extreme_model
            self.model_extreme = self._build_model()
            self.model_extreme.load_state_dict(
                torch.load(standard_extreme_model_path, map_location=self.device))
            self.model_extreme = self.model_extreme.to(self.device)


        if self.pattern == 'finetune':
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))
            self.model = self.model.to(self.device)
        if self.pattern in ['train_whole', 'finetune','merge_test']:
            self.model_extreme.eval()
        self.model.eval()

        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
                pred,targets,flag = self._predict(inputs, targets)
                Y.append(targets)
                Pred.append(pred)
                if flag == 1:
                    e_num += 1
                elif flag == 0:
                    n_num += 1
            print(e_num,n_num)
        pred_array = np.squeeze(np.concatenate(Pred))   # shape: (T, N)
        true_array = np.squeeze(np.concatenate(Y))      # shape: (T, N)
        return pred_array, true_array


    def finetune(self,retrain_loader ):

        if self.pattern == 'finetune':
            standard_model_path = self.checkpoint_path_standard_model
            self.model_extreme = self._build_model()
            self.model_extreme.load_state_dict(torch.load(standard_model_path, map_location=self.device)) # 加载预训练的模型权重到当前模型中
            self.model_extreme = self.model_extreme.to(self.device)
        else:
            self.model_extreme = copy.deepcopy(self.model)

        if hasattr(self.model_extreme, "input_proj"):
            for param in self.model_extreme.input_proj.parameters():
                param.requires_grad = False
        if hasattr(self.model_extreme, "tod_embedding"):
            for param in self.model_extreme.tod_embedding.parameters():
                param.requires_grad = False
        if hasattr(self.model_extreme, "dow_embedding"):
            for param in self.model_extreme.dow_embedding.parameters():
                param.requires_grad = False
        if hasattr(self.model_extreme, "node_emb"):
            self.model_extreme.node_emb.requires_grad = False
        if hasattr(self.model_extreme, "adaptive_embedding"):
            self.model_extreme.adaptive_embedding.requires_grad = False

        self.optimizer_extreme = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model_extreme.parameters()),
            lr=1e-4,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        if self.loss_type_extreme == 'mse':
            self.loss_extreme = nn.MSELoss()
        elif self.loss_type_extreme == 'mae':
            self.loss_extreme = nn.L1Loss()
        elif self.loss_type_extreme == 'huber':
            self.loss_extreme = nn.HuberLoss()
        elif self.loss_type_extreme == 'weighted_huber':
            self.loss_extreme = WeightedHuberLoss(delta=1.0, weight_extreme=10.0)
        elif self.loss_type_extreme == 'extreme':
            self.loss_extreme = WeightedExtremeMSELoss(extreme_threshold=1.6, mean=self.scaler.mean, std=self.scaler.std,normal_weight=self.normal_weight,extreme_weight=self.extreme_weight)
        else:
            raise ValueError(f"Unknown loss_type_extreme: {self.loss_type_extreme}")
        self.model_extreme.train()
        for epoch in range(self.finetune_epochs):
            iter_data = (
                tqdm(
                    retrain_loader,
                    total=len(retrain_loader),
                    ncols=100,
                    desc=f"finetune {epoch}:" ))
            total_loss = 0.0
            num_batches = 0
            for i, (batch_x, batch_y) in enumerate(iter_data):

                self.optimizer_extreme.zero_grad()
                pred, true = self._process_one_batch_model(batch_x, batch_y)

                pred = self.scaler.inverse_transform(pred)
                true = self.scaler.inverse_transform(true)

                curr_loss = self.loss_extreme(pred.to(self.device), true)   # 这里要修改
                total_loss += curr_loss.item()
                num_batches += 1
                curr_loss.backward()
                self.optimizer_extreme.step()
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")


    def _process_one_batch_model(self, batch_x, batch_y):
        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        outputs = self.model_extreme(x)

        true = batch_y[:, :, :, 0:1].to(self.device)

        return outputs, true

def is_extreme_data(data, lower_threshold=-1.6, upper_threshold=1.6, extreme_ratio=0.2):
    traffic_flow = data[..., 0]
    extreme_mask = (traffic_flow < lower_threshold) | (traffic_flow > upper_threshold)
    extreme_count = torch.sum(extreme_mask)
    total_points = traffic_flow.numel()
    return 1 if extreme_count >= extreme_ratio * total_points else 0
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, weight_extreme=5.0):
        super().__init__()
        self.delta = delta
        self.weight_extreme = weight_extreme

    def forward(self, pred, target):
        error = pred - target
        abs_error = torch.abs(error)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta).to(error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        weight_mask = ((target > 1.6) | (target < -1.6)).float() * self.weight_extreme + 1
        loss = loss * weight_mask
        return loss.mean()
class WeightedExtremeMSELoss(nn.Module):
    def __init__(self, extreme_threshold=1.6, mean=0, std=0, normal_weight=1.0, extreme_weight=5.0):
        super().__init__()
        self.mean = mean
        self.std = std
        self.extreme_threshold = extreme_threshold
        self.normal_weight = normal_weight
        self.extreme_weight = extreme_weight

    def forward(self, pred, target):
        error = pred - target
        loss = error ** 2
        lower_bound = self.mean - self.extreme_threshold * self.std
        upper_bound = self.mean + self.extreme_threshold * self.std
        is_extreme = ((target > upper_bound) | (target < lower_bound)).float()
        weights = self.normal_weight * (1.0 - is_extreme) + self.extreme_weight * is_extreme
        weighted_loss = loss * weights
        return weighted_loss.mean()



