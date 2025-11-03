from ts_forecasting_traffic.model.trainer import TrainerBase
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from atp.model.torchmodel import TorchModel
import numpy as np
import torch
import os
from ts_forecasting_traffic.model.STAEformer_finetune.model.STAEformer import Model
from tqdm import tqdm
import copy




class Exp_STAEformer_finetune(TorchModel):
    def __init__(self):
        super(Exp_STAEformer_finetune,self).__init__()
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
        self.Model = 'STAEformer'
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
        self.loss_type_extreme = 'weighted_huber'  # 可选: 'mse', 'mae', 'huber', 'weighted_huber'
        self.detect_sample_num = 0 #
        self.dataset = 'PEMS04'
        self.root_path = './ts_forecasting_traffic/data'
        self.checkpoint_path_pntrain_model = 'ts_forecasting_traffic/checkpoints/STAEformer/PEMS04/checkpoint.pth'
        self.checkpoint_path_standard_model = 'ts_forecasting_traffic/checkpoints/STAEformer/PEMS04_standard/checkpoint.pth'
        self.checkpoint_path_standard_extreme_model = 'ts_forecasting_traffic/checkpoints/STAEformer/PEMS04_standard/checkpoint_0_100_1.6'
        self.finetune_path = 'ts_forecasting_traffic/checkpoints/STAEformer/PEMS04/finetune_type/checkpoint.pth'
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

    def opt_one_batch(self, inputs, targets=None):   # 训练一个批次的数据，返回该批次的损失
        """
        Parameters
        ----------
        batch: 输入的优化数据

        Returns
        -------
        返回一个至少包含'loss' 关键字的字典。 loss的值表示当前bat数据下算出来的损失值。
        """
        inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device) # squeeze(0) 用于去掉第一维（通常是batch size为1时），确保输入和目标都是二维的
        targets = targets[..., :self.output_dim]
        out = self.model(inputs)
        self.optimizer.zero_grad()

        # 逆归一化
        out = self.scaler.inverse_transform(out)
        targets = self.scaler.inverse_transform(targets)
        loss_pred = self.loss(out, targets)

        loss = loss_pred
        loss.backward()
        self.optimizer.step()

        Loss_dict = {}
        Loss_dict['loss'] = float( loss.data.cpu().numpy())  # 不管数据在gpu还是cpu都统一存入cpu
        return Loss_dict

    def train(self, train_dataloader, val_dataloader=None, test_dataloader=None, valid_func=None, cb_progress=lambda x: None):


        self.model = self._build_model()  # 实例化模型对象
        # 把模型放到gpu或cpu上
        self.model.to(self.device)

        # 设置优化方法及相关参数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay= self.weight_decay,
            eps= 1e-8,
        ) # 定义了优化器
        if self.pattern == 'extreme_train':
            self.loss = WeightedExtremeMSELoss(extreme_threshold=1.6, mean=self.scaler.mean, std=self.scaler.std,
                               normal_weight=self.normal_weight, extreme_weight=self.extreme_weight)
        else:
            self.loss = nn.HuberLoss() # 原论文
        #self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=self.milestones,
        #     gamma=self.lr_decay_rate,
        #     verbose=False,
        # ) # 设置 学习率调度器
        # # tensorboard设置
        writer = SummaryWriter(self.tensorboard_path)

        #self.standard_scaler = train_ds.scaler

        # 构建训练器，trainer，自动根据训练数据、验证数据和验证函数进行验证，并将中间过程记录到writer中
        # 需要注意的是：当前模型必须实现save，load，opt_one_batch， eval_data 函数
        # trainer = TrainerBase(self.nepochs)
        # 如果需要显示训练集上的验证结果，则用如下函数构建trainner
        trainer = TrainerBase(self.epochs, valid_on_train_set=True)
        trainer.train(self, train_dataloader, val_dataloader, test_dataloader, valid_func, writer)
        os.makedirs(os.path.dirname(self.checkpoint_path_pntrain_model), exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path_pntrain_model)
    def _predict(self, inputs, targets):  # 返回每个批次的预测结果
        #判断数据是否为极值
        is_extreme = is_extreme_data(targets, lower_threshold=-self.extreme_min, upper_threshold=self.extreme_max,
                                     extreme_ratio=self.extreme_ratio)
        if self.pattern in ['train_whole','finetune','merge_test']: # 这些模型下选择是否使用融合策略
            if self.merge:# 融合正常模型和极端模型的结果
                # label = inputs[..., 3] # shape: # shape: [8, 12, 307] input极值分布（用历史窗口的极值分布来代替预测窗口的极值分布来选择模型）
                label = targets[..., 3]  # shape: # shape: [8, 12, 307] 概率标签或0,1标签，用分类器的结果来选择预测模型
                label = label.unsqueeze(-1)  # shape: [8, 12, 307, 1]
                output_no = self.model(inputs) # 正常模型得到的预测
                output_ex = self.model_extreme(inputs) # 极值模型得到的预测
                if self.use_possibility: # 使用概率加权，用分类标签的概率直接加权两模型的结果
                    output = label * output_ex + (1 - label) * output_no
                else: # 使用硬切换，label==1 的位置取极端模型输出，否则取正常模型输出。
                    output = torch.where(label == 1, output_ex, output_no)

                targets = targets[..., :self.output_dim] # 只保留回归目标需要的通道
                output = self.scaler.inverse_transform(output)
                y_lbl = self.scaler.inverse_transform(targets)
                return output.detach().cpu().numpy(), y_lbl.detach().cpu().numpy(), is_extreme

                # 用正常模型的极值分布或者极值模型的极值分布来加权融合两模型
                # output_no = self.model(inputs)
                # output_ex = self.model_extreme(inputs)
                # # extreme_mask = (output_no > self.extreme_max) | (output_no < -self.extreme_min)#正常模型极值分布
                # extreme_mask = (output_ex > self.extreme_max) | (output_ex < -self.extreme_min)#极值模型极值分布
                # output =  torch.where(extreme_mask, output_ex, output_no)
                # output = self.scaler.inverse_transform(output)
                # targets=targets[...,: self.output_dim]
                # y_lbl = self.scaler.inverse_transform(targets)
                # return output.detach().cpu().numpy(), y_lbl.detach().cpu().numpy(), is_extreme
            else: # 不采用融合策略，如果判定该 batch 属于极端样本，就用极端模型预测，否则用正常模型预测；
                # if is_extreme == 1:
                #     output = self.model_extreme(inputs)
                # elif is_extreme == 0:
                #     output = self.model(inputs)

                # output = self.model_extreme(inputs) # 只用微调出的极值模型预测

                output = self.model(inputs)  # 只用正常模型预测
                output = self.scaler.inverse_transform(output)
                targets = targets[..., :self.output_dim]
                y_lbl = self.scaler.inverse_transform(targets)
                return output.detach().cpu().numpy(),y_lbl.detach().cpu().numpy(), is_extreme

        elif self.pattern in ['train', 'train_alone','extreme_train','oversampling']: # 这些模式下只有单模型，直接预测
            output = self.model(inputs)  # 将输入、目标和选择的数据集传入模型，进行前向传播，得到输出 out。
            targets = targets[..., :self.output_dim]

            output = self.scaler.inverse_transform(output)
            y_lbl = self.scaler.inverse_transform(targets)


            return output.detach().cpu().numpy(),y_lbl.detach().cpu().numpy(),is_extreme

    def eval_data(self, dataloader, metric, inbatch=None) -> float:  # 评估模型性能（可以在训练集，验证集或者测试集上评估）


        self.model.eval()

        Y = []
        Pred = []
        with torch.no_grad():
            for inputs, targets in  dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets[..., :self.output_dim].squeeze(0).to(self.device)
                out = self.model(inputs)  # 将输入、目标和选择的数据集传入模型，进行前向传播，得到输出 out。

                out = self.scaler.inverse_transform(out)
                targets = self.scaler.inverse_transform(targets)
                loss_pred = self.loss(out, targets)
                Y.append(targets.detach().cpu().numpy()) # 真实值
                Pred.append(out.detach().cpu().numpy()) # 预测值
        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        self.model.train()
        return metric(Y, Pred)

    def predict(self, test_dataloader, cb_progress=lambda x: None):  # 返回所有的预测结果Y,Y是一个列表，其中每个元素对应一个批次的预测结果
        """
        Args:
    .        ds: TSForecastingDataset 结构数据， 在模型运行结束时运行
            Return: 训练结束后模型在测试集运行结果
        """
        Y = [] # 真实值（逐批收集）
        Pred = [] # 预测值（逐批收集）
        e_num = 0  # 统计：标记为“极值路径”的批次数
        n_num = 0  # 统计：标记为“正常路径”的批次数
        # self.model = self._build_model()
        # self.model.load_state_dict(
        #     torch.load(self.checkpoint_path_standard_model, map_location=self.device))
        # self.model = self.model.to(self.device)
        if self.pattern == 'merge_test': # 加载正常模型和极值模型
            # 1) 加载“标准模型”（正常模型）
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))  # 加载标准的模型权重到当前模型中
            self.model = self.model.to(self.device)
            # 2) 加载“极端模型”（在标准模型上微调得到）
            standard_extreme_model_path = self.checkpoint_path_standard_extreme_model
            self.model_extreme = self._build_model()
            self.model_extreme.load_state_dict(
                torch.load(standard_extreme_model_path, map_location=self.device))  # 加载预训练的模型权重到当前模型中
            self.model_extreme = self.model_extreme.to(self.device)


        if self.pattern == 'finetune': # 极值模型已经微调好，只需要加载正常模型
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))  # 加载标准的模型权重到当前模型中
            self.model = self.model.to(self.device)
        if self.pattern in ['train_whole', 'finetune','merge_test']:
            # 若当前流程会用到极端模型做推理，则将其切到 eval 模式（关闭 Dropout/BN 的训练分支）
            self.model_extreme.eval()
        # 标准模型设置为 eval 模式
        self.model.eval()

        with torch.no_grad(): # 评估阶段不需要梯度
            for inputs, targets in test_dataloader:
                # 取出一个 batch，并把最前面的 batch 维度（若为1）去掉，再搬到指定设备
                inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
                pred,targets,flag = self._predict(inputs, targets)
                Y.append(targets)
                Pred.append(pred)
                if flag == 1:
                    e_num += 1
                elif flag == 0:
                    n_num += 1
            print(e_num,n_num)
        # cb_progress(1.0)

        pred_array = np.squeeze(np.concatenate(Pred))   # shape: (T, N)
        true_array = np.squeeze(np.concatenate(Y))      # shape: (T, N)

        #
        mean = self.scaler.mean  # shape: (N,)
        std = self.scaler.std    # shape: (N,)


        # === 2. 极值标签化（根据 self.extreme_max 和 extreme_min）===
        upper = mean + self.extreme_max * std
        lower = mean - self.extreme_min * std

        pred_label = ((pred_array > upper) | (pred_array < lower)).astype(int)
        true_label = ((true_array > upper) | (true_array < lower)).astype(int)

        # === 3. 差异统计分析 ===
        assert pred_label.shape == true_label.shape, f"预测与标签 shape 不一致：{pred_label.shape} vs {true_label.shape}"

        no = np.sum(true_label == 0)
        ex = np.sum(true_label == 1)
        total_diff = np.sum(pred_label != true_label)
        count_0_to_1 = np.sum((true_label == 0) & (pred_label == 1))
        count_1_to_0 = np.sum((true_label == 1) & (pred_label == 0))

        print(f"\n📊 极值识别分析：")
        print(f"实际正常点：{no}")
        print(f"实际极值点：{ex}")
        print(f"总不同点数：{total_diff}")
        print(f"原为 0，预测为 1（新增极值）：{count_0_to_1}")
        print(f"原为 1，预测为 0（删掉极值）：{count_1_to_0}")

        return pred_array, true_array

        # return np.squeeze(np.concatenate(Pred)),np.squeeze(np.concatenate(Y))

    def finetune(self,retrain_loader ):

        if self.pattern == 'finetune':  # 构建极端模型
            standard_model_path = self.checkpoint_path_standard_model
            self.model_extreme = self._build_model()
            # 从标准模型（正常模型）的权重初始化极端模型
            self.model_extreme.load_state_dict(torch.load(standard_model_path, map_location=self.device)) # 加载预训练的模型权重到当前模型中
            self.model_extreme = self.model_extreme.to(self.device)
        else:
            # 如果不是专门的 finetune 模式，就从当前 self.model 深拷贝一份作为极端模型
            self.model_extreme = copy.deepcopy(self.model)

        # ===== 冻结 Embedding 层参数 =====
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

        # ===== 只优化需要更新的参数 =====
        self.optimizer_extreme = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model_extreme.parameters()),
            lr=1e-4,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )
        # # 为极端模型构建优化器（Adam）
        # self.optimizer_extreme = torch.optim.Adam(
        #     self.model_extreme.parameters(),
        #     lr=1e-4,
        #     weight_decay=self.weight_decay,
        #     eps=1e-8,
        # )  # 定义了优化器  1e-5
        # 根据配置选择极端模型的损失函数
        if self.loss_type_extreme == 'mse':
            self.loss_extreme = nn.MSELoss()
        elif self.loss_type_extreme == 'mae':
            self.loss_extreme = nn.L1Loss()
        elif self.loss_type_extreme == 'huber':
            self.loss_extreme = nn.HuberLoss()
        elif self.loss_type_extreme == 'weighted_huber':
            # 自定义的加权 Huber，通常对极端样本加更大权重
            self.loss_extreme = WeightedHuberLoss(delta=1.0, weight_extreme=10.0)
        elif self.loss_type_extreme == 'extreme':
            # 自定义的“极值加权 MSE”，用阈值与 (mean,std) 判定极端与否并加权
            self.loss_extreme = WeightedExtremeMSELoss(extreme_threshold=1.6, mean=self.scaler.mean, std=self.scaler.std,normal_weight=self.normal_weight,extreme_weight=self.extreme_weight)
        else:
            raise ValueError(f"Unknown loss_type_extreme: {self.loss_type_extreme}")
        # 进入训练模式（启用 Dropout/BN 的训练分支）
        self.model_extreme.train()
        for epoch in range(self.finetune_epochs):
            # 用 tqdm 包装数据迭代器，显示进度条
            iter_data = (
                tqdm(
                    retrain_loader,
                    total=len(retrain_loader),
                    ncols=100,
                    desc=f"finetune {epoch}:" ))
            total_loss = 0.0
            num_batches = 0
            for i, (batch_x, batch_y) in enumerate(iter_data):
                # 梯度清零
                self.optimizer_extreme.zero_grad()
                pred, true = self._process_one_batch_model(batch_x, batch_y)
                # 反标准化：把标准化空间的预测/真值还原到原始量纲
                pred = self.scaler.inverse_transform(pred)
                true = self.scaler.inverse_transform(true)
                # 计算损失（注意：这里在“原始量纲”上计算）
                curr_loss = self.loss_extreme(pred.to(self.device), true)   # 这里要修改
                total_loss += curr_loss.item()
                num_batches += 1
                curr_loss.backward()
                self.optimizer_extreme.step()
            # 每个 epoch 打印平均损失
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")


        # torch.save(self.optimizer_extreme.state_dict(),self.finetune_path )

    def _process_one_batch_model(self, batch_x, batch_y):
        x = batch_x.float().to(self.device)
        batch_y = batch_y.float()
        outputs = self.model_extreme(x)

        true = batch_y[:, :, :, 0:1].to(self.device)  # 提取预测的真实值

        return outputs, true

def is_extreme_data(data, lower_threshold=-1.6, upper_threshold=1.6, extreme_ratio=0.2):
    """
    判断输入数据是否为极端数据。

    参数:
        data: torch.Tensor, shape=(batch_size, seq_len, num_nodes, 3)
              数据最后一维分别代表交通流量、日特征、周特征，数据已标准化
        lower_threshold: float, 下阈值，默认值为 -1.6
                         低于 lower_threshold 的点视为极端值
        upper_threshold: float, 上阈值，默认值为 1.6
                         高于 upper_threshold 的点视为极端值
        extreme_ratio: float, 极端点数量占比的阈值，默认是 10%
                       超过该比例的点数认为是极端数据

    返回:
        int: 若交通流量中极端点（超过上下阈值）的数量超过总点数的10%，返回1（极端数据），否则返回0（正常数据）
    """
    # 提取交通流量数据（标准化后的数据，假设在最后一维的第0个位置）
    traffic_flow = data[..., 0]

    # 判断哪些点是极端值（超出上下阈值范围的点）
    extreme_mask = (traffic_flow < lower_threshold) | (traffic_flow > upper_threshold)

    # 统计极端值点的数量
    extreme_count = torch.sum(extreme_mask)

    # 计算总点数
    total_points = traffic_flow.numel()

    # 判断极端值点的比例是否超过阈值（10% 默认）
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

        # 对极值点加权（假设极值范围已标准化到 ±1.6 以外）
        weight_mask = ((target > 1.6) | (target < -1.6)).float() * self.weight_extreme + 1
        loss = loss * weight_mask
        return loss.mean()

import torch
import torch.nn as nn

class WeightedExtremeMSELoss(nn.Module):
    def __init__(self, extreme_threshold=1.6, mean=0, std=0, normal_weight=1.0, extreme_weight=5.0):
        """
        :param extreme_threshold: 极值判定 z-score 阈值（标准差倍数）
        :param mean: 原始数据均值（非标准化）
        :param std: 原始数据标准差
        :param normal_weight: 正常值损失权重（建议设置为 1.0）
        :param extreme_weight: 极值损失权重（建议设置为 >1.0，如 5.0）
        """
        super().__init__()
        # 保存均值、标准差（用于恢复原始空间阈值）
        self.mean = mean
        self.std = std
        self.extreme_threshold = extreme_threshold
        # 正常样本权重与极值样本权重
        self.normal_weight = normal_weight
        self.extreme_weight = extreme_weight

    def forward(self, pred, target):
        """
        :param pred: 模型预测值 (B, T, N, 1)
        :param target: 真实值 (B, T, N, 1)
        :return: 加权 MSE
        """
        # 普通的 MSE 误差
        error = pred - target
        loss = error ** 2

        # 在原始数值空间计算极值阈值
        lower_bound = self.mean - self.extreme_threshold * self.std
        upper_bound = self.mean + self.extreme_threshold * self.std

        # 构造极值掩码：target 超过阈值就记为极值（=1.0），否则为正常（=0.0）
        is_extreme = ((target > upper_bound) | (target < lower_bound)).float()
        # 权重矩阵：极值点用 extreme_weight，正常点用 normal_weight
        weights = self.normal_weight * (1.0 - is_extreme) + self.extreme_weight * is_extreme
        weighted_loss = loss * weights
        return weighted_loss.mean()

        # # 分开求mean
        # # 分别计算极值和正常值损失
        # extreme_loss = (loss * is_extreme).sum()
        # normal_loss = (loss * (1 - is_extreme)).sum()
        # # 分别计算样本数（避免均值除以0）
        # extreme_count = is_extreme.sum().clamp(min=1.0)
        # normal_count = (1 - is_extreme).sum().clamp(min=1.0)
        # # 加权组合两个部分的平均损失
        # extreme_loss_mean = extreme_loss / extreme_count
        # normal_loss_mean = normal_loss / normal_count
        # # 使用权重进行加权
        # return self.extreme_weight * extreme_loss_mean + self.normal_weight * normal_loss_mean


