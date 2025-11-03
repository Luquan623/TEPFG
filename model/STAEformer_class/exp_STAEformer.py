from ts_forecasting_traffic.model.trainer import TrainerBase
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from atp.model.torchmodel import TorchModel
import numpy as np
import torch
import os
from ts_forecasting_traffic.model.STAEformer_class.model.STAEformer import Model
from tqdm import tqdm
import copy

import matplotlib.pyplot as plt



class Exp_STAEformer_class(TorchModel):
    def __init__(self):
        super(Exp_STAEformer_class,self).__init__()
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
        self.pred = 1

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
        self.finetune_path = 'ts_forecasting_traffic/checkpoints/STAEformer/PEMS04/finetune_type/checkpoint.pth'
        self.label = False
        self.label_save = "result/PEMS04/label/predictions00.npy"
        self.best_threshold = 0.5
        self.val_loss_curve = []  # 用于记录每个 epoch 的验证损失
        self.use_GMM = False
        self.save_possibility = False
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
        targets = targets[..., -self.output_dim:].float()

        # # 拆解每一列
        # x0 = inputs[..., 0:1]  # 第 1 个特征
        # x1 = inputs[..., 1:2]  # 第 2 个特征
        # x2 = inputs[..., 2:3]  # 第 3 个特征
        # x3 = inputs[..., 3:4]  # 第 4 个特征
        # # 调换顺序：将 x0 和 x3 对调
        # inputs = torch.cat([x3, x1, x2, x0], dim=-1)

        out = self.model(inputs)
        self.optimizer.zero_grad()



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
        ratio = 10
        pos_weight = torch.tensor([ratio], device=self.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        writer = SummaryWriter(self.tensorboard_path)
        trainer = TrainerBase(self.epochs, valid_on_train_set=True)

        trainer.train(self, train_dataloader, val_dataloader, test_dataloader, valid_func, writer)
        os.makedirs(os.path.dirname(self.checkpoint_path_pntrain_model), exist_ok=True)
        self.plot_val_metric_curve(ylabel= valid_func.__class__.__name__)
        torch.save(self.model.state_dict(), self.checkpoint_path_pntrain_model)
    def _predict(self, inputs, targets):  # 返回每个批次的预测结果
        # # 拆解每一列
        # x0 = inputs[..., 0:1]  # 第 1 个特征
        # x1 = inputs[..., 1:2]  # 第 2 个特征
        # x2 = inputs[..., 2:3]  # 第 3 个特征
        # x3 = inputs[..., 3:4]  # 第 4 个特征
        # # 调换顺序：将 x0 和 x3 对调
        # inputs = torch.cat([x3, x1, x2, x0], dim=-1)

        output= self.model(inputs)
        probs = torch.sigmoid(output)
        if self.save_possibility:
            output = probs
        else:
            output  = ( probs > self.best_threshold).int()
        y_lbl =  targets[..., -self.output_dim:].float()
        return output.detach().cpu().numpy(),y_lbl.detach().cpu().numpy()

    def eval_data(self, dataloader, metric, inbatch=None) -> float:  # 评估模型性能（可以在训练集，验证集或者测试集上评估）


        self.model.eval()
        Y = []
        Pred = []
        with torch.no_grad():
            for inputs, targets in  dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets[..., -self.output_dim:].squeeze(0).float().to(self.device)
                # # 拆解每一列
                # x0 = inputs[..., 0:1]  # 第 1 个特征
                # x1 = inputs[..., 1:2]  # 第 2 个特征
                # x2 = inputs[..., 2:3]  # 第 3 个特征
                # x3 = inputs[..., 3:4]  # 第 4 个特征
                # # 调换顺序：将 x0 和 x3 对调
                # inputs = torch.cat([x3, x1, x2, x0], dim=-1)

                out = self.model(inputs)  # 将输入、目标和选择的数据集传入模型，进行前向传播，得到输出 out。
                probs = torch.sigmoid(out)
                out = ( probs > self.best_threshold).int()
                Y.append(targets.detach().cpu().numpy()) # 真实值
                Pred.append(out.detach().cpu().numpy()) # 预测值
        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        self.model.train()
        score = metric(Y, Pred)
        self.val_loss_curve.append(score)
        return score

    def predict(self, test_dataloader, cb_progress=lambda x: None):  # 返回所有的预测结果Y,Y是一个列表，其中每个元素对应一个批次的预测结果
        """
        Args:
    .        ds: TSForecastingDataset 结构数据， 在模型运行结束时运行
            Return: 训练结束后模型在测试集运行结果
        """
        # self.model = self.model.to(self.device)
        if self.pattern == 'test':
            standard_model_path = self.checkpoint_path_standard_model
            self.model = self._build_model()
            self.model.load_state_dict(
                torch.load(standard_model_path, map_location=self.device))  # 加载标准的模型权重到当前模型中
            self.model = self.model.to(self.device)
        Y = [] # 真实值
        Pred = []
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
                pred,targets = self._predict(inputs, targets)
                Y.append(targets)
                Pred.append(pred)
        # 1. 提取目录部分
        save_dir = os.path.dirname(self.label_save)
        # 2. 自动创建上级目录
        os.makedirs(save_dir, exist_ok=True)
        # 合并 & squeeze
        pred_array = np.concatenate(Pred, axis=0)
        if pred_array.shape[-1] == 1:
            pred_array = np.squeeze(pred_array, axis=-1)
        # 保存为 .npy 文件
        np.save(self.label_save, pred_array)
        # 保存为 .csv 文件（转换为 2D 再保存）
        csv_ready_array = pred_array.reshape(pred_array.shape[0], -1)
        csv_path = os.path.join(save_dir, "predictions00.csv")
        np.savetxt(csv_path, csv_ready_array, delimiter=",", fmt="%.4f")
        if not self.save_possibility:
            # 1. 总点数（所有元素数量）
            total_points = pred_array.size
            # 2. 极值点数（值为1的数量）
            extreme_points = np.sum(pred_array == 1)
            # 3. 极值比例（百分比）
            extreme_ratio = extreme_points / total_points
            print(f"总预测点数: {total_points}")
            print(f"预测为极值（1）的点数: {extreme_points}")
            print(f"极值比例: {extreme_ratio:.4%}")
        return np.squeeze(np.concatenate(Pred)),np.squeeze(np.concatenate(Y))

    # # 遍历多个阈值  需要self.save_possibility为True
    # def predict(self, test_dataloader, cb_progress=lambda x: None):
    #     if self.pattern == 'test':
    #         standard_model_path = self.checkpoint_path_standard_model
    #         self.model = self._build_model()
    #         self.model.load_state_dict(torch.load(standard_model_path, map_location=self.device))
    #         self.model = self.model.to(self.device)
    #
    #     Y = []
    #     Pred = []
    #     self.model.eval()
    #     with torch.no_grad():
    #         for inputs, targets in test_dataloader:
    #             inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
    #             pred, targets = self._predict(inputs, targets)
    #             Y.append(targets)
    #             Pred.append(pred)
    #
    #     save_dir = os.path.dirname(self.label_save)
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     pred_array = np.concatenate(Pred, axis=0)
    #     if pred_array.shape[-1] == 1:
    #         pred_array = np.squeeze(pred_array, axis=-1)
    #     y_array = np.concatenate(Y, axis=0)
    #     if y_array.shape[-1] == 1:
    #         y_array = np.squeeze(y_array, axis=-1)
    #
    #     # 遍历多个阈值
    #     threshold_list = np.arange(0.1, 1.0, 0.1)
    #     for threshold in threshold_list:
    #         binary_pred = (pred_array >= threshold).astype(int)
    #         # 保存 npy
    #         npy_path = os.path.join(save_dir, f"pred_threshold_{threshold:.2f}.npy")
    #         np.save(npy_path, binary_pred)
    #         # 打印信息
    #         total_points = binary_pred.size
    #         extreme_points = np.sum(binary_pred == 1)
    #         extreme_ratio = extreme_points / total_points
    #         print(f"[阈值={threshold:.2f}] 极值点数: {extreme_points}, 占比: {extreme_ratio:.2%}")
    #
    #     return np.squeeze(pred_array), np.squeeze(y_array)

    # def predict(self, test_dataloader, cb_progress=lambda x: None):
    #     """
    #     执行模型预测 + PR 曲线分析 + 自动最优阈值选择
    #     """
    #     if self.pattern == 'test':
    #         standard_model_path = self.checkpoint_path_standard_model
    #         self.model = self._build_model()
    #         self.model.load_state_dict(
    #             torch.load(standard_model_path, map_location=self.device))
    #         self.model = self.model.to(self.device)
    #
    #     self.model.eval()
    #
    #     Y = []  # ground truth
    #     Logits = []  # raw model output (no sigmoid)
    #
    #     with torch.no_grad():
    #         for inputs, targets in test_dataloader:
    #             inputs, targets = inputs.squeeze(0).to(self.device), targets.squeeze(0).to(self.device)
    #
    #             # # 拆解顺序
    #             # x0 = inputs[..., 0:1]
    #             # x1 = inputs[..., 1:2]
    #             # x2 = inputs[..., 2:3]
    #             # x3 = inputs[..., 3:4]
    #             # inputs = torch.cat([x3, x1, x2, x0], dim=-1)
    #
    #             logits = self.model(inputs)  # (B, T, N, 1) raw logits
    #             probs = torch.sigmoid(logits)  # (B, T, N, 1)  映射成概率
    #             Logits.append( probs.detach().cpu().numpy())
    #             Y.append(targets[..., -self.output_dim:].detach().cpu().numpy())
    #
    #     # === 合并 logits 和标签 ===
    #     logits_array = np.concatenate(Logits, axis=0)  # (B, T, N, 1)
    #
    #     y_true = np.concatenate(Y, axis=0)  # (B, T, N, 1)
    #
    #     # 去掉尾部维度（如果为 1）
    #     if logits_array.shape[-1] == 1:
    #         logits_array = np.squeeze(logits_array, axis=-1)  # -> (B, T, N)
    #     if y_true.shape[-1] == 1:
    #         y_true = np.squeeze(y_true, axis=-1)  # -> (B, T, N)
    #
    #     # reshape 成二维结构 (时间 × 节点)
    #     logits_2d = logits_array.reshape(-1, logits_array.shape[-1])  # (B×T, N)
    #     y_true_2d = y_true.reshape(-1, y_true.shape[-1])  # (B×T, N)
    #
    #     # === PR 分析 ===（flatten 后评估）
    #     logits_flat = logits_2d.flatten()
    #     y_true_flat = y_true_2d.flatten()
    #
    #     from sklearn.metrics import precision_recall_curve, classification_report
    #     import matplotlib.pyplot as plt
    #
    #     precision, recall, thresholds = precision_recall_curve(y_true_flat, logits_flat)
    #     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    #     best_idx = np.argmax(f1_scores)
    #     best_threshold = thresholds[best_idx]
    #
    #     print(f"\n📈 [PR 曲线分析]")
    #     print(f"最佳阈值: {best_threshold:.3f}")
    #     print(f"Precision: {precision[best_idx]:.4f}")
    #     print(f"Recall   : {recall[best_idx]:.4f}")
    #     print(f"F1 Score : {f1_scores[best_idx]:.4f}")
    #
    #     # 可视化
    #     plt.figure()
    #     plt.plot(recall, precision, label='PR Curve')
    #     plt.scatter(recall[best_idx], precision[best_idx], color='red', label='Best Threshold')
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title("Precision-Recall Curve")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # === 分类预测（保留二维结构）
    #     y_pred_best_2d = (logits_2d > best_threshold).astype(int)  # (B×T, N)
    #
    #     # === 打印评估报告（用 1D）
    #     print("\n📊 使用最佳阈值的分类评估:")
    #     print(classification_report(y_true_flat, y_pred_best_2d.flatten(), digits=4))
    #
    #     # === 保存为 .npy
    #     save_dir = os.path.dirname(self.label_save)
    #     os.makedirs(save_dir, exist_ok=True)
    #     np.save(self.label_save, y_pred_best_2d)  # 直接保存二维标签
    #
    #     # === 极值比例统计
    #     total_points = y_pred_best_2d.size
    #     extreme_points = np.sum(y_pred_best_2d == 1)
    #     extreme_ratio = extreme_points / total_points
    #     print(f"\n🧾 总预测点数: {total_points}")
    #     print(f"预测为极值的点数: {extreme_points}")
    #     print(f"极值比例: {extreme_ratio:.4%}")
    #
    #     # === 返回结构清晰的结果
    #     return y_pred_best_2d, y_true_2d


    def plot_val_metric_curve(self, ylabel="Validation Metric"):
        """
        绘制每个 epoch 的验证集评估指标曲线。
        """
        if  len(self.val_loss_curve) == 0:
            print("⚠️ 未记录验证集评估指标，请确认是否已启用记录。")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.val_loss_curve) + 1), self.val_loss_curve, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} per Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.show()




