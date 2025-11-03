import io
import numpy as np
import pickle
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ts_forecasting_traffic.model.earlystopping import EarlyStopping

class TrainerBase:
    def __init__(self, epochs, evaluate_steps=1,
                 valid_on_train_set = False,
                 valid_on_test_set = True,
                 verbose=True):
        """
        epochs: 训练轮数
        evaluate_steps: Default(1)验证次数，如果<=0，则不进行任何验证。 否证训练evaluate_steps次epochs训练后保存一次最优模型。
        valid_on_train_set： Default（False）,是否在训练集上计算验证指标。
        valid_on_test_set:  Default（True）,是否在测试集上计算验证指标。
        verbose:Defult(True),是否打印中间过程

        早停的参数配置：见 trainer.earlyStopping.EarlyStopping
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            tmpFile (file): file for the checkpoint to be saved to.
                            Default: 'None'
            trace_func (function): trace print function.
                            Default: print

        参数由self.create_EarlyStopping() 函数，从model中读取。model中的参数来自于配置文件。

        example:
        早停机制的使用：
            (1) 根据训练集的loss值是否降低来判断是否早停  ,evaluate_steps=1
                trainer = TrainerBase(epochs,evaluate_steps=1)
                trainer.train(self, train_loader, None, None, writer)
            (2) 根据训练集的metric值是否提升来判断是否早停  ,evaluate_steps=1, vald_on_train_set = True
                trainer = TrainerBase(epochs,vald_on_train_set = True，evaluate_steps=1)
                trainer.train(self, train_loader, None, valid_func, writer)
            (3) 根据验证集的metric值是否提升来判断是否早停机制  ,evaluate_steps=1, 训练时给出valid_loader
                trainer = TrainerBase(epochs,evaluate_steps=1)
                trainer.train(self, train_loader, valid_loader, valid_func, writer)
            (4) 不使用早停  ,evaluate_steps=0
                trainer = TrainerBase(epochs,evaluate_steps=0)
                trainer.train(self, train_loader, None, None, writer)

        """
        self.epochs = epochs
        self.verbose = verbose
        self.evaluate_steps = evaluate_steps
        # self.valid_on_train_set = valid_on_train_set
        self.valid_on_train_set = False
        self.valid_on_test_set = False

    def save_state(self, filepath,epoch,early_stopping):
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        state={'epocho': epoch,
        'earl_stopping':early_stopping
        }
        with open(trainer_filepath,'wb') as fout:
            pickle.dump(state, fout)

    def load_state(self, filepath,epoch,early_stopping):
        state = {'epocho': epoch,
                 'earl_stopping': early_stopping
                 }
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        try :
            with open(trainer_filepath,'rb') as fin:
                state = pickle.load( fin)
        except:
            pass
        return state['epocho'],state['earl_stopping']


    def _train_epoch(self, model, train_loader, epoch):  # 训练完一个epoch，返回每个批次的平均损失
        """
        完成一轮训练
        return: 字典，必须包含平均损失{loss：xxx},
            assert 'loss' in result
        """
        iter_data = ( # 则 iter_data 会被包装成一个 tqdm 对象，并可视化训练进度
            tqdm(
                train_loader,
                total=len(train_loader), # 提示进度条长度，用于显示完成度。
                ncols=100, # 在命令行中指定进度条的宽度。
                desc=f"Train {epoch}:>5" # 进度条左侧显示的描述信息
            )
            if self.verbose # 判断是否启用进度条
            else train_loader
        )
        headers = None
        data = []
        for inputs, targets in iter_data:
            # batch是train_loader的一个batch
            result = model.opt_one_batch(inputs, targets) # 一个批次的损失
            assert isinstance(result, dict),   "opt_one_batch 返回数据必须是dict类型"
            assert 'loss' in result, "opt_one_batch 返回的字典必须包含loss关键字"
            if headers is None :
                headers = result.keys()
            data.append(list(result.values())) # 提取 result 的所有值（转为列表形式），添加到 data 中
        # if len(iter_data) > 1:
        data = np.mean(np.array(data), axis=0)  # 按照数据的最后一个维度求均值，data中必须不能是tensor数据，
        # data = np.mean(data,axis=1)[-1] #按照数据的最后一个维度求均值
        return dict(zip(headers,data)) # 返回一个字典

    def _eval_data(self, model, dataloader, valid_fun):
        # 进行一次测试，数据可能来自于训练集，也可能来自于测试集
        return model.eval_data(dataloader, valid_fun)


    def create_EarlyStopping(self, model):
        """
        根据模型中的参数，创建早停函数
        """
        patience = 10
        delta = 0 # 指标需要至少提升这么多才算“改进”
        trace_func = print # 记录信息的函数，这里用的是 print
        if hasattr(model, 'checkpoint_path'):
            checkpoint = model.checkpoint_path
        else:
            checkpoint = 'checkpoint'
        if hasattr(model, 'es_patience'): # 允许用户在模型类中自定义 es_patience
            patience = model.es_patience
        if hasattr(model, 'es_delta'): # # 允许用户在模型类中自定义 es_delta
            delta = model.es_delta

        return EarlyStopping(patience, self.verbose, delta, trace_func, checkpoint)

    def train(self, model, train_loader, valid_loader,test_loader, valid_func, loger=sys.stdout):
        """
        param:
            model:算法模型
            train_loader:训练集loader
            valid_loader:验证集loader
            test_loader :测试集loader
            valid_func:验证函数
            loger:打印，默认值为：sys.stdout
        ______________________
        执行过程：
            1.获得评价指标名字
            2.根据model中的参数初始化早停
            3.对于每一个epoch，执行以下过程：
            4.      调用_train_epoch执行一次训练，并返回损失值
            5.      如果开启了打印函数，则打印一次训练返回的所有结果，结果中必须包含loss值
            6.      如果不进行验证，只做训练。则回到第4步
            7.      如果满足验证条件，则开始验证
            8.          是否需要在训练集上计算指标?若需要，则调用_eval_data函数并打印train_score
            9           如果验证集不为空，模型则在验证集上指标结果作为筛选模型的标准
            10.         如果开启了训练集的验证标志，优先使用该指标为筛选模型的标准
            11.         既没有提供验证集，又不在训练集上计算指标时，就直接使用训练集上的损失值作为筛选模型的标准。
            12.     判断验证集score用于早停，分数越小越好，则取反进行判断
            13.     保存当前最优模型，因为是在训练类里面实现的load和save。
            14.若模型有验证过程，则加载保存的最优模型
        _______________________
        return:
            没有返回值
        """

        assert hasattr(model,
                       'opt_one_batch'), f"模型 {model} 必须实现, ModelAbstract的 opt_one_batch 函数才能使用 TrainerBase"
        assert hasattr(model,
                       'eval_data'), f"模型 {model} 必须实现, ModelAbstract的 opt_one_batch 函数才能使用 eval_data"
        assert hasattr(model,
                       'save'), f"模型 {model} 必须实现, ModelAbstract的 opt_one_batch 函数才能使用 save"
        assert hasattr(model,
                       'load'), f"模型 {model} 必须实现, ModelAbstract的 opt_one_batch 函数才能使用 load"

        # 将第epoch个的score打印出来
        def printf(key, value, epoch): # 将中间过程打印到日志中
            if isinstance(loger, SummaryWriter):# 检查 loger 是否是一个 SummaryWriter 对象
                loger.add_scalar(key, value, global_step=epoch) # 将关键字 key 和对应的值 value 记录到 TensorBoard 日志中，并指定当前的全局步数为 epoch
            elif loger == sys.stdout: # 如果 loger 不是 SummaryWriter 对象，而是标准输出流 sys.stdout（即打印到控制台）
                line = f"{key}={value}\t\t  epocho={epoch}" # 构造一条信息记录的字符串，包括关键字 key、对应的值 value，以及当前的迭代轮数 epoch
                loger.write(line + "\n")
                #loger.writelines(line)  # 将构造的信息记录字符串写入到标准输出流中（通常是控制台），以便在控制台中查看

        metric_name = str(valid_func)  # 获得评价指标名字
        early_stoping = self.create_EarlyStopping(model)  # 根据model中的参数初始化早停
        epoch_start=0

        # 开始迭代
        for epoch in range(epoch_start+1, self.epochs + 1):
            if early_stoping.early_stop: break
            results = self._train_epoch(model, train_loader, epoch)  # 一次轮训练，并返回损失值
            if self.verbose:
                for key, value in results.items(): #打印一次训练返回的所有结果，必须包含loss值
                    #printf(key, value, epoch)
                    printf(f"{key}/train", value, epoch)
            # 不进行验证，只做训练。
            if self.evaluate_steps <=0 :
                continue

            # 如果满足验证条件，则开始验证
            if (epoch - 1) % self.evaluate_steps == 0 :  # 当前训练需要 计算验证指标
                # 模型在验证集上的结果
                if valid_loader is not None:  # 如果验证集不为空，模型则在验证集上指标结果作为筛选模型的标准
                    # val_score = self._eval_data(model, valid_loader, valid_func)
                    # printf(f"{metric_name}@valid set", val_score, epoch)
                    val_score = self._eval_data(model, valid_loader, valid_func)
                    printf(f"{metric_name}/val", val_score, epoch)

            if valid_loader != None and  hasattr(valid_func, 'bigger') and valid_func.bigger == False:
                val_score = -val_score  # 分数越小越好，则取反进行判断

            early_stoping(val_score, model, epoch)# 保存当前最优模型，因为是在训练类里面实现的load和save。

        if self.evaluate_steps > 0:  #模型有验证过程
            early_stoping.get_best(model)  # 加载保存的最优模型