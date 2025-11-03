
import io
import logging

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    '''
    Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            tmpFile (file): file for the checkpoint to be saved to.
                            Default: 'io.BytesIO()'
            trace_func (function): trace print function.
                            Default: print
    '''
    def __init__(self, patience=7, verbose=False, delta=0,  trace_func=print, tmpFile=io.BytesIO()):
        self.patience = patience
        self.verbose = verbose # 是否打印每次提升的提示信息。
        self.counter = 0 # 连续未提升的 epoch 数量
        self.best_score = None # 当前最优得分
        self.early_stop = False # # 是否触发早停
        self.val_score = float("-inf") # 当前保存的指标值
        self.delta = delta # 指标必须提升至少 delta 才算有效提升。
        self.tmpFile = tmpFile
        self.trace_func = trace_func  #<class 'builtin_function_or_method'># 打印日志用的函数，默认是 print，可以换成 logger.info。

    def __call__(self, score, model,epch): # 每个 epoch 验证后调用，判断是否早停
        if self.best_score is None: # 如果是第一次调用，初始化 best_score 并保存模型
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:#指标未提升
            self.counter += 1 # 增加早停计数器 counter
            logging.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}, at Epcho {epch}')
            if self.counter >= self.patience: # 如果计数器超过容忍次数 patience，就触发 self.early_stop = True，训练器那边就会中断循环
                self.early_stop = True
        else: # 指标提升，更新最优分数并保存模型
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def get_best(self, model):
        if isinstance(self.tmpFile,io.IOBase):
            self.tmpFile.seek(0)
        return model.load(self.tmpFile)

    def save_checkpoint(self, val_loss, model): # 当模型指标提升时调用，保存模型权重。
        '''Saves models when validation loss decrease.'''
        if self.verbose: # 如果开启了 verbose，输出提升信息。
            logging.info(f'validate metric increased {self.val_score:.6f} --> {val_loss:.6f}).  Saving models ...')
        if isinstance(self.tmpFile, io.IOBase): # 将模型写入 tmpFile 中（一般是内存文件或文件路径）
            self.tmpFile.seek(0)
        model.save(self.tmpFile)
        self.val_score = val_loss