import logging
import os
import importlib
import shutil
import time
import hashlib


def myhash(data):
    return hashlib.md5(data.encode(encoding='UTF-8')).hexdigest()








def check_result_existing(log_file):
    # 检查结果是否已经存在，如果存在就返回True，否则返回False
    if os.path.isfile(log_file):
        return True
    return False


class ScenarioAbstract:

    def run(self, algorithm,protocol,metrics,data_path, out_path,over_write):

        data_name = os.path.splitext(data_path)[0]  # 去除路径包含的文件后缀名
        data_name = os.path.split(data_name)[-1]  # 去掉文件的父路径，只保留文件名
        """
        程序会根据out_path 指定的目录创建当前算法的类名相关的子目录，然后再创建一个与模型参数相关的hash目录，在hash目录下存放所有实验结果。
        例如：
        out_path = D:\data
        PMF类
        PMF对象的hash字符串为 348790274389023  (PMF的任何参数发生改变，has字符串都会改变，如果参数一样则字符串会一样。)

        最终的输出目录 out_path=D:\data\PMF\348790274389023   （如果目录不存在，则自动创建）
        结果文件： D:\data\PMF\348790274389023\result.log (存放模型参数和模型结果，如果该文件已经存在，说明已经运行过实验，则自动退出，)
        checkpoint_path:  D:\data\PMF\348790274389023\checkpoint  (存放模型的最优状态，模型训练用的)
        log_tensorboard: D:\data\PMF\348790274389023\log_tensorboard  (存放模型训练的中间结果)
        """
        # 记录实验配置到log文件
        cache_dir = os.path.join(out_path,
                                algorithm.class_name() + '-' + data_name,'cache') # 构造缓存目录路径
        out_path = os.path.join(out_path,
                                algorithm.class_name() + '-' + data_name, str(myhash(str(algorithm))))# 最终结果存放路径

        if os.path.isdir(out_path):  # 如果目录存在
            if over_write: # 重写目录
                shutil.rmtree(out_path) # 删除所有内容
                os.makedirs(out_path,exist_ok=True) #   # 重新创建
            else: # 是否中断继续
                if hasattr(algorithm,'resume_state') and algorithm.resume_state:
                    os.makedirs(out_path,exist_ok=True)   # 恢复训练，保留原有内容
                else:
                    print(f"结果目录已经存在{out_path}")
                    exit(-1)
        else: #  # 目录不存在，直接创建
            os.makedirs(out_path)

        log_path = os.path.join(out_path, 'result.log')  #最终结果存放的路径， os.path.join用于将多个路径组合成一个路径
        start_time = str(time.strftime("%Y%m%d%H%M%S", time.localtime())) # 开始时间

        algorithm.checkpoint_path = os.path.join('checkpoint',out_path, 'checkpoint') # 模型存放路径
        os.makedirs(os.path.join('checkpoint',out_path), exist_ok=True)
        # algorithm.checkpoint_path = os.path.join(out_path, 'checkpoint')
        if not hasattr(algorithm,"tensorboard_path") or not algorithm.tensorboard_path:  # 创建tensorboard路径,hasattr用于检查一个对象是否有某个属性
            algorithm.tensorboard_path = os.path.join(out_path, 'log_tensorboard')

        # 设置缓存目录，用于存放模型的中间结果。
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        if not hasattr(algorithm,'cache_dir') or not algorithm.cache_dir: # 如果算法对象没有 cache_dir 属性，或者该属性是空的，则把刚创建的 cache_dir 赋值给它
            algorithm.cache_dir = cache_dir

        logging.basicConfig(filename=log_path,  # 配置日志保存路径,记录 info 级别以上的日志,日志内容只打印 message，不含时间、等级等
                            level=logging.INFO,
                            format="%(message)s")
        logging.info('Train Start Time: ' + start_time)  # 程序启动时间
        logging.info("ds" + ":" + data_path) # 数据集路径
        logging.info("algorithm-parameter:")
        logging.info(algorithm)  # 打印模型参数

        #  实验代码
        results = protocol.test(algorithm, metrics)
        #  实验结束后，记录结果、运行时间
        logging.info("RESULTS=" + str(results))  # 保存指标和结果
        logging.info('Train Stop Time: ' + str(time.strftime("%Y%m%d%H%M%S", time.localtime())))  # 记录结束时间

        assert len(results) == len(metrics)

        return results  # 返回每一个指标的测评结果。

    def parse_data(self, data_dir):
        raise Exception("This is an Abstract Class, Do Not Call It")
