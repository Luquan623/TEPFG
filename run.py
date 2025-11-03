import importlib
import io
import time

import yaml

import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ATP: 你基本不需要修改此文件！！！

def smart_convert(value):
    assert isinstance(value, str)
    if value.count('.') > 0:
        try:
            return float(value)
        except:
            pass
    try:  # 检查整数
        return int(value)
    except:
        return value

def set_seed(seed=2333):
    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def need_import(value):
    """

    Parameters
    ----------
    value 如果传入值是字符串，而且前后都用`_`包住，则需要导入，返回True，否则返回False

    Returns
    -------

    """
    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False


def create_obj_from_json(js):
    """

    Parameters
    ----------
    js  json对象，所有key，value，如果以前后以`_`包住的字符串将被当作对象进行创建。

    Returns
    -------
    返回创建的对象
    """

    if isinstance(js, dict): #传入是字典
        rtn_dict = {}
        for key, values in js.items():   # 遍历字典的键值对，key为算法类名，values为参数字典
            if need_import(key):
                assert values is None or isinstance(values,
                                                    dict), f"拟导入的对象{key}的值必须为dict或None，用于初始化该对象"
                assert len(js) == 1, f"{js} 中包含了需要导入的{key}对象，不能再包含其他键值对"
                key = key[1:-1]  # 去掉 key的前后 `_`
                cls = my_import(key) # cls 是对应的模型类
                if "__init__" in values:  # 如果该类被设置了初始化函数，则需要读取初始化函数的值
                    assert isinstance(values, dict), f"__init__ 关键字，放入字典对象，作为父类{key}的初始化函数"
                    init_params = create_obj_from_json(values['__init__'])  # 获取初始化的参数，并返回。
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)  # 清空"__init__"相关的值，方便后续处理。
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
                # 此处已经不包含 "__init__"的key，value对
                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js,str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else: # 其他对象直接返回
        return js


def my_import(name):  # name指定的算法类名，eg：algorithm.pmf.MF
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)  # 在py文件中获取在pmf.py文件下获取模型类 <classes 'algorithm.pmf.MF'>
    return cls  # 返回的是类


def myloads(jstr):
    if hasattr(yaml, 'full_load'):
        js = yaml.full_load(io.StringIO(jstr))
    else:
        js = yaml.load(io.StringIO(jstr))
    if isinstance(js, str):
        return {js: {}}
    else:
        return js


start_time = str(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

parser = argparse.ArgumentParser(description='算法测试程序')
parser.add_argument('-f', dest='argFile', type=str, required=True,
                    default=None,
                    help='通过YAML文件指定试验参数文件。')
parser.add_argument('-w', dest='over_write', type=bool,
                    help='强制覆盖已经存在的结果')
parser.add_argument('-o', dest='out_path', type=str, required=False,
                    help='指定结果存放路径')
parser.add_argument('-s', dest='scenario', type=myloads, required=False,
                    help='选择需要的试验类，在scenario.py文件中查找。')
parser.add_argument('-p', dest='protocol', type=myloads, required=False,
                    help='验证方法')
parser.add_argument('-m', dest='metrics', type=str, required=False,
                    help='验证指标列表，逗号分割,如atp.metric.metric.RMSE, atp.metric.metric.MAE')
parser.add_argument('-a', dest='alg', type=str, required=False,
                    help='需要验证的算法')
parser.add_argument('-d', dest='data_path', type=str, required=False,
                    help='数据目录或者数据集压缩包')
parser.add_argument('-r', dest='params', type=myloads, required=False,
                    # default="{}",
                    help='''算法参数，是一个json，例如"{d: 20,lr: 0.1,n_itr: 1000}" ''')


# python run.py -f atp/config/logistic_iris.yaml
# python run.py -s scenario.ScenarioRec -r {d:50,lr:0.001,n_itr:1000}
# tensorboard启动命令：tensorboard --logdir=log --port 8888
#

def update_parameters(param: dict, to_update: dict) -> dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k], (dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k] = to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param


def str_obj_js(mystr):
    # 将字符串转化为 描述对象的json
    if ':' in mystr:
        return myloads(mystr)
    else:
        return {mystr: {}}


def enclose_class_name(value):  # 为类名的前后加上“_”
    if isinstance(value,dict): # 如果vale是一个字典,{apt.scenario.ScenarioAbstract:{...}}
        assert len(value)==1, "只能有一个类"
        for k,v in value.items():  # 遍历字典的键值对
            if k[0]==k[-1]=="_":
                return {k:v}
            else:
                return {f"_{k}_":v}
    elif isinstance(value,str): # 如果vale是一个字符串,apt.scenario.ScenarioAbstract
        #如果发现value一定是类名，则自动给其添加前后`_`
        if value[0]==value[-1]=="_":
            return value
        else:
            return f"_{value}_"
    else:
        return value
def parse_objects(filedict):
    algorithm = create_obj_from_json(enclose_class_name({filedict['algorithm']:filedict['algorithm_parameters']}))
    protocol = create_obj_from_json(enclose_class_name(filedict['protocol']))
    scenario = create_obj_from_json(enclose_class_name(filedict['scenario']))
    metrics = []
    for m in filedict['metrics']:
        metrics.append(create_obj_from_json(enclose_class_name(m)))

    return scenario, protocol, algorithm, metrics


if __name__ == '__main__':
    args = parser.parse_args()

    filedict = {}  # 这里存放yaml文件内的所有参数
    if args.argFile is not None:  # 通过YAML文件指定试验参数文件
        filelist = args.argFile.split(',')  # split(',')将一个字符串按照逗号进行拆分,这里可以是指定多个yaml文件
        for fname in filelist:  # 依次打开每个yaml文件
            with open(fname.strip(), 'rb') as infile:  # 打开指定文件
                fd = yaml.safe_load(infile)  # 安全地加载YAML格式的数据
                update_parameters(filedict, fd)  # 将参数放入filedict字典




    # metrics 默认必须是数组,如果是字符串，则自动转化为数组
    if 'metrics' in filedict and isinstance(filedict['metrics'],str):  # "isinstance"用于判断一个对象是否属于某个类或其子类。
        filedict['metrics'] = [filedict['metrics']]

    # 这里是用arg_dict来存放命令行定义的参数
    m = None if args.metrics is None else args.metrics.split(',')
    arg_dict = {
        "algorithm": args.alg,
        "algorithm_parameters": args.params,
        "protocol": args.protocol,
        "data_path": args.data_path,
        "out_path": args.out_path,
        "over_write": args.over_write,
        "metrics": m}

    update_parameters(filedict, arg_dict)  # 将命令行传入的参数更新到YAML配置文件指定的参数中
    # 当配置文件和命令行有相同的参数时，以命令行的参数为准，命令行参数优先级最高



    # 根据filedict的内容创建出场景对象（scenario）、协议对象（protocol）、算法对象（algorithm）和指标对象的列表（metrics）
    scenario, protocol, algorithm, metrics = parse_objects(filedict)

    data_path = filedict['data_path']  # 数据所在路径
    out_path = filedict['out_path']    # 输出结果保存的路径
    over_write = filedict['over_write']

    assert data_path, "必须给定数据目录"
    if not out_path: out_path='log' # 默认输出目录为 log 目录

    # 设置随机种子
    set_seed(12) # 默认43

    results = scenario.run(algorithm, protocol, metrics,data_path, out_path, over_write)  # 调度入口
    #print(results.is_cuda)
    print("Final Results is :", results, filedict)
