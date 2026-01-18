import importlib
import io
import time
import yaml
import argparse
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def smart_convert(value):
    assert isinstance(value, str)
    if value.count('.') > 0:
        try:
            return float(value)
        except:
            pass
    try:
        return int(value)
    except:
        return value


def set_seed(seed=2333):
    import random, os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def need_import(value):
    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False


def create_obj_from_json(js):
    if isinstance(js, dict):
        rtn_dict = {}
        for key, values in js.items():
            if need_import(key):
                assert values is None or isinstance(values, dict), f"The value of imported object {key} must be dict or None"
                assert len(js) == 1, f"{js} contains imported object {key}, cannot contain other keys"
                key = key[1:-1]
                cls = my_import(key)
                if "__init__" in values:
                    assert isinstance(values, dict), f"__init__ must be provided as dict for class {key}"
                    init_params = create_obj_from_json(values['__init__'])
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js, str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else:
        return js


def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls


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

parser = argparse.ArgumentParser(description='Algorithm Evaluation Program')
parser.add_argument('-f', dest='argFile', type=str, required=True, default=None,
                    help='Specify experiment configuration via YAML file.')
parser.add_argument('-w', dest='over_write', type=bool,
                    help='Force overwrite existing results')
parser.add_argument('-o', dest='out_path', type=str, required=False,
                    help='Specify output directory')
parser.add_argument('-s', dest='scenario', type=myloads, required=False,
                    help='Select scenario class')
parser.add_argument('-p', dest='protocol', type=myloads, required=False,
                    help='Evaluation protocol')
parser.add_argument('-m', dest='metrics', type=str, required=False,
                    help='Metric list, comma-separated')
parser.add_argument('-a', dest='alg', type=str, required=False,
                    help='Algorithm to evaluate')
parser.add_argument('-d', dest='data_path', type=str, required=False,
                    help='Dataset path or archive')
parser.add_argument('-r', dest='params', type=myloads, required=False,
                    help='Algorithm parameters in JSON format')


def update_parameters(param: dict, to_update: dict) -> dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k], dict):
                    param[k].update(to_update[k])
                else:
                    param[k] = to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param


def str_obj_js(mystr):
    if ':' in mystr:
        return myloads(mystr)
    else:
        return {mystr: {}}


def enclose_class_name(value):
    if isinstance(value, dict):
        assert len(value) == 1, "Only one class is allowed"
        for k, v in value.items():
            if k[0] == k[-1] == "_":
                return {k: v}
            else:
                return {f"_{k}_": v}
    elif isinstance(value, str):
        if value[0] == value[-1] == "_":
            return value
        else:
            return f"_{value}_"
    else:
        return value


def parse_objects(filedict):
    algorithm = create_obj_from_json(enclose_class_name({filedict['algorithm']: filedict['algorithm_parameters']}))
    protocol = create_obj_from_json(enclose_class_name(filedict['protocol']))
    scenario = create_obj_from_json(enclose_class_name(filedict['scenario']))
    metrics = []
    for m in filedict['metrics']:
        metrics.append(create_obj_from_json(enclose_class_name(m)))
    return scenario, protocol, algorithm, metrics


if __name__ == '__main__':
    args = parser.parse_args()

    filedict = {}
    if args.argFile is not None:
        filelist = args.argFile.split(',')
        for fname in filelist:
            with open(fname.strip(), 'rb') as infile:
                fd = yaml.safe_load(infile)
                update_parameters(filedict, fd)

    if 'metrics' in filedict and isinstance(filedict['metrics'], str):
        filedict['metrics'] = [filedict['metrics']]

    m = None if args.metrics is None else args.metrics.split(',')
    arg_dict = {
        "algorithm": args.alg,
        "algorithm_parameters": args.params,
        "protocol": args.protocol,
        "data_path": args.data_path,
        "out_path": args.out_path,
        "over_write": args.over_write,
        "metrics": m
    }

    update_parameters(filedict, arg_dict)

    scenario, protocol, algorithm, metrics = parse_objects(filedict)

    data_path = filedict['data_path']
    out_path = filedict['out_path']
    over_write = filedict['over_write']

    assert data_path, "Dataset path must be specified"
    if not out_path:
        out_path = 'log'

    set_seed(12)

    results = scenario.run(algorithm, protocol, metrics, data_path, out_path, over_write)
    print("Final Results:", results, filedict)
