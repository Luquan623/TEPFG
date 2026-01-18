import copy
import io
import types

import numpy as np
import yaml
import json


class ModelAbstract:
    def __init__(self):
        self.checkpoint_path = None
        self.tensorboard_path = None
        self.cache_dir = None
        self.resume_state = True

    def train(self, ds, valid_ds=None, test_ds=None, valid_funcs=None, cb_progress=lambda x: None):
        return None

    def predict(self, ds, cb_progress=lambda x: None):
        return None

    def save(self, filepath: str = None):
        return None

    def load(self, filepath: str = None):
        return None

    def opt_one_batch(self, batch, targets=None) -> dict:
        pass

    def eval_data(self, dataloader, metric, inbatch) -> float:
        pass

    def class_name(self):
        return str(self.__class__)[8:-2].split('.')[-1].lower()

    def __str__(self):
        parameters_dic = copy.deepcopy(self.__dict__)
        parameters = get_parameters_js(parameters_dic)
        return dict_to_yamlstr({self.class_name(): parameters})

    def __getitem__(self, key):
        if isinstance(key, str) and hasattr(self, key):
            return getattr(self, key)
        return None

    def __setitem__(self, key, value):
        if isinstance(key, str):
            setattr(self, key, value)
        return None


def dict_to_yamlstr(d: dict) -> str:
    with io.StringIO() as mio:
        json.dump(d, mio)
        mio.seek(0)
        if hasattr(yaml, "full_load"):
            y = yaml.full_load(mio)
        else:
            y = yaml.load(mio)
        return yaml.dump(y)


def get_parameters_js(js) -> dict:
    if isinstance(js, dict):
        return {
            k: get_parameters_js(v)
            for k, v in js.items()
            if not isinstance(v, types.BuiltinMethodType)
        }
    elif isinstance(js, (float, int, str)):
        return js
    elif isinstance(js, (list, set, tuple)):
        return [get_parameters_js(x) for x in js]
    elif js is None:
        return None
    else:
        return {get_full_class_name(js): get_parameters_js(js.__dict__)}


def get_full_class_name(c) -> str:
    s = str(type(c))
    return s[8:-2]
