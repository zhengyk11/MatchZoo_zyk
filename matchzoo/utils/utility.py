# -*- coding=utf-8 -*-
import os
import sys
import traceback
from keras.models import Model

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))

def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)

def import_module(import_str):
    __import__(import_str)
    return sys.modules[import_str]


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, model_config['model_path'])

        model = import_object(model_config['model_py'], model_config)
        mo = model.build()
    return mo
