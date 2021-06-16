import pickle
import numpy as np
from tensorflow import keras


def set_bits(bits):
    if bits == 32:
        keras.backend.set_floatx('float32')
        return np.float32
    elif bits == 64:
        keras.backend.set_floatx('float64')
        return np.float64
    else:
        raise ValueError()


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)
