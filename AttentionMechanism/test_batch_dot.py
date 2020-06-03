from keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras import backend as K

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
import tensorflow as tf
from keras.callbacks import TensorBoard

q = K.ones(shape=(36,8,24))
k = K.ones(shape=(36,8,24))
result = K.batch_dot(q,k,axes=[1,1])
print(result)
print(K.eval(result))

# q = K.ones(shape=(2, 4))
# k = K.ones(shape=(2, 4))
# print(K.batch_dot(q,k,axes=[1,1]))