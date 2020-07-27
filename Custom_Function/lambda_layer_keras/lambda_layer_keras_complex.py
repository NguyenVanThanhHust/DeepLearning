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
import tensorflow 
import numpy

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

x_train = x_train.astype(numpy.float64) / 255.0
x_test = x_test.astype(numpy.float64) / 255.0

x_train = x_train.reshape((x_train.shape[0], numpy.prod(x_train.shape[1:])))
x_test = x_test.reshape((x_test.shape[0], numpy.prod(x_test.shape[1:])))
y_test = tensorflow.keras.utils.to_categorical(y_test)
y_train = tensorflow.keras.utils.to_categorical(y_train)

print(x_train.shape)
print(y_train.shape)

def custom_layer(tensor):
    tensor1 = tensor[0]
    tensor2 = tensor[1]
    return tensor1 + tensor2


input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
activ_layer_1 = tensorflow.keras.layers.ReLU(name="activ_layer_1")(dense_layer_1)

dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)
activ_layer_3 = tensorflow.keras.layers.ReLU(name="relu_layer_3")(dense_layer_3)

before_lambda_model_1 = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model_1")
before_lambda_model_2 = tensorflow.keras.models.Model(input_layer, activ_layer_3, name="before_lambda_model_2")

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")([dense_layer_3, activ_layer_3])
after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")


dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(lambda_layer)
output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")

model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
model.summary()

m1 = before_lambda_model_1.predict(x_train)
m2 = before_lambda_model_2.predict(x_train)
m3 = after_lambda_model.predict(x_train)

print(m1[0, :])
print(m2[0, :])
print(m3[0, :])
"""
[-0.8513353  -0.4369371   0.3260426  -0.0225438   0.5096403  -0.24065451
 -0.18013945  0.61647296 -0.04329283 -0.03838666 -0.25315332 -0.22907384
 -0.07703415  0.20469122 -0.5868674  -0.09735474 -0.15910766 -0.22670946
  0.5480242   0.05398711]
[0.         0.         0.3260426  0.         0.5096403  0.
 0.         0.61647296 0.         0.         0.         0.
 0.         0.20469122 0.         0.         0.         0.
 0.5480242  0.05398711]
[-0.8513353  -0.4369371   0.6520852  -0.0225438   1.0192806  -0.24065451
 -0.18013945  1.2329459  -0.04329283 -0.03838666 -0.25315332 -0.22907384
 -0.07703415  0.40938243 -0.5868674  -0.09735474 -0.15910766 -0.22670946
  1.0960484   0.10797422]

"""
# model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
