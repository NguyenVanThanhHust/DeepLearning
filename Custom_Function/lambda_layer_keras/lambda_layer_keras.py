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
    return tensor + 2


input_layer = tensorflow.keras.layers.Input(shape=(784), name="input_layer")

dense_layer_1 = tensorflow.keras.layers.Dense(units=500, name="dense_layer_1")(input_layer)
activ_layer_1 = tensorflow.keras.layers.ReLU(name="activ_layer_1")(dense_layer_1)

dense_layer_2 = tensorflow.keras.layers.Dense(units=250, name="dense_layer_2")(activ_layer_1)
activ_layer_2 = tensorflow.keras.layers.ReLU(name="relu_layer_2")(dense_layer_2)

dense_layer_3 = tensorflow.keras.layers.Dense(units=20, name="dense_layer_3")(activ_layer_2)

before_lambda_model = tensorflow.keras.models.Model(input_layer, dense_layer_3, name="before_lambda_model")

lambda_layer = tensorflow.keras.layers.Lambda(custom_layer, name="lambda_layer")(dense_layer_3)
after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")

after_lambda_model = tensorflow.keras.models.Model(input_layer, lambda_layer, name="after_lambda_model")
dense_layer_4 = tensorflow.keras.layers.Dense(units=10, name="dense_layer_4")(lambda_layer)
output_layer = tensorflow.keras.layers.Softmax(name="output_layer")(dense_layer_4)

model = tensorflow.keras.models.Model(input_layer, output_layer, name="model")

model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.0005), loss="categorical_crossentropy")
model.summary()

p = model.predict(x_train)

m1 = before_lambda_model.predict(x_train)
m2 = after_lambda_model.predict(x_train)

print(m1[0, :])
print(m2[0, :])
"""
[ 0.72876686 -0.0671312   0.16848794 -0.41023576  0.165883    0.06766549
  0.19400428 -0.4696085  -0.37589905  0.31583726  0.08088666 -0.08404255
 -0.29947743  0.58523345  0.47382885 -0.18972354 -0.1515251  -0.2454896
 -0.07209273  0.2115869 ]
[2.728767  1.9328688 2.168488  1.5897642 2.165883  2.0676656 2.1940043
 1.5303915 1.6241009 2.3158374 2.0808866 1.9159575 1.7005225 2.5852334
 2.4738288 1.8102765 1.8484749 1.7545104 1.9279072 2.211587 ]

"""
# model.fit(x_train, y_train, epochs=20, batch_size=256, validation_data=(x_test, y_test))
