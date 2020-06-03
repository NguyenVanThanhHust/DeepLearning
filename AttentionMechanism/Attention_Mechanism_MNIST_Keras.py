
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout,  Conv2D, Input, Lambda, Flatten, TimeDistributed
from tensorflow.keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.engine.topology import Layer
from tensorflow.keras.callbacks import TensorBoard

def MultiHeadAttention(l=8*8, d=512, dv=64, dim_out=512, nv=8):
    """
    Args:
        l: number of blocks in feature map
        d: dimension of the block
        dv: dimension of linear space to be projected
        nv: number of project for each block
    """
    value_vector_1 = Input(shape=(l,d))
    query_vector_1 = Input(shape=(l,d))
    key_vector_1 = Input(shape=(l,d))

    value_vector_2 = Dense(dv*nv, activation="relu")(value_vector_1)
    query_vector_2 = Dense(dv*nv, activation="relu")(query_vector_1)
    key_vector_2 = Dense(dv*nv, activation="relu")(key_vector_1)

    value = Reshape([l, nv, dv])(value_vector_2)
    query = Reshape([l, nv, dv])(query_vector_2)
    key = Reshape([l, nv, dv])(key_vector_2)

    attention = tf.einsum('baik,baij->bakj',query, key)/np.sqrt(dv)
    attention = Lambda(lambda x: K.softmax(x), output_shape=(l, nv, nv))(attention)
    output = tf.einsum('bajk,baik->baji',attention, value)
    output = Reshape([l, d])(output)

    output = Add()([output, query_vector_1])
    output = Dense(dim_out, activation='relu')(output)

    return Model(inputs=[query_vector_1, key_vector_1, value_vector_1], outputs=output)

class NormLayer(Layer):
    
    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight varialbe for this layer
        self.a = self.add_weight(name='kernel', 
                                shape=(1, input_shape[-1]),
                                initializer='ones',
                                trainable=True)
        self.b = self.add_weight(name='kernel', 
                                shape=(1, input_shape[-1]),
                                initializer='ones',
                                trainable=True)
        super(NormLayer, self).build(input_shape)

    def call(self, x):
        eps = 0.00001
        mu = K.mean(x, keepdims=True, axis=-1)
        signma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x-mu)/(signma+eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == '__main__':
    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    X_train = X_train.reshape(60000, 28,28,1)
    X_test = X_test.reshape(10000, 28,28,1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print("Training matrix shape", X_train.shape)
    print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    inp = Input(shape = (28,28,1))
    x = Conv2D(32,(2,2),activation='relu', padding='same')(inp)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64,(2,2),activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = Conv2D(64*3,(2,2),activation='relu')(x)

    x = Reshape([6*6,64*3])(x)    
    att = MultiHeadAttention(l=6*6, d=64*3 , dv=8*3, dim_out=32, nv = 8 )
    x = att([x,x,x])
    x = Reshape([6,6,32])(x)   
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='relu')(x)

    model = Model(inputs=inp, outputs=x)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    tbCallBack = TensorBoard(log_dir='./Graph/mhatt1', histogram_freq=0, write_graph=True, write_images=True)
    
    model.fit(X_train, Y_train,
              batch_size=128, 
              epochs=100,
              verbose=1,          
              validation_data=(X_test, Y_test),
              callbacks=[tbCallBack]
             )
