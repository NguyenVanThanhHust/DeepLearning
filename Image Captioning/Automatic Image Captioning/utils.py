import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def load_file(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)

def get_image_name(text_description_file):
    img_name_list = list()
    with open(text_description_file, 'r') as f:
        lines = f.read()
        for line in lines.split('\n'):
            content = line.split()
            img_name = content[0]
            img_name_list.append(img_name)
    return set(img_name_list)

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the
    # inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode_img(model, image_path):
    img = preprocess(image_path)
    feature_vector = model.predict(img)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector


def save_encoded_image(encoding, filename):
    with open(filename, "wb") as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)


def get_idx_word_correspondence(vocab):
    idx_to_word = {}
    word_to_idx = {}

    ix = 1
    for w in vocab:
        word_to_idx[w] = ix
        idx_to_word[ix] = w
        ix += 1
        
    return word_to_idx, idx_to_word
    

def get_max_length_caption(descriptions):
    # descriptions is dictionary so we can loop 
    max_length = 0
    for key in descriptions.keys():
        desc = descriptions[key]
        for each_desc in desc:
            if len(each_desc) > max_length:
                max_length = len(each_desc)
    return max_length
    

def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

