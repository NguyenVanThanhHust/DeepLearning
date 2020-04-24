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

def get_description_from_text(doc):
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()

        # first token is image id, the rest are descriptions
        try:
            img_id, img_desc = tokens[0], tokens[1:]

            # extract filename from image id
            img_id = img_id.split('.')[0]

            # convert descriptions tokens back to string
            img_desc = ' '.join(img_desc)
            if img_id not in descriptions:
                descriptions[img_id] = list()
            descriptions[img_id].append(img_desc)
        except:
            print("This token is empty. \n")
    return descriptions

# Clean data
def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)


def create_vocabulary(descriptions):
    vocabulary = set()
    for key in descriptions.keys():
        [vocabulary.update(d.split()) for d in descriptions[key]]
    print('Original Vocabulary Size: %d' % len(vocabulary))
    return vocabulary


def get_unique_word(descriptions):
    all_captions = []
    for key, val in descriptions.items():
        for cap in val:
            all_captions.append(cap)

    # Consider only words which occur at least 10 times in the corpus
    word_count_threshold = 5
    word_counts = {}
    nsents = 0
    for sent in all_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    print('preprocessed words %d ' % len(vocab))
    return vocab


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