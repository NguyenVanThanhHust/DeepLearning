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

from utils import load_file, save_descriptions, get_image_name, preprocess, \
                   encode_img
from text_processing import get_description_from_text, clean_descriptions, \
                            create_vocabulary, get_unique_word, \
                            get_idx_word_correspondence, get_max_length_caption

# Sample
doc = load_file("../../../Dataset/Flickr8k/Flickr8k_text/Flickr8k.token.txt")
print(doc[:100])

dataset_descriptions = get_description_from_text(doc)
dataset_descriptions['1016887272_03199f49c4']

clean_descriptions(dataset_descriptions)

dataset_voc = create_vocabulary(dataset_descriptions)

get_unique_word(dataset_descriptions)

# get train dataset
train_txt = "../../../Dataset/Flickr8k/Flickr8k_text/Flickr8k.token.txt"
train_document = load_file(train_txt)
train_desc = get_description_from_text(train_document)

save_descriptions(train_desc, 'train_descriptions.txt')
save_descriptions(dataset_descriptions, 'dataset_descriptions.txt')

train_img_file = get_image_name('train_descriptions.txt')

train_img_file

# build model to get image feature
model = InceptionV3(weights='imagenet')
# Remove the last layer (output softmax layer) from the inception v3
model_new = Model(model.input, model.layers[-2].output)


base_path = "../../../Dataset/Flickr8k/Flicker8k_Dataset"
encoding_train = {}
for img_name in train_img_file:
    img_path = os.path.join(base_path, img_name + ".jpg")
    encoding_train[img_name] = encode_img(model_new, img_path)

def save_encoded_image(encoding, filename):
    with open(filename, "wb") as encoded_pickle:
        pickle.dump(encoding, encoded_pickle)

save_encoded_image(encoding_train, "./encodeD_train_image")

train_vocab = get_unique_word(train_descriptions)
total_vocab = get_unique_word(dataset_descriptions)

train_word_to_idx, train_idx_to_word = get_idx_word_correspondence(train_vocab)
train_vocab_size = len(train_idx_to_word) + 1

train_max_length = get_max_length_caption(train_descriptions)

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

# Embedding with GLOVE 
glove_dir = "../../../Dataset/glove.6B/glove.6B.200d.txt"
with open(glove_dir, encoding="utf-8") as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
# create embedding matrix
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in train_word_to_idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
print("embedding_matrix.shape: ", embedding_matrix.shape)

# define model
input_feature_img = Input(shape=(2048, ))
fe1 = Dropout(0.5)(input_feature_img)
fe2 = Dense(256, activation = 'relu')(fe1)

input_feature_word = Input(shapes=(max_length, ))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

# decoder (feed forward) model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

# merge the two input models
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

print("Model structure")
model.summary()

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')

epochs = 10
number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_' + str(i) + '.h5')
    
model.optimizer.lr = 0.0001
epochs = 10
number_pics_per_bath = 6
steps = len(train_descriptions)//number_pics_per_bath

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('./model_weights/model_lr_' + str(i) + '.h5')
    

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final