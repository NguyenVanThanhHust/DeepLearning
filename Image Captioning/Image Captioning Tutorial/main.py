import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os 
from PIL import Image
from cached import cached

from keras import backend as K 
from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import load_image, process_images_train, process_images_val, load_records, _load_records
from cache import cache
from data import TokenizerWrap

image_model = VGG16(include_top = True, weights='./model_weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
image_model.summary()

transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input, 
                            outputs = transfer_layer.output)
                            
img_size = K.int_shape(image_model.input)[1:3]
print("image size: ", img_size)

transfer_values_size = K.int_shape(transfer_layer.output)[1]
print("transfer values size: ", transfer_values_size)

train_data_dir = '../../../Dataset/COCO/val2014'
val_data_dir = '../../../Dataset/COCO/val2014'

_, filenames_train, captions_train = load_records(image_folder='../../../Dataset/COCO/val2014', 
                                        label_file='../../../Dataset/COCO/annotations/captions_val2014.json')
_, filenames_val, captions_val = load_records(image_folder='../../../Dataset/COCO/val2014',
                                        label_file='../../../Dataset/COCO/annotations/captions_val2014.json')

num_images_train = len(filenames_train)

transfer_values_train = process_images_train(cache_path='./model_weight',
                                               model=image_model_transfer, 
                                            train_data_dir=train_data_dir)
print("dtype:", transfer_values_train.dtype)
print("shape:", transfer_values_train.shape)
transfer_values_val = process_images_val(cache_path='./model_weight', 
                                            model=image_model_transfer, 
                                        val_data_dir=val_data_dir)
print("dtype:", transfer_values_val.dtype)
print("shape:", transfer_values_val.shape)

mark_start = 'start_token'
mark_end = 'end_token'

def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    
    return captions_marked
    
captions_train_marked = mark_captions(captions_train)

def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    
    return captions_list
    
captions_train_flat = flatten(captions_train_marked)

num_words = 10000
        
tokenizer = TokenizerWrap(texts=captions_train_flat,
                          num_words=num_words)
# convert all the caption from training set to sequences of integer tokens
tokens_train = tokenizer.captions_to_tokens(captions_train_marked)
def get_random_caption_tokens(idx):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption,
    and return a list of all these token-sequences.
    """
    
    # Initialize an empty list for the results.
    result = []

    # For each of the indices.
    for i in idx:
        # The index i points to an image in the training-set.
        # Each image in the training-set has at least 5 captions
        # which have been converted to tokens in tokens_train.
        # We want to select one of these token-sequences at random.

        # Get a random index for a token-sequence.
        j = np.random.choice(len(tokens_train[i]))

        # Get the j'th token-sequence for image i.
        tokens = tokens_train[i][j]

        # Add this token-sequence to the list of results.
        result.append(tokens)

    return result
    
def batch_generator(num_images_train, batch_size):
    """
    Generator function for creating random batces of training data
    It selects the data completely randomly for each batch_generator
    """
    while True:
        # create a list of random indices for images in the training set
        idx = np.random.randint(num_img_train, size=batch_size)
        
        # get pre-computed transfer values
        transfer_value = transfer_values_train[idx]
        
        # for each of randomly chosen images, 
        # select 1 of 5 captions describing contents of images
        tokens = get_random_caption_tokens(idx)
        
        # max number of tokens
        max_tokens = np.max(num_tokens)
        
        # pad all the token-sequence with zeros so they have the same length
        # and can be input to the neural network as a numpy array
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        # decoder part of the neural netowrk will map the token sequences to themselves
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:1]
        
        x_data =
        {
            'decoder_input': decoder_input_data, 
            'transfer_values_input': transfer_value
        }
        
        y_data = 
        {
            'decoder_output': decoder_output_data
        }
        
        yeild (x_data, y_data)
        
batch_size = 384
generator = batch_generator(num_images_train=num_images_train, batch_size=batch_size)

num_captions_train = [len(captions) for captions in captions_train]
total_num_captions_train = np.sum(num_captions_train)
steps_per_epoch = int(total_num_captions_train / batch_size)

state_size = 512
embedding_size = 128
        
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')