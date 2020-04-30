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

from utils import load_image, process_images_train, process_images_val
from cache import cache
from data import TokenizerWrap, load_records, mark_captions

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
   
captions_train_marked = mark_captions(captions_train)
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

# Create the recurrent network
# state size of 3 GRU layers
state_size = 512

# embedding layers convert interger tokens intor vectors if this length
embedding_size = 128

#inputs transfer values to the decoder
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')
                              
decoder_transfer_map = Dense(state_size, activation='tanh', 
                            name = 'decoder_transfer_map')
                            
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')

decoder_gru1 = GRU(state_size, name='decoder_gru1',
                   return_sequences=True)
decoder_gru2 = GRU(state_size, name='decoder_gru2',
                   return_sequences=True)
decoder_gru3 = GRU(state_size, name='decoder_gru3',
                   return_sequences=True)
                   
decoder_dense = Dense(num_words,
                      activation='softmax',
                      name='decoder_output')
                      
      
def connect_decoder(transfer_values):
    # Map the transfer values to with dimensionality of internal state of GRU layers
    initial_state = decoder_transfer_map(transfer_values)
    
    net = decoder_input
    
    # Connect the embedding-layer.
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)

    # Connect the final dense layer that converts to
    # one-hot encoded arrays.
    decoder_output = decoder_dense(net)
    
    return decoder_output
    
decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])
                      
decoder_model.compile(optimizer=RMSprop(lr=1e-3),
                      loss='sparse_categorical_crossentropy')
                      
path_checkpoint = '22_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      verbose=1,
                                      save_weights_only=True)
                                      
callback_tensorboard = TensorBoard(log_dir='./22_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callbacks = [callback_checkpoint, callback_tensorboard]

decoder_model.fit(x=generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=20,
                  callbacks=callbacks)

def generate_caption(image_path, max_tokens=30):
    """
    Generate a caption for the image in the given path.
    The caption is limited to the given number of tokens (words).
    """

    # Load and resize the image.
    image = load_image(image_path, size=img_size)
    
    # Expand the 3-dim numpy array to 4-dim
    # because the image-model expects a whole batch as input,
    # so we give it a batch with just one image.
    image_batch = np.expand_dims(image, axis=0)

    # Process the image with the pre-trained image-model
    # to get the transfer-values.
    transfer_values = image_model_transfer.predict(image_batch)

    # Pre-allocate the 2-dim array used as input to the decoder.
    # This holds just a single sequence of integer-tokens,
    # but the decoder-model expects a batch of sequences.
    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'transfer_values_input': transfer_values,
            'decoder_input': decoder_input_data
        }

        # Note that we input the entire sequence of tokens
        # to the decoder. This wastes a lot of computation
        # because we are only interested in the last input
        # and output. We could modify the code to return
        # the GRU-states when calling predict() and then
        # feeding these GRU-states as well the next time
        # we call predict(), but it would make the code
        # much more complicated.
        
        # Input this data to the decoder and get the predicted output.
        decoder_output = decoder_model.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        # Note that this is not limited by softmax, but we just
        # need the index of the largest element so it doesn't matter.
        token_onehot = decoder_output[0, count_tokens, :]

        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = tokenizer.token_to_word(token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # This is the sequence of tokens output by the decoder.
    output_tokens = decoder_input_data[0]

    # Plot the image.
    plt.imshow(image)
    plt.show()
    
    # Print the predicted caption.
    print("Predicted caption:")
    print(output_text)
                                   