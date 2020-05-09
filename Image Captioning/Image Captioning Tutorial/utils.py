import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os 
import sys
import json
from PIL import Image
from cache import cache

from keras import backend as K 
from keras.models import Model
from keras.layers import Input, Dense, GRU, Embedding
from keras.applications import VGG16
from keras.optimizers import RMSprop

def load_image(img_path, size = None):
    """
    Load image and resize to prefix size
    """
    img = Image.open(img_path)
    
    if size is not None:
        img = img.resize(size=size, resample=Image.LANCZOS)
        
    img = np.array(img)
    
    # scale image-pixels so they fall between 0.0 and 1.0
    img = img /255.0
    
    # convert 2-dim gray sclae array to 3-dim RGB array
    if(len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
    return img
    
def print_progress(count, max_count):
    # Percentage completion
    pct_complete = count / max_count
    
    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
    
def process_images(data_dir, model, img_size=(224,224), transfer_values_size=4096, batch_size=32):
    """
    Process all image in a folder and return feature vector
    
    """
    print(100*"--")
    print("data_dir: ", data_dir)
    list_images = next(os.walk(data_dir))[2]
    num_images = len(list_images)
    
    # Pre-allocate input-batch-array for images.
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)

    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    # Initialize index into the filenames.
    start_index = 0

    # Process batches of image-files.
    while start_index < num_images:
        # Print the percentage progress
        print_progress(count=start_index, max_count=num_images)
        
        end_index = start_index + batch_size
        
         # Ensure end-index is within bounds.
        if end_index > num_images:
            end_index = num_images

        # The last batch may have a different batch-size.
        current_batch_size = end_index - start_index

        # Load all the images in the batch.
        for i, filename in enumerate(list_images[start_index:end_index]):
            # Path for the image-file.
            path = os.path.join(data_dir, filename)

            # Load and resize the image.
            # This returns the image as a numpy-array.
            img = load_image(path, size=img_size)

            # Save the image for later use.
            image_batch[i] = img

        # Use the pre-trained image-model to process the image.
        # Note that the last batch may have a different size,
        # so we only use the relevant images.
        transfer_values_batch = \
            model.predict(image_batch[0:current_batch_size])

        # Save the transfer-values in the pre-allocated array.
        transfer_values[start_index:end_index] = \
            transfer_values_batch[0:current_batch_size]

        # Increase the index for the next loop-iteration.
        start_index = end_index

    # Print newline.
    print()

    return transfer_values
    
def process_images_train(cache_path, model, train_data_dir='../../../Dataset/COCO/val2014'):
    # Path for the cache-file.
    cache_path = os.path.join(cache_path,
                              "transfer_values_train.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            model=model,
                            data_dir=train_data_dir)

    return transfer_values
    
def process_images_val(cache_path, model, val_data_dir='../../../Dataset/COCO/val2014'):
    # Path for the cache-file.
    cache_path = os.path.join(cache_path, "transfer_values_val.pkl")

    # If the cache-file already exists then reload it,
    # otherwise process all images and save their transfer-values
    # to the cache-file so it can be reloaded quickly.
    transfer_values = cache(cache_path=cache_path,
                            fn=process_images,
                            model=model, 
                            data_dir=val_data_dir)

    return transfer_values
    

