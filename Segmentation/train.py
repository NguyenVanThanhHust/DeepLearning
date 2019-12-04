import random
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras_fcn.models import FCN_VGG16
import logging, os
from keras import backend as K

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATA_PATH = 'dataset'
IMG_PATH = os.path.join(DATA_PATH, 'img')
GT_PATH = os.path.join(DATA_PATH, 'gt')

all_frames = next(os.walk(IMG_PATH))[2]
all_masks = next(os.walk(GT_PATH))[2]

all_frames = sorted(all_frames)
all_masks = sorted(all_masks)

train_split = int(0.7*len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

train_frames_names = [x[:-4] for x in train_frames]
val_frames_names = [x[:-4] for x in val_frames]
test_frames_names = [x[:-4] for x in test_frames]

train_masks = [f for f in all_masks if f[:-4] in train_frames_names]
val_masks = [f for f in all_masks if f[:-4] in val_frames_names]
test_masks = [f for f in all_masks if f[:-4] in test_frames_names]

def add_frames(dir_name, image):
#   img = Image.open(IMG_PATH+image)
  img = Image.open(os.path.join(IMG_PATH,image))
  img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

def add_masks(dir_name, image):
  img = Image.open(os.path.join(GT_PATH,image))
  img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                 (test_frames, 'test_frames')]

mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                (test_masks, 'test_masks')]

for each_pair in frame_folders:
    folder_name = each_pair[1]
    if not os.path.isdir(os.path.join(DATA_PATH, folder_name)):
        os.mkdir(os.path.join(DATA_PATH, folder_name))
        
for each_pair in mask_folders:
    folder_name = each_pair[1]
    if not os.path.isdir(os.path.join(DATA_PATH, folder_name)):
        os.mkdir(os.path.join(DATA_PATH, folder_name))

# Add frames
for folder in frame_folders:
  array = folder[0]
  name = [folder[1]] * len(array)
  list(map(add_frames, name, array))
         
# Add masks
for folder in mask_folders:
  array = folder[0]
  name = [folder[1]] * len(array)
  list(map(add_masks, name, array))

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)
val_datagen = ImageDataGenerator(
    rescale = 1./255
)

train_image_generator = train_datagen.flow_from_directory(
    'dataset/train_frames',
    batch_size = 4
)
train_mask_generator = train_datagen.flow_from_directory(
    'dataset/train_masks',
    batch_size = 4
)

val_image_generator = val_datagen.flow_from_directory(
    'dataset/val_frames',
    batch_size = 4
)
val_mask_generator = val_datagen.flow_from_directory(
    'dataset/val_masks',
    batch_size = 4
)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

# in case we want to use custom data generator
def custom_data_generator(img_folder, mask_folder, batch_size):
    count = 0
    n = os.listdir(img_folder)
    random.shuffle(n)
    while True:
        img = np.zeros((batch_size, 512, 512, 3)).astype('float')
        mask = np.zeros((batch_size, 512, 512, 3)).astype('float')
        for i in range(count, count + batch_size):
            train_img = cv2.imread(os.path.join(img_folder, n[i]))/255.0
            train_img = cv2.resize(train_img, (512, 512))
            img[count - i] = train_img

            train_mask = cv2.imread(os.path.join(mask_folder, n[i]))/255.0
            train_mask = cv2.resize(train_mask, (512, 512))
            mask[count - i] = train_mask

        count += batch_size
        if (count + batch_size > len(n)):
            count = 0
            random.shuffle(n)
        yield img, mask

NO_OF_TRAINING_IMAGES = len(os.listdir('dataset/train_frames/'))
NO_OF_VAL_IMAGES = len(os.listdir('dataset/val_frames/'))

NO_OF_EPOCHS = 30

BATCH_SIZE = 4

weight_path = './model/'
if not os.path.isdir(weight_path):
    os.mkdir(weight_path)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

img_folder = os.path.join(DATA_PATH, 'train_frames')
mask_folder = os.path.join(DATA_PATH, 'mask_frames') 
custom_data_generator = custom_data_generator(img_folder, mask_folder, BATCH_SIZE)


FCN_model = FCN_VGG16(input_shape=(500, 500, 3), classes=3,  
                      weights='imagenet', trainable_encoder=True)
optimizer = Adam(lr = 1e-5, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)

FCN_model.compile(loss = jaccard_distance_loss,   
                optimizer = optimizer, 
                metrics=['mae', 'acc'])

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

callbacks_list = [checkpoint, csv_logger, earlystopping]

results = FCN_model.fit_generator(custom_data_generator, epochs=NO_OF_EPOCHS, 
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_generator, 
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE), 
                          callbacks=callbacks_list)
FCN_model.save('Model.h5')