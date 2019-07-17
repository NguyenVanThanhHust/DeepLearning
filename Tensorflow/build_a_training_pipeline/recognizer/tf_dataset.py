import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .pair_generator import PairGenerator
from .model import Inputs, Model

class Dataset(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    label = 'same person'
	
    def __init__(self, generator = PairGenerator()):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, pair_gen: PairGenerator):
        batch_size = 10
        prefetch_batch_buffer = 5
        # set up a plent to use generator : define type for each of element in generator
        dataset = tf.data.Dataset.from_generator(pair_gen.get_next_par, 
                                                output_type  = {PairGenerator.person1 : tf.string,
                                                                PairGenerator.person2 : tf.string,
                                                                PairGenerator.label : tf.bool})
																
		# setup all task necessary to get from generator input(file names) 
        dataset = dataset.map(self._read_image_and_size)
		# batch images into bundles with consistent number of element
        dataset = dataset.batch(batch_size)
		# let tensorflow do the book keeping involved in setting up a queue such that the data piple line 
		# continue to read and enqueue data 
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()
		
        return Inputs(element[self.img1_resized],
                      element[self.img2_resized],
                      element[PairGenerator, label])
	
    def read_image_and_resize(self, pair_element):
        target_size = [128, 128]
		
		#read the file from disk
        img1_file = tf.read_file(pair_element[PairGenerator.person1])
        img2_file = tf.read_file(pair_element[PairGenerator.person2])
        img1 = tf.image.decode(img1_file)
        img2 = tf.image.decode(img2_file)
		
		# led the tensorflow know that the loaded images have unknown dimensions and 3 colors channels
        img1.set_shape([None, None, 3])
        img2.set_shape([None, None, 3])
		
		# resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        img2_resized = tf.image.resize_images(img2, target_size)
		
        pair_element[self.img1_resized] = img1_resized
        pair_element[self.img2_resized] = img2_resized
        pair_element[self.label] = tf.cast(pair_element[PairGenerator.label], tf.float32)

        return pair_element