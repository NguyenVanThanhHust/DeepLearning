import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from tensorflow.examples.tutorials.mnist import input_data

dir = os.path.dirname(os.path.realpath(__file__))
mnist = input_data.read_datsets('MMNIST', one_hot = True)