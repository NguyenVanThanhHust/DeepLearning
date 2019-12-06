import os, copy
import tensorflow as tf

class BasicAgent(object):
    # to build your model, pass a "configuration" which is a dictionary
    def __init__(self, config):
        # just keep the best hyper parameter found in side the model itself
		# This is a mechanism to load the best hyper parameter and override the configuration
		if config['best']:
		    config.update(self.get_best_config(config['env_name']))
		
		#make a 'deepcopy' of the configuration before using it
		# to avoid any potential mutation when iteratte asynchonrously over configurations
		sef.config = copy.deepcopy(config)
		