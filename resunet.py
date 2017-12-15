import tensorflow as tf 
import numpy as np 
import time

from fcnn2d import FCNN_2D, Base
from resnet import RESNET
from layers import *
from auxiliary import *
from pre_process import *


class RES_UNET(RESNET):

	def build(self, X, y):

		'''
		resnet + unet
		'''

		stack_1 = stack_layer(X, 4, 64, 64, self.is_training, 'stack_1')
		max_1 = max_pool_layer(stack_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')

		stack_2 = stack_layer(max_1, 64, 128, 128, self.is_training, 'stack_2')
		max_2 = max_pool_layer(stack_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		stack_3 = stack_layer(max_2, 128, 256, 256, self.is_training, 'stack_3')
		max_3 = max_pool_layer(stack_3, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		stack_4 = stack_layer(max_3, 256, 512, 512, self.is_training, 'stack_4')
		max_4 = max_pool_layer(stack_4, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		stack_5 = stack_layer(max_4, 512, 1024, 1024, self.is_training, 'stack_5')

		# upsample layers No.1 for comp binary classification

		upsample_6 = up_layer(stack_5, tf.shape(stack_4), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, stack_4], axis = 3)
		stack_6 = bottle_layer(concat_6, 1024, 1024, 512, self.is_training, 'stack_6')

		upsample_7 = up_layer(stack_6, tf.shape(stack_3), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, stack_3], axis = 3)
		stack_7 = bottle_layer(concat_7, 512, 512, 256, self.is_training, 'stack_7')

		upsample_8 = up_layer(stack_7, tf.shape(stack_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, stack_2], axis = 3)
		stack_8 = bottle_layer(concat_8, 256, 256, 128, self.is_training, 'stack_8')

		upsample_9 = up_layer(stack_8, tf.shape(stack_1), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, stack_1], axis = 3)
		stack_9 = bottle_layer(concat_9, 128, 128, 64, self.is_training, 'stack_9')

		conv_10 = conv_layer_res(stack_9, [1, 1, 64, 5], [1, 1, 1, 1], 'conv_10')	


		logits = conv_10
		self.result = tf.argmax(logits, axis = 3)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix

if __name__ == '__main__':
	print 'loading from LGG_train.npz...'
	f = np.load(Base + '/LGG_train.npz')
	X = f['X']
	y = f['y']

	print X.shape, y.shape		

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	net = RES_UNET(input_shape = (240, 240, 4), num_classes = 5)
	net.multi_gpu_train(X, y, model_name = 'model_resunet_3', train_mode = 0,
	 batch_size = 8, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 2e5)
	# else:
	# 	exit(0)
