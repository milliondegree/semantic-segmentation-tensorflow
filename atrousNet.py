import tensorflow as tf 
import numpy as np 
import time

from fcnn2d import FCNN_2D, Base
from resnet import RESNET
from resunet import RES_UNET
from layers import *
from auxiliary import *
from pre_process import *

class atrousNet(RES_UNET):

	def multi_binary_build(self, X, y):
		stack_1 = stack_layer(X, 4, 64, 64, self.is_training, 'stack_1')
		max_1 = max_pool_layer(stack_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')
	
		stack_2 = stack_layer(max_1, 64, 128, 128, self.is_training, 'stack_2')
		max_2 = max_pool_layer(stack_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		stack_3 = stack_layer(max_2, 128, 256, 256, self.is_training, 'stack_3')
		max_3 = max_pool_layer(stack_3, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		stack_4 = stack_layer(max_3, 256, 512, 512, self.is_training, 'stack_4')
		max_4 = max_pool_layer(stack_4, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		# stack_5 = stack_layer(max_4, 512, 1024, 1024, self.is_training, 'stack_5')
		bn_1 = bn_layer(max_4, self.is_training, 'bn_1')
		atrous_1 = atrous_conv_layer(bn_1, [1, 1, 512, 512], 1, 'atrous_1', if_relu = True)
		atrous_2 = atrous_conv_layer(bn_1, [3, 3, 512, 512], 2, 'atrous_2', if_relu = True)
		atrous_3 = atrous_conv_layer(bn_1, [3, 3, 512, 512], 4, 'atrous_3', if_relu = True)
		atrous_4 = atrous_conv_layer(bn_1, [3, 3, 512, 512], 6, 'atrous_4', if_relu = True)
		atrous_5 = atrous_conv_layer(bn_1, [3, 3, 512, 512], 12, 'atrous_5', if_relu = True)

		concat_1 = tf.concat([atrous_1, atrous_2, atrous_3, atrous_4, atrous_5], axis = 3)
		bn_2 = bn_layer(concat_1, self.is_training, 'bn_2')
		conv_1 = conv_layer_res(bn_2, [1, 1, 2560, 1024], [1, 1, 1, 1], 'conv_1', if_relu = True)

		upsample_6 = up_layer(conv_1, tf.shape(stack_4), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, stack_4], axis = 3)
		stack_6 = stack_layer(concat_6, 1024, 1024, 512, self.is_training, 'stack_6')

		upsample_7 = up_layer(stack_6, tf.shape(stack_3), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, stack_3], axis = 3)
		stack_7 = stack_layer(concat_7, 512, 512, 256, self.is_training, 'stack_7')

		upsample_8 = up_layer(stack_7, tf.shape(stack_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, stack_2], axis = 3)
		stack_8 = stack_layer(concat_8, 256, 256, 128, self.is_training, 'stack_8')

		upsample_9 = up_layer(stack_8, tf.shape(stack_1), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, stack_1], axis = 3)
		stack_9 = stack_layer(concat_9, 128, 128, 64, self.is_training, 'stack_9')

		logits = conv_layer_res(stack_9, [1, 1, 64, self.num_classes], [1, 1, 1, 1], 'logits')
		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		prob = tf.nn.softmax(logits = reshaped_logits)
		prob_0 = tf.reduce_sum(prob[:, 1:5], axis = 1, keep_dims = True)
		prob_1 = tf.reduce_sum(prob[:, 3:5], axis = 1, keep_dims = True) + tf.reshape(prob[:, 1], [-1, 1])
		prob_2 = tf.reshape(prob[:, 4], [-1, 1])

		return [prob_0, prob_1, prob_2]



if __name__ == '__main__':
	print 'loading from HGG_train.npz...'
	f = np.load(Base + '/HGG_train.npz')
	X = f['X']
	y = f['y']

	print X.shape, y.shape		

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	net = atrousNet(input_shape = (240, 240, 4), num_classes = 5)
	# net.multi_gpu_train(X, y, model_name = 'model_resunet_5', train_mode = 1,
	#  batch_size = 8, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 2e5)

	net.multi_dice_train(X, y, model_name = 'model_HGG_atrous', train_mode = 1, num_gpu = 3,
	 batch_size = 8, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e6)
	# else:
	# 	exit(0)