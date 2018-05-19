import tensorflow as tf 
import numpy as np 
import os
import time

from fcnn2d import FCNN_2D, Base
from layers import *
from auxiliary import *
from pre_process import remove_background_3d_with_label

class RESNET(FCNN_2D):

	def build(self, X, y):
		# images = (X_train - tf.reduce_mean(X_train, axis = 0, keep_dims = True)) / tf.reduce_max(X_train, axis = 0, keep_dims = True)

		# with tf.device('gpu:1'):

		with tf.variable_scope('conv_1'):
			conv_1 = conv_layer_res(X, [7, 7, 4, 64], [1, 1, 1, 1], 'conv_1_1')
			bn_1 = tf.nn.relu(bn_layer(conv_1, self.is_training, 'bn_1'))
			max_1 = max_pool_layer(bn_1, [1, 3, 3, 1], [1, 2, 2, 1], name = 'max_1')

		with tf.variable_scope('bottleneck_1'):
			bottleneck_1_1 = bottle_layer(max_1, 64, 64, 128, self.is_training, 'bottle_1')
			bottleneck_1_2 = bottle_layer(bottleneck_1_1, 128, 64, 128, self.is_training, 'bottle_2')
			bottleneck_1_3 = bottle_layer(bottleneck_1_2, 128, 64, 128, self.is_training, 'bottle_3')

		with tf.variable_scope('bottleneck_2'):
			max_2 = max_pool_layer(bottleneck_1_3, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_2')
			bottleneck_2_1 = bottle_layer(max_2, 128, 128, 256, self.is_training, 'bottle_1')
			bottleneck_2_2 = bottle_layer(bottleneck_2_1, 256, 128, 256, self.is_training, 'bottle_2')
			bottleneck_2_3 = bottle_layer(bottleneck_2_2, 256, 128, 256, self.is_training, 'bottle_3')
			bottleneck_2_4 = bottle_layer(bottleneck_2_3, 256, 128, 256, self.is_training, 'bottle_4')

		# with tf.device('gpu:2'):

		with tf.variable_scope('bottleneck_3'):
			max_3 = max_pool_layer(bottleneck_2_4, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_3')
			bottleneck_3_1 = bottle_layer(max_3, 256, 256, 1024, self.is_training, 'bottle_1')
			bottleneck_3_2 = bottle_layer(bottleneck_3_1, 1024, 256, 1024, self.is_training, 'bottle_2')
			bottleneck_3_3 = bottle_layer(bottleneck_3_2, 1024, 256, 1024, self.is_training, 'bottle_3')
			bottleneck_3_4 = bottle_layer(bottleneck_3_3, 1024, 256, 1024, self.is_training, 'bottle_4')
			bottleneck_3_5 = bottle_layer(bottleneck_3_4, 1024, 256, 1024, self.is_training, 'bottle_5')
			bottleneck_3_6 = bottle_layer(bottleneck_3_5, 1024, 256, 1024, self.is_training, 'bottle_6')

		with tf.variable_scope('bottleneck_4'):
			max_4 = max_pool_layer(bottleneck_3_6, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_4')
			bottleneck_4_1 = bottle_layer(max_4, 1024, 512, 2048, self.is_training, 'bottle_1')
			bottleneck_4_2 = bottle_layer(bottleneck_4_1, 2048, 512, 2048, self.is_training, 'bottle_2')
			bottleneck_4_3 = bottle_layer(bottleneck_4_2, 2048, 512, 2048, self.is_training, 'bottle_3')

		max_5 = max_pool_layer(bottleneck_4_3, [1, 2, 2, 1], [1, 2, 2, 1], name = 'max_5')

		fc_1 = tf.nn.dropout(conv_layer_res(max_5, [1, 1, 2048, 2048], [1, 1, 1, 1], 'fc_1'), self.dropout)
		fc_2 = conv_layer_res(fc_1, [1, 1, 2048, self.num_classes], [1, 1, 1, 1], 'fc_2')

		# Now we start upsampling and skip layer connections.
		img_shape = tf.shape(X)
		dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])
		upsample_1 = upsample_layer(fc_2, dconv3_shape, self.num_classes, 'upsample_1', 32)

		skip_1 = skip_layer_connection(max_4, 'skip_1', 1024, stddev=0.00001)
		upsample_2 = upsample_layer(skip_1, dconv3_shape, self.num_classes, 'upsample_2', 16)

		skip_2 = skip_layer_connection(max_3, 'skip_2', 256, stddev=0.0001)
		upsample_3 = up_layer(skip_2, dconv3_shape, 5, 5, 8, 'upsample_3')


		logits = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))
		self.result = tf.argmax(logits, axis = 3)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix



if __name__ == '__main__':
	print 'loading from HGG_train.npz...'
	f = np.load(Base + '/HGG_train2.npz')
	X = f['X']
	y = f['y']

	print X.shape, y.shape		

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	net = RESNET(input_shape = (240, 240, 4), num_classes = 5)
	net.multi_gpu_train(X, y, model_name = 'model_resnet_1', train_mode = 1, num_gpu = 1, 
     batch_size = 32, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e10, thre = 1.0)
 
	# else:
	# 	exit(0)
