import tensorflow as tf 
import numpy as np 
import time

from fcnn2d import FCNN_2D, Base
from layers import *
from auxiliary import *
from pre_process import *

class UNET(FCNN_2D):

	def build(self):

		# subsample layers
		img_shape = tf.shape(self.X_train)
		dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])

		conv_1_1 = conv_layer_res(self.X_train, [3, 3, 4, 64], [1, 1, 1, 1], 'conv_1_1', if_relu = True)
		conv_1_2 = conv_layer_res(conv_1_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_1_2', if_relu = True)
		max_1 = max_pool_layer(conv_1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')

		conv_2_1 = conv_layer_res(max_1, [3, 3, 64, 128], [1, 1, 1, 1], 'conv_2_1', if_relu = True)
		conv_2_2 = conv_layer_res(conv_2_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_2_2', if_relu = True)
		max_2 = max_pool_layer(conv_2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		conv_3_1 = conv_layer_res(max_2, [3, 3, 128, 256], [1, 1, 1, 1], 'conv_3_1', if_relu = True)
		conv_3_2 = conv_layer_res(conv_3_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_3_2', if_relu = True)
		max_3 = max_pool_layer(conv_3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		conv_4_1 = conv_layer_res(max_3, [3, 3, 256, 512], [1, 1, 1, 1], 'conv_4_1', if_relu = True)
		conv_4_2 = conv_layer_res(conv_4_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_4_2', if_relu = True)
		max_4 = max_pool_layer(conv_4_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		conv_5_1 = conv_layer_res(max_4, [3, 3, 512, 1024], [1, 1, 1, 1], 'conv_5_1', if_relu = True)
		conv_5_2 = conv_layer_res(conv_5_1, [3, 3, 1024, 1024], [1, 1, 1, 1], 'conv_5_2', if_relu = True)

		# upsample layers No.1 for comp binary classification

		upsample_6 = up_layer(conv_5_2, tf.shape(conv_4_2), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, conv_4_2], axis = 3)
		conv_6_1 = conv_layer_res(concat_6, [3, 3, 1024, 512], [1, 1, 1, 1], 'conv_6_1', if_relu = True)
		conv_6_2 = conv_layer_res(conv_6_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_6_2', if_relu = True)

		upsample_7 = up_layer(conv_6_2, tf.shape(conv_3_2), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, conv_3_2], axis = 3)
		conv_7_1 = conv_layer_res(concat_7, [3, 3, 512, 256], [1, 1, 1, 1], 'conv_7_1', if_relu = True)
		conv_7_2 = conv_layer_res(conv_7_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_7_2', if_relu = True)

		upsample_8 = up_layer(conv_7_2, tf.shape(conv_2_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, conv_2_2], axis = 3)
		conv_8_1 = conv_layer_res(concat_8, [3, 3, 256, 128], [1, 1, 1, 1], 'conv_8_1', if_relu = True)
		conv_8_2 = conv_layer_res(conv_8_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_8_2', if_relu = True)

		upsample_9 = up_layer(conv_8_2, tf.shape(conv_1_2), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, conv_1_2], axis = 3)
		conv_9_1 = conv_layer_res(concat_9, [3, 3, 128, 64], [1, 1, 1, 1], 'conv_9_1', if_relu = True)
		conv_9_2 = conv_layer_res(conv_9_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_9_2', if_relu = True)

		conv_10 = conv_layer_res(conv_9_2, [1, 1, 64, 5], [1, 1, 1, 1], 'conv_10')		

		logits = conv_10
		self.result = tf.argmax(logits, axis = 3)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(self.labels, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)

		loss = tf.reduce_mean(cross_entropy)

		weighted_loss = self._weighted_loss(cross_entropy, reshaped_labels)

		self_weighted_loss, new_cross_entropy = self._self_weighted_loss(cross_entropy, reshaped_labels)

		bootstrapping_loss = self._bootstrapping_loss(cross_entropy, prob)

		return bootstrapping_loss, prob


class WNET:

	def __init__(self, input_shape):
		self.sess = tf.Session()

		self.W, self.H, self.C = input_shape

		self.X_train = tf.placeholder(tf.float32, [None, self.W, self.H, self.C], name = 'X_train')
		self.y_1 = tf.placeholder(tf.uint8, [None, self.W, self.H], name = 'y_1')
		self.y_2 = tf.placeholder(tf.uint8, [None, self.W, self.H], name = 'y_2')
		self.y_3 = tf.placeholder(tf.uint8, [None, self.W, self.H], name = 'y_3')
		self.label_1 = tf.one_hot(indices = self.y_1, depth = 2)
		self.label_2 = tf.one_hot(indices = self.y_2, depth = 2)
		self.label_3 = tf.one_hot(indices = self.y_3, depth = 2)
		
		self.dropout = tf.placeholder(tf.float32, name = 'dropout')
		self.N_worst = tf.placeholder(tf.int32, name = 'N_worst')
		self.thre = tf.placeholder(tf.float32, name = 'threshold')
		self.loss = self.build()

		self.last = time.time()

	def build(self):

		conv_1_1 = conv_layer_res(self.X_train, [3, 3, 4, 64], [1, 1, 1, 1], 'conv_1_1', if_relu = True)
		conv_1_2 = conv_layer_res(conv_1_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_1_2', if_relu = True)
		max_1 = max_pool_layer(conv_1_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')

		conv_2_1 = conv_layer_res(max_1, [3, 3, 64, 128], [1, 1, 1, 1], 'conv_2_1', if_relu = True)
		conv_2_2 = conv_layer_res(conv_2_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_2_2', if_relu = True)
		max_2 = max_pool_layer(conv_2_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		conv_3_1 = conv_layer_res(max_2, [3, 3, 128, 256], [1, 1, 1, 1], 'conv_3_1', if_relu = True)
		conv_3_2 = conv_layer_res(conv_3_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_3_2', if_relu = True)
		max_3 = max_pool_layer(conv_3_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		conv_4_1 = conv_layer_res(max_3, [3, 3, 256, 512], [1, 1, 1, 1], 'conv_4_1', if_relu = True)
		conv_4_2 = conv_layer_res(conv_4_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_4_2', if_relu = True)
		max_4 = max_pool_layer(conv_4_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		conv_5_1 = conv_layer_res(max_4, [3, 3, 512, 1024], [1, 1, 1, 1], 'conv_5_1', if_relu = True)
		conv_5_2 = conv_layer_res(conv_5_1, [3, 3, 1024, 1024], [1, 1, 1, 1], 'conv_5_2', if_relu = True)

		upsample_6 = up_layer(conv_5_2, tf.shape(conv_4_2), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, conv_4_2], axis = 3)
		conv_6_1 = conv_layer_res(concat_6, [3, 3, 1024, 512], [1, 1, 1, 1], 'conv_6_1', if_relu = True)
		conv_6_2 = conv_layer_res(conv_6_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_6_2', if_relu = True)

		upsample_7 = up_layer(conv_6_2, tf.shape(conv_3_2), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, conv_3_2], axis = 3)
		conv_7_1 = conv_layer_res(concat_7, [3, 3, 512, 256], [1, 1, 1, 1], 'conv_7_1', if_relu = True)
		conv_7_2 = conv_layer_res(conv_7_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_7_2', if_relu = True)

		upsample_8 = up_layer(conv_7_2, tf.shape(conv_2_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, conv_2_2], axis = 3)
		conv_8_1 = conv_layer_res(concat_8, [3, 3, 256, 128], [1, 1, 1, 1], 'conv_8_1', if_relu = True)
		conv_8_2 = conv_layer_res(conv_8_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_8_2', if_relu = True)

		upsample_9 = up_layer(conv_8_2, tf.shape(conv_1_2), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, conv_1_2], axis = 3)
		conv_9_1 = conv_layer_res(concat_9, [3, 3, 128, 64], [1, 1, 1, 1], 'conv_9_1', if_relu = True)
		conv_9_2 = conv_layer_res(conv_9_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_9_2', if_relu = True)

		conv_10 = conv_layer_res(conv_9_2, [1, 1, 64, 1], [1, 1, 1, 1], 'conv_10')

		self.sig_1 = tf.sigmoid(tf.reshape(conv_10, [-1]))
		loss_1 = self._dice_loss(self.sig_1, tf.cast(tf.reshape(self.y_1, [-1]), tf.float32))
		self.dice_1 = - loss_1		


		conv_11_1 = conv_layer_res(conv_10, [3, 3, 1, 64], [1, 1, 1, 1], 'conv_11_1', if_relu = True)
		# conv_11_2 = conv_layer_res(conv_11_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_11_2', if_relu = True)
		max_11 = max_pool_layer(conv_11_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_11')

		conv_12_1 = conv_layer_res(max_11, [3, 3, 64, 128], [1, 1, 1, 1], 'conv_12_1', if_relu = True)
		# conv_12_2 = conv_layer_res(conv_12_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_12_2', if_relu = True)
		max_12 = max_pool_layer(conv_12_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_12')

		conv_13_1 = conv_layer_res(max_12, [3, 3, 128, 256], [1, 1, 1, 1], 'conv_13_1', if_relu = True)
		# conv_13_2 = conv_layer_res(conv_13_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_13_2', if_relu = True)
		max_13 = max_pool_layer(conv_13_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_13')

		conv_14_1 = conv_layer_res(max_13, [3, 3, 256, 512], [1, 1, 1, 1], 'conv_14_1', if_relu = True)
		# conv_14_2 = conv_layer_res(conv_14_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_14_2', if_relu = True)
		max_14 = max_pool_layer(conv_14_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_14')

		conv_15_1 = conv_layer_res(max_14, [3, 3, 512, 1024], [1, 1, 1, 1], 'conv_15_1', if_relu = True)
		# conv_15_2 = conv_layer_res(conv_15_1, [3, 3, 1024, 1024], [1, 1, 1, 1], 'conv_15_2', if_relu = True)

		upsample_16 = up_layer(conv_15_1, tf.shape(conv_14_1), 512, 1024, 2, 'upsample_16')
		concat_16 = tf.concat([upsample_16, conv_14_1], axis = 3)
		conv_16_1 = conv_layer_res(concat_16, [3, 3, 1024, 512], [1, 1, 1, 1], 'conv_16_1', if_relu = True)
		# conv_16_2 = conv_layer_res(conv_16_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_16_2', if_relu = True)

		upsample_17 = up_layer(conv_16_1, tf.shape(conv_13_1), 256, 512, 2, 'upsample_17')
		concat_17 = tf.concat([upsample_17, conv_13_1], axis = 3)
		conv_17_1 = conv_layer_res(concat_17, [3, 3, 512, 256], [1, 1, 1, 1], 'conv_17_1', if_relu = True)
		# conv_17_2 = conv_layer_res(conv_17_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_17_2', if_relu = True)

		upsample_18 = up_layer(conv_17_1, tf.shape(conv_12_1), 128, 256, 2, 'upsample_18')
		concat_18 = tf.concat([upsample_18, conv_12_1], axis = 3)
		conv_18_1 = conv_layer_res(concat_18, [3, 3, 256, 128], [1, 1, 1, 1], 'conv_18_1', if_relu = True)
		# conv_18_2 = conv_layer_res(conv_18_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_18_2', if_relu = True)

		upsample_19 = up_layer(conv_18_1, tf.shape(conv_11_1), 64, 128, 2, 'upsample_19')
		concat_19 = tf.concat([upsample_19, conv_11_1], axis = 3)
		conv_19_1 = conv_layer_res(concat_19, [3, 3, 128, 64], [1, 1, 1, 1], 'conv_19_1', if_relu = True)
		# conv_19_2 = conv_layer_res(conv_19_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_19_2', if_relu = True)

		conv_20 = conv_layer_res(conv_19_1, [1, 1, 64, 1], [1, 1, 1, 1], 'conv_20')

		self.sig_2 = tf.sigmoid(tf.reshape(conv_20, [-1]))
		loss_2 = self._dice_loss(self.sig_2, tf.cast(tf.reshape(self.y_2, [-1]), tf.float32))
		self.dice_2 = - loss_2


		conv_21_1 = conv_layer_res(conv_20, [3, 3, 1, 64], [1, 1, 1, 1], 'conv_21_1', if_relu = True)
		# conv_21_2 = conv_layer_res(conv_21_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_21_2', if_relu = True)
		max_21 = max_pool_layer(conv_21_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_21')

		conv_22_1 = conv_layer_res(max_21, [3, 3, 64, 128], [1, 1, 1, 1], 'conv_22_1', if_relu = True)
		# conv_22_2 = conv_layer_res(conv_22_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_22_2', if_relu = True)
		max_22 = max_pool_layer(conv_22_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_22')

		conv_23_1 = conv_layer_res(max_22, [3, 3, 128, 256], [1, 1, 1, 1], 'conv_23_1', if_relu = True)
		# conv_23_2 = conv_layer_res(conv_23_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_23_2', if_relu = True)
		max_23 = max_pool_layer(conv_23_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_23')

		conv_24_1 = conv_layer_res(max_23, [3, 3, 256, 512], [1, 1, 1, 1], 'conv_24_1', if_relu = True)
		# conv_24_2 = conv_layer_res(conv_24_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_24_2', if_relu = True)
		max_24 = max_pool_layer(conv_24_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_24')

		conv_25_1 = conv_layer_res(max_24, [3, 3, 512, 1024], [1, 1, 1, 1], 'conv_25_1', if_relu = True)
		# conv_25_2 = conv_layer_res(conv_25_1, [3, 3, 1024, 1024], [1, 1, 1, 1], 'conv_25_2', if_relu = True)

		upsample_26 = up_layer(conv_25_1, tf.shape(conv_24_1), 512, 1024, 2, 'upsample_26')
		concat_26 = tf.concat([upsample_26, conv_24_1], axis = 3)
		conv_26_1 = conv_layer_res(concat_26, [3, 3, 1024, 512], [1, 1, 1, 1], 'conv_26_1', if_relu = True)
		# conv_26_2 = conv_layer_res(conv_26_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv_26_2', if_relu = True)

		upsample_27 = up_layer(conv_26_1, tf.shape(conv_23_1), 256, 512, 2, 'upsample_27')
		concat_27 = tf.concat([upsample_27, conv_23_1], axis = 3)
		conv_27_1 = conv_layer_res(concat_27, [3, 3, 512, 256], [1, 1, 1, 1], 'conv_27_1', if_relu = True)
		# conv_27_2 = conv_layer_res(conv_27_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv_27_2', if_relu = True)

		upsample_28 = up_layer(conv_27_1, tf.shape(conv_22_1), 128, 256, 2, 'upsample_28')
		concat_28 = tf.concat([upsample_28, conv_22_1], axis = 3)
		conv_28_1 = conv_layer_res(concat_28, [3, 3, 256, 128], [1, 1, 1, 1], 'conv_28_1', if_relu = True)
		# conv_28_2 = conv_layer_res(conv_28_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv_28_2', if_relu = True)

		upsample_29 = up_layer(conv_28_1, tf.shape(conv_21_1), 64, 128, 2, 'upsample_29')
		concat_29 = tf.concat([upsample_29, conv_21_1], axis = 3)
		conv_29_1 = conv_layer_res(concat_29, [3, 3, 128, 64], [1, 1, 1, 1], 'conv_29_1', if_relu = True)
		#1conv_29_2 = conv_layer_res(conv_29_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv_29_2', if_relu = True)

		conv_30 = conv_layer_res(conv_29_1, [1, 1, 64, 1], [1, 1, 1, 1], 'conv_30')

		self.sig_3 = tf.sigmoid(tf.reshape(conv_30, [-1]))
		loss_3 = self._dice_loss(self.sig_3, tf.cast(tf.reshape(self.y_3, [-1]), tf.float32))
		self.dice_3 = - loss_3

		loss = (loss_1 + loss_2 + loss_3) / 3
		return loss

	def train(self, X, y_1, y_2, y_3, 
		model_name,
		batch_size = 8, 
		learning_rate = 1e-4, 
		epoch = 25, 
		dropout = 0.5, 
		restore = False,
		N_worst = 1e6,
		thre = 0.9
		):


		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		# self._print_global_variables()

		saver = tf.train.Saver()
		if restore:
			print 'loading from model...'
			saver.restore(self.sess, './models/' + model_name + '.ckpt')
		else:
			self.sess.run(tf.global_variables_initializer())

		# Start training 
		for e in xrange(epoch):

			X_train, y_1_train, y_2_train, y_3_train, X_val, y_1_val, y_2_val, y_3_val = select_val_set_4_arg(X, y_1, y_2, y_3)
			X_train, y_1_train, y_2_train, y_3_train = remove_background_3d_with_label_4_arg(X_train, y_1_train, y_2_train, y_3_train)
			X_val, y_1_val, y_2_val, y_3_val = remove_background_3d_with_label_4_arg(X_val, y_1_val, y_2_val, y_3_val)

			print X_train.shape, y_1_train.shape, y_2_train.shape, y_3_train.shape
			print X_val.shape, y_1_val.shape, y_2_val.shape, y_3_val.shape

			num_train = X_train.shape[0]
			num_val = X_val.shape[0]
			indices = np.arange(num_train)
			indices_val = np.arange(num_val)
			ite_per_epoch = np.round(num_train / batch_size)
		
			self.last = time.time()

			for i in xrange(ite_per_epoch):
				np.random.shuffle(indices)
				index = indices[:batch_size]

				X_train_sample = X_train[index]
				y_1_train_sample = y_1_train[index]
				y_2_train_sample = y_2_train[index]
				y_3_train_sample = y_3_train[index]

				_, dice_1_val, dice_2_val, dice_3_val= self.sess.run([optimizer, 
					self.dice_1, self.dice_2, self.dice_3], feed_dict = {
					self.X_train: X_train_sample, self.y_1: y_1_train_sample, self.y_2: y_2_train_sample, self.y_3: y_3_train_sample, 
					})

				now = time.time()
				interval = now - self.last
				self.last = time.time()

				if i % 3 == 0:
					print 'ite {0} finished, loss: {1:.6e} {2:.6e} {3:.6e} time consumed: {4:.3f}s'.format(i,
					 dice_1_val, dice_2_val, dice_3_val, interval), (dice_1_val + dice_2_val + dice_3_val) / 3
						# num_pixels_val.astype('int32'), np.sum(label_val.reshape((-1, 5)), axis = 0).astype('int32')

			if X_val.shape[0] > 12:
				np.random.shuffle(indices_val)
				index_val = indices_val[:12]
				print 'epoch', self.predict(X_val[index_val], y_1_val[index_val],y_2_val[index_val], y_3_val[index_val])
			else:
				print 'epoch', self.predict(X_val, y_1_val, y_2_val, y_3_val)

			print 'epoch {0} finished'.format(e)
			print

			if (e + 26) % 25 == 0:
				thre -= 0.1
				saver.save(self.sess, './models/' + model_name + np.str(e) + '.ckpt')
				print 'save success'

		print 'Training Finished!'

	def predict(self, X_test, y_1, y_2, y_3, restore = False, model_name = None):
		saver = tf.train.Saver()
		if restore:
			print 'loading from model' + model_name 
			saver.restore(self.sess, './models/' + model_name + '.ckpt')

		dice_1_val, dice_2_val, dice_3_val= self.sess.run([
			self.dice_1, self.dice_2, self.dice_3], feed_dict = {
			self.X_train: X_test, self.y_1: y_1, self.y_2: y_2, self.y_3: y_3
			})

		return dice_1_val, dice_2_val, dice_3_val

	def eval(self, X_test, y_1, y_2, y_3, model_name):
		saver = tf.train.Saver()
		print 'loading from model' + model_name 
		saver.restore(self.sess, './models/' + model_name + '.ckpt')

		if len(X_test.shape) == 5:
			mean_pre = np.zeros([X_test.shape[0]])
			mean_IU = np.zeros([X_test.shape[0]])
			mean_dice = np.zeros([X_test.shape[0]])
			for i in xrange(X_test.shape[0]):
				prob_1_val = np.empty([0])
				prob_2_val = np.empty([0])
				prob_3_val = np.empty([0])
				for j in xrange(X_test.shape[1] / 31):
					prob_1, prob_2, prob_3 = self.sess.run([self.sig_1, self.sig_2, self.sig_3], feed_dict = {
						self.X_train: X_test[i, j * 31:(j + 1) * 31]
						})
					prob_1_val = np.concatenate([prob_1_val, prob_1], axis = 0)
					prob_2_val = np.concatenate([prob_2_val, prob_2], axis = 0)
					prob_3_val = np.concatenate([prob_3_val, prob_3], axis = 0)

				pre = np.array([binary_recall(1 - prob_1_val.reshape((-1)), - y_1[i].reshape((-1)) + 1),
					binary_recall(prob_1_val.reshape((-1)), y_1[i].reshape((-1))), 
					binary_recall(prob_2_val.reshape((-1)), y_2[i].reshape((-1))),
					binary_recall(prob_3_val.reshape((-1)), y_3[i].reshape((-1)))
					])  
				IU = np.array([binary_IoU(1 - prob_1_val.reshape((-1)), - y_1[i].reshape((-1)) + 1),
					binary_IoU(prob_1_val.reshape((-1)), y_1[i].reshape((-1))), 
					binary_IoU(prob_2_val.reshape((-1)), y_2[i].reshape((-1))),
					binary_IoU(prob_3_val.reshape((-1)), y_3[i].reshape((-1)))
					]) 
				dice = np.array([binary_dice(1 - prob_1_val.reshape((-1)), - y_1[i].reshape((-1)) + 1),
					binary_dice(prob_1_val.reshape((-1)), y_1[i].reshape((-1))), 
					binary_dice(prob_2_val.reshape((-1)), y_2[i].reshape((-1))),
					binary_dice(prob_3_val.reshape((-1)), y_3[i].reshape((-1)))
					]) 

				print i
				print pre, np.mean(pre)
				print IU, np.mean(IU)
				print dice, np.mean(dice)

				mean_pre[i] = np.mean(pre)
				mean_IU[i] = np.mean(IU)
				mean_dice[i] = np.mean(dice)

			print np.mean(mean_pre), '\n', np.mean(mean_IU), '\n', np.mean(mean_dice)


	def _dice_loss(self, prob, label):

		intersection = tf.reduce_sum(prob * label)
   		return - (2. * intersection + 1) / (tf.reduce_sum(prob) + tf.reduce_sum(label) + 1)


if __name__ == '__main__':
	print 'loading from LGG_train.npz...'
	f = np.load(Base + '/LGG_new.npz')
	X = f['X']
	y_1 = f['y0']
	y_2 = f['y1']
	y_3 = f['y2']

	print X.shape, y_1.shape, y_2.shape, y_3.shape		

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	net = WNET(input_shape = (240, 240, 4), num_classes = 5)
	net.train(X, y_1, y_2, y_3, model_name = 'model_wnet_1_',
	 batch_size = 12, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e6, thre = 0.6)
	# else:
	# 	exit(0)