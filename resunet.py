import tensorflow as tf 
import numpy as np 
import time

from fcnn2d import FCNN_2D, Base
from resnet import RESNET
from layers import *
from auxiliary import *
from pre_process import *

class RESUNET_3D(RESNET):
	
	def build(self, X, y):
		stack_1 = stack_layer_3d(X, 4, 64, 64, self.is_training, 'stack_1')
		max_1 = max_pool_layer_3d(stack_1, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'max_1')
	
		stack_2 = stack_layer_3d(max_1, 64, 128, 128, self.is_training, 'stack_2')
		max_2 = max_pool_layer_3d(stack_2, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'max_2')

		stack_3 = stack_layer_3d(max_2, 128, 256, 256, self.is_training, 'stack_3')
		max_3 = max_pool_layer_3d(stack_3, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'max_3')

		stack_4 = stack_layer_3d(max_3, 256, 512, 512, self.is_training, 'stack_4')
		max_4 = max_pool_layer_3d(stack_4, [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'max_4')

		stack_5 = stack_layer_3d(max_4, 512, 1024, 1024, self.is_training, 'stack_5')

		# upsample layers No.1 for comp binary classification

		upsample_6 = up_layer_3d(stack_5, tf.shape(stack_4), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, stack_4], axis = 4)
		stack_6 = stack_layer_3d(concat_6, 1024, 1024, 512, self.is_training, 'stack_6')

		upsample_7 = up_layer_3d(stack_6, tf.shape(stack_3), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, stack_3], axis = 4)
		stack_7 = stack_layer_3d(concat_7, 512, 512, 256, self.is_training, 'stack_7')

		upsample_8 = up_layer_3d(stack_7, tf.shape(stack_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, stack_2], axis = 4)
		stack_8 = stack_layer_3d(concat_8, 256, 256, 128, self.is_training, 'stack_8')

		upsample_9 = up_layer_3d(stack_8, tf.shape(stack_1), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, stack_1], axis = 4)
		stack_9 = stack_layer_3d(concat_9, 128, 128, 64, self.is_training, 'stack_9')

		conv_10 = conv_layer_res_3d(stack_9, [1, 1, 1, 64, self.num_classes], [1, 1, 1, 1, 1], 'conv_10')


		logits = conv_10
		self.result = tf.argmax(logits, axis = 4)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix



class RES_UNET(RESNET):

	def build(self, X, y):

		'''
		resnet + unet
		'''

		stack_1 = stack_layer_not_act(X, 4, 64, 64, self.is_training, 'stack_1')
		# bn_1 = tf.nn.relu(bn_layer(stack_1, self.is_training, 'bn_1'))
		max_1 = max_pool_layer(stack_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')
	
		stack_2 = stack_layer_not_act(max_1, 64, 128, 128, self.is_training, 'stack_2')
		# bn_2 = tf.nn.relu(bn_layer(stack_2, self.is_training, 'bn_2'))
		max_2 = max_pool_layer(stack_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		stack_3 = stack_layer_not_act(max_2, 128, 256, 256, self.is_training, 'stack_3')
		# bn_3 = tf.nn.relu(bn_layer(stack_3, self.is_training, 'bn_3'))
		max_3 = max_pool_layer(stack_3, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		stack_4 = stack_layer_not_act(max_3, 256, 512, 512, self.is_training, 'stack_4')
		# bn_4 = tf.nn.relu(bn_layer(stack_4, self.is_training, 'bn_4'))
		max_4 = max_pool_layer(stack_4, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		stack_5 = stack_layer_not_act(max_4, 512, 1024, 1024, self.is_training, 'stack_5')
		# bn_5 = tf.nn.relu(bn_layer(stack_5, self.is_training, 'bn_5'))

		# upsample layers No.1 for comp binary classification

		upsample_6 = up_layer(stack_5, tf.shape(stack_4), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, stack_4], axis = 3)
		stack_6 = stack_layer_not_act(concat_6, 1024, 1024, 512, self.is_training, 'stack_6')
		# bn_6 = tf.nn.relu(bn_layer(stack_6, self.is_training, 'bn_6'))

		upsample_7 = up_layer(stack_6, tf.shape(stack_3), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, stack_3], axis = 3)
		stack_7 = stack_layer_not_act(concat_7, 512, 512, 256, self.is_training, 'stack_7')
		# bn_7 = tf.nn.relu(bn_layer(stack_7, self.is_training, 'bn_7'))

		upsample_8 = up_layer(stack_7, tf.shape(stack_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, stack_2], axis = 3)
		stack_8 = stack_layer_not_act(concat_8, 256, 256, 128, self.is_training, 'stack_8')
		# bn_8 = tf.nn.relu(bn_layer(stack_8, self.is_training, 'bn_8'))

		upsample_9 = up_layer(stack_8, tf.shape(stack_1), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, stack_1], axis = 3)
		stack_9 = stack_layer_not_act(concat_9, 128, 128, 64, self.is_training, 'stack_9')
		# bn_9 = tf.nn.relu(bn_layer(stack_9, self.is_training, 'bn_9'))

		conv_10 = conv_layer_res(stack_9, [1, 1, 64, self.num_classes], [1, 1, 1, 1], 'conv_10')


		logits = conv_10
		self.result = tf.argmax(logits, axis = 3)

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix

	def multi_binary_build(self, X, y):
		stack_1 = stack_layer(X, 4, 64, 64, self.is_training, 'stack_1')
		# bn_1 = tf.nn.relu(bn_layer(stack_1, self.is_training, 'bn_1'))
		max_1 = max_pool_layer(stack_1, [1, 2, 2, 1], [1, 2, 2, 1], 'max_1')
	
		stack_2 = stack_layer(max_1, 64, 128, 128, self.is_training, 'stack_2')
		# bn_2 = tf.nn.relu(bn_layer(stack_2, self.is_training, 'bn_2'))
		max_2 = max_pool_layer(stack_2, [1, 2, 2, 1], [1, 2, 2, 1], 'max_2')

		stack_3 = stack_layer(max_2, 128, 256, 256, self.is_training, 'stack_3')
		# bn_3 = tf.nn.relu(bn_layer(stack_3, self.is_training, 'bn_3'))
		max_3 = max_pool_layer(stack_3, [1, 2, 2, 1], [1, 2, 2, 1], 'max_3')

		stack_4 = stack_layer(max_3, 256, 512, 512, self.is_training, 'stack_4')
		# bn_4 = tf.nn.relu(bn_layer(stack_4, self.is_training, 'bn_4'))
		max_4 = max_pool_layer(stack_4, [1, 2, 2, 1], [1, 2, 2, 1], 'max_4')

		stack_5 = stack_layer(max_4, 512, 1024, 1024, self.is_training, 'stack_5')
		# bn_5 = tf.nn.relu(bn_layer(stack_5, self.is_training, 'bn_5'))

		# upsample layers No.1 for comp binary classification

		upsample_6 = up_layer(stack_5, tf.shape(stack_4), 512, 1024, 2, 'upsample_6')
		concat_6 = tf.concat([upsample_6, stack_4], axis = 3)
		stack_6 = stack_layer(concat_6, 1024, 1024, 512, self.is_training, 'stack_6')
		# bn_6 = tf.nn.relu(bn_layer(stack_6, self.is_training, 'bn_6'))

		upsample_7 = up_layer(stack_6, tf.shape(stack_3), 256, 512, 2, 'upsample_7')
		concat_7 = tf.concat([upsample_7, stack_3], axis = 3)
		stack_7 = stack_layer(concat_7, 512, 512, 256, self.is_training, 'stack_7')
		# bn_7 = tf.nn.relu(bn_layer(stack_7, self.is_training, 'bn_7'))

		upsample_8 = up_layer(stack_7, tf.shape(stack_2), 128, 256, 2, 'upsample_8')
		concat_8 = tf.concat([upsample_8, stack_2], axis = 3)
		stack_8 = stack_layer(concat_8, 256, 256, 128, self.is_training, 'stack_8')
		# bn_8 = tf.nn.relu(bn_layer(stack_8, self.is_training, 'bn_8'))

		upsample_9 = up_layer(stack_8, tf.shape(stack_1), 64, 128, 2, 'upsample_9')
		concat_9 = tf.concat([upsample_9, stack_1], axis = 3)
		stack_9 = stack_layer(concat_9, 128, 128, 64, self.is_training, 'stack_9')
		# bn_9 = tf.nn.relu(bn_layer(stack_9, self.is_training, 'bn_9'))

		# if self.is_training:
		# 	logits_1 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_1')
		# 	logits_2 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_2') + logits_1 / 2
		# 	logits_3 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_3') + logits_2 / 2
		# else:
		# logits_1 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_1')
		# logits_2 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_2')
		# logits_3 = conv_layer_res(bn_9, [1, 1, 64, 1], [1, 1, 1, 1], 'logits_3')

		logits = conv_layer_res(stack_9, [1, 1, 64, self.num_classes], [1, 1, 1, 1], 'logits')
		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		prob = tf.nn.softmax(logits = reshaped_logits)
		prob_0 = tf.reduce_sum(prob[:, 1:5], axis = 1, keep_dims = True)
		prob_1 = tf.reduce_sum(prob[:, 3:5], axis = 1, keep_dims = True) + tf.reshape(prob[:, 1], [-1, 1])
		prob_2 = tf.reshape(prob[:, 4], [-1, 1])


		# reshaped_logits_1 = tf.reshape(logits_1, [-1, 1])
		# reshaped_logits_2 = tf.reshape(logits_2, [-1, 1])
		# reshaped_logits_3 = tf.reshape(logits_3, [-1, 1])
		# reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		# prob_1 = tf.sigmoid(reshaped_logits_1)
		# prob_2 = tf.sigmoid(reshaped_logits_2)
		# prob_3 = tf.sigmoid(reshaped_logits_3)

		return [prob_0, prob_1, prob_2]


	def multi_dice_train(self, X_input, y_input, 
		model_name,
		train_mode = 0,
		num_gpu = 3,
		batch_size = 8, 
		learning_rate = 1e-4, 
		epoch = 25, 
		dropout = 0.5, 
		restore = False,
		N_worst = 1e6,
		thre = 0.9
		):
		
		with tf.device('/cpu:0'):

			global_step = tf.get_variable(
        	 'global_step', [],
        	 initializer=tf.constant_initializer(0), trainable=False)

			lr = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.95, staircase = True)
			optimizer = tf.train.AdamOptimizer(lr)

			tower_grads = []
			with tf.variable_scope(tf.get_variable_scope()):
				for i in xrange(num_gpu):
					with tf.device('/gpu:%d' % i):
						with tf.name_scope('gpu_%d' % i) as scope:

							# allocate input placeholder
							X, y, label = self._get_input()
							prob_0, prob_1, prob_2 = self.multi_binary_build(X, label)
							loss_list = self._binary_dice_loss([prob_0, prob_1, prob_2], tf.reshape(label, [-1, self.num_classes]))
							loss = sum(loss_list) / 3

							# add to collections
							tf.add_to_collection('Xs', X)
							tf.add_to_collection('ys', y)
							tf.add_to_collection('labels', tf.reshape(label, [-1, self.num_classes]))
							tf.add_to_collection('losses', loss)
							tf.add_to_collection('loss_list', loss_list)
							tf.add_to_collection('prob_0s', prob_0)
							tf.add_to_collection('prob_1s', prob_1)
							tf.add_to_collection('prob_2s', prob_2)

							tf.get_variable_scope().reuse_variables()
							grads = optimizer.compute_gradients(loss)
							tower_grads.append(grads)

			total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
			total_label = tf.concat(tf.get_collection('labels'), axis = 0, name = 'total_label')
			total_loss_list = tf.get_collection('loss_list')
			total_prob_0 = tf.concat(tf.get_collection('prob_0s'), axis = 0, name = 'total_prob_0')
			total_prob_1 = tf.concat(tf.get_collection('prob_1s'), axis = 0, name = 'total_prob_1')
			total_prob_2 = tf.concat(tf.get_collection('prob_2s'), axis = 0, name = 'total_prob_2')

			grads = self._average_gradients(tower_grads)
			apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

			variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
			variables_averages_op = variable_averages.apply(tf.trainable_variables())

			train_op = tf.group(apply_gradient_op, variables_averages_op)

			saver = tf.train.Saver()

			# judge whether to restore from saved models
			if restore:
				print 'loading from model' + model_name
				start_epoch = np.int(model_name.split('_')[-1]) + 1
				saver.restore(self.sess, './models/' + model_name + '.ckpt')
			else:
				start_epoch = 0
				self.sess.run(tf.global_variables_initializer())

			# Start training 
			for e in xrange(start_epoch, start_epoch + epoch):

				X_train, y_train, X_val, y_val = select_val_set(X_input, y_input)
				X_train, y_train = remove_background_3d_with_label(X_train, y_train)
				X_val, y_val = remove_background_3d_with_label(X_val, y_val)

				print X_train.shape, y_train.shape
				print X_val.shape, y_val.shape

				num_train = X_train.shape[0]
				num_val = X_val.shape[0]
				indices = np.arange(num_train)
				indices_val = np.arange(num_val)
				ite_per_epoch = np.round(num_train / batch_size / num_gpu)
			
				self.last = time.time()

				for i in xrange(ite_per_epoch):
					np.random.shuffle(indices)
					index = indices[:batch_size * num_gpu]

					X_train_sample = X_train[index]
					y_train_sample = y_train[index]

					X_list_val = []
					y_list_val = []
					for j in xrange(num_gpu):
						X_list_val.append(X_train_sample[j * batch_size: (j + 1) * batch_size])
						y_list_val.append(y_train_sample[j * batch_size: (j + 1) * batch_size])

					_, loss_list = self.sess.run(
						[train_op, total_loss_list], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_list_val), tuple(tf.get_collection('ys')): tuple(y_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: True})

					now = time.time()
					interval = now - self.last
					self.last = time.time()

					print 'ite ' + np.str(i) + ' time: ' + np.str(interval) + ' ', loss_list

				# Now it's time to evaluate on the validation data set
				if X_val.shape[0] > 32 * num_gpu:
					X_val_list_val = []
					y_val_list_val = []
					for i in xrange(num_gpu):
						X_val_list_val.append(X_val[i * 32: (i + 1) * 32])
						y_val_list_val.append(y_val[i * 32: (i + 1) * 32])
					prob_0_val, prob_1_val, prob_2_val, label_val = self.sess.run([
						total_prob_0, total_prob_1, total_prob_2, total_label], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_val_list_val), tuple(tf.get_collection('ys')): tuple(y_val_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: False})
					y_out = y_val[index_val]
				else:
					X_val_list_val = []
					y_val_list_val = []
					for i in xrange(num_gpu):
						X_val_list_val.append(X_val[i * X_val.shape[0] // num_gpu:(i + 1) * X_val.shape[0] // num_gpu])
						y_val_list_val.append(y_val[i * y_val.shape[0] // num_gpu:(i + 1) * y_val.shape[0] // num_gpu])
					prob_0_val, prob_1_val, prob_2_val, label_val = self.sess.run([
						total_prob_0, total_prob_1, total_prob_2, total_label], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_val_list_val), tuple(tf.get_collection('ys')): tuple(y_val_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: False})
					y_out = y_val
				
				y_bin = np.bincount(y_out.reshape(-1), minlength = 4)

				print 'epoch {0} finished'.format(e)
				print self._binary_eval([prob_0_val, prob_1_val, prob_2_val], label_val)

				if train_mode == 0:
					ans = raw_input('Do you want to save? [y/q/else]: ')
					if ans == 'y':
						l = model_name.split('_')
						saver.save(self.sess, './models/' + l[0] + '_' + l[1] + '_' + l[2] + '_' + np.str(e) + '.ckpt')
					elif ans == 'q':
						exit(0)
				else:
					if (e + 1) % 25 == 0:
						l = model_name.split('_')
						saver.save(self.sess, './models/' + l[0] + '_' + l[1] + '_' + l[2] + '_' + np.str(e) + '.ckpt')	 
					else:
						pass

			print 'Training Finished!'


	def multi_dice_predict(self, model_name, X_test, y_test = None, dropout = 0.5, num_gpu = 3):

		with tf.device('cpu:0'):

			global_step = tf.get_variable(
        	 'global_step', [],
        	 initializer=tf.constant_initializer(0), trainable=False)

			tower_grads = []
			with tf.variable_scope(tf.get_variable_scope()):
				for i in xrange(num_gpu):
					with tf.device('/gpu:%d' % i):
						with tf.name_scope('gpu_%d' % i) as scope:

							# allocate input placeholder
							X, y, label = self._get_input()
							prob_0, prob_1, prob_2 = self.multi_binary_build(X, label)
							loss_list = self._binary_pre_loss([prob_0, prob_1, prob_2], tf.reshape(label, [-1, self.num_classes]))
							loss = sum(loss_list) / 3

							# add to collections
							tf.add_to_collection('Xs', X)
							tf.add_to_collection('ys', y)
							tf.add_to_collection('labels', tf.reshape(label, [-1, self.num_classes]))
							tf.add_to_collection('losses', loss)
							tf.add_to_collection('loss_list', loss_list)
							tf.add_to_collection('prob_0s', prob_0)
							tf.add_to_collection('prob_1s', prob_1)
							tf.add_to_collection('prob_2s', prob_2)

							tf.get_variable_scope().reuse_variables()

			total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
			total_label = tf.concat(tf.get_collection('labels'), axis = 0, name = 'total_label')
			total_loss_list = tf.get_collection('loss_list')
			total_prob_0 = tf.concat(tf.get_collection('prob_0s'), axis = 0, name = 'total_prob_0')
			total_prob_1 = tf.concat(tf.get_collection('prob_1s'), axis = 0, name = 'total_prob_1')
			total_prob_2 = tf.concat(tf.get_collection('prob_2s'), axis = 0, name = 'total_prob_2')

			saver = tf.train.Saver()

			print 'loading from model ' + model_name 
			saver.restore(self.sess, './models/' + model_name + '.ckpt')

			if y_test is not None:

				if len(X_test.shape) == 5:
					N, D, W, H, C = X_test.shape
					
					final_eval_mat = np.empty([0, 4, 4])
					# caculate every 3D image 
					for i in xrange(N):
						label_val = np.empty([0, self.num_classes])
						prob_val_0 = np.empty([0, 1])
						prob_val_1 = np.empty([0, 1])
						prob_val_2 = np.empty([0, 1])

						# batch_size_test = X_test.shape[0] // num_gpu
						for k in xrange(5):
							X_test_list_val = []
							y_test_list_val = []
							for j in xrange(num_gpu):
								X_test_list_val.append(X_test[i, D * k / 5 + 31 * j // num_gpu: D * k / 5 + 31 * (j + 1) // num_gpu])
								y_test_list_val.append(y_test[i, D * k / 5 + 31 * j // num_gpu: D * k / 5 + 31 * (j + 1) // num_gpu])
							prob_val_t_0, prob_val_t_1, prob_val_t_2, label_val_t = self.sess.run(
								[total_prob_0, total_prob_1, total_prob_2, total_label], feed_dict = {
								tuple(tf.get_collection('Xs')): tuple(X_test_list_val), tuple(tf.get_collection('ys')): tuple(y_test_list_val), 
								self.dropout: dropout, self.is_training: False})

							label_val_t = label_val_t.reshape((-1, self.num_classes))
							label_val = np.concatenate([label_val, label_val_t], axis = 0)
							prob_val_0 = np.concatenate([prob_val_0, prob_val_t_0], axis = 0)
							prob_val_1 = np.concatenate([prob_val_1, prob_val_t_1], axis = 0)
							prob_val_2 = np.concatenate([prob_val_2, prob_val_t_2], axis = 0)

						conmat = self._binary_eval([prob_val_0, prob_val_1, prob_val_2], label_val)
						print conmat
						final_eval_mat = np.concatenate([final_eval_mat, conmat.reshape(1, 4, 4)], axis = 0)
					print np.mean(final_eval_mat, axis = 0)

						
				elif len(X_test.shape) == 4:
					X_test_list_val = []
					y_test_list_val = []
					for j in xrange(num_gpu):
						X_test_list_val.append(X_test[j * X_test.shape[0] // num_gpu: (j + 1) * X_test.shape[0] // num_gpu])
						y_test_list_val.append(y_test[j * X_test.shape[0] // num_gpu: (j + 1) * X_test.shape[0] // num_gpu])
					prob_val_t, label_val_t, confusion_matrix_val = self.sess.run([
						total_prob, total_label, total_confusion_matrix], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_test_list_val), tuple(tf.get_collection('ys')): tuple(y_test_list_val), 
						self.dropout: dropout, self.is_training: False})
					label_val = label_val.reshape((-1, self.num_classes))
					print label_val.shape, prob_val.shape

					pre, IU, dice = self.evaluate(prob_val, label_val)

					return pre, IU, dice, confusion_matrix_val

				else:
					print "input's dimention error!"
					exit(1)

			# When we need to output the prediction results
			else:
				if len(X_test.shape) == 5:
					N, D, W, H, C = X_test.shape
					result = np.empty((N, D, W, H))
					for i in xrange(N):
						X_test_list = []
						for j in xrange(num_gpu):
							X_test_list.append(X_test[i, j * D // 3:(j + 1) * D // 3])
						prob_val = self.sess.run(total_prob, feed_dict = {tuple(tf.get_collection('Xs')): tuple(X_test_list_val),
						 self.dropout: dropout, self.is_training: False})
						result[i] = np.argmax(prob_val.reshape(D, W, H, C), axis = -1)
					return result

				elif len(X_test.shape) == 4:
					N, W, H, C = X_test.shape
					X_test_list = []
					for j in xrange(num_gpu):
						X_test_list.append(X_test[j * D // 3:(j + 1) * D // 3])
					prob_val = self.sess.run(total_prob, feed_dict = {tuple(tf.get_collection('Xs')): tuple(X_test_list_val),
						 self.dropout: dropout, self.is_training: False})
					result = np.argmax(prob_val.reshape(N, W, H, C), axis = -1)
					return result
				else:
					print "input's dimention error!"
					exit(1)

if __name__ == '__main__':
	print 'loading from HGG_train.npz...'
	f = np.load(Base + '/HGG_train2.npz')
	X = f['X']
	y = f['y']

	print X.shape, y.shape		

	# ans = raw_input('Do you want to continue? [y/else]: ')
	# if ans == 'y':
	net = RES_UNET(input_shape = (240, 240, 4), num_classes = 5)
	# net.multi_gpu_train(X, y, model_name = 'model_resunet_5', train_mode = 1,
	#  batch_size = 8, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 2e5)
	net.multi_gpu_train(X, y, model_name = 'model_resunet_1', train_mode = 1, num_gpu = 1, 
     batch_size = 32, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e6, thre = 0.9)
 

	# else:
	# 	exit(0)
