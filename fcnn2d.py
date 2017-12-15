import numpy as np
import os
import time
import tensorflow as tf 
from layers import *
from pre_process import remove_background_3d_with_label
from auxiliary import *

Base = './data/npz'


class FCNN_2D:
	'''
	A Fully Convolutional Neural Network designed for MRI images' segmentation
	Region 1: Complete tumor (labels 1+2+3+4 for patient data)
	Region 2: Tumor core (labels 1+3+4 for patient data)
	Region 3: Enhancing tumor (label 4 for patient data) 
	'''

	def __init__(self, input_shape, num_classes):
		self.sess = tf.Session(config=tf.ConfigProto(
         allow_soft_placement=True,
         log_device_placement=False))

		self.input_shape = input_shape
		self.W, self.H, self.C = input_shape
		self.num_classes = num_classes
		
		self.dropout = tf.placeholder(tf.float32, name = 'dropout')
		self.N_worst = tf.placeholder(tf.int32, name = 'N_worst')
		self.thre = tf.placeholder(tf.float32, name = 'threshold')
		self.is_training = tf.placeholder(tf.bool, name = 'is_training')

		self.last = time.time()


	def confusion_matrix(self, prob, label):
		labels = tf.argmax(label, axis = -1)
		probs = tf.argmax(prob, axis = -1)
		return tf.confusion_matrix(labels, probs)


	def build(self, X, y):

		conv1_1 = conv_layer(X, 'conv1_1_W', 'conv1_1_b', name='conv1_1')
		conv1_2 = conv_layer(conv1_1, 'conv1_2_W', 'conv1_2_b', name='conv1_2')
		pool1 = max_pool_layer(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool1')

		conv2_1 = conv_layer(pool1, 'conv2_1_W', 'conv2_1_b', name='conv2_1')
		conv2_2 = conv_layer(conv2_1, 'conv2_2_W', 'conv2_2_b', name='conv2_2')
		pool2 = max_pool_layer(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool2')

		conv3_1 = conv_layer(pool2, 'conv3_1_W', 'conv3_1_b', name='conv3_1')
		conv3_2 = conv_layer(conv3_1, 'conv3_2_W', 'conv3_2_b', name='conv3_2')
		conv3_3 = conv_layer(conv3_2, 'conv3_3_W', 'conv3_3_b', name='conv3_3')
		pool3 = max_pool_layer(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool3')

		conv4_1 = conv_layer(pool3, 'conv4_1_W', 'conv4_1_b', name='conv4_1')
		conv4_2 = conv_layer(conv4_1, 'conv4_2_W', 'conv4_2_b', name='conv4_2')
		conv4_3 = conv_layer(conv4_2, 'conv4_3_W', 'conv4_3_b', name='conv4_3')
		pool4 = max_pool_layer(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool4')

		conv5_1 = conv_layer(pool4, 'conv5_1_W', 'conv5_1_b', name='conv5_1')
		conv5_2 = conv_layer(conv5_1, 'conv5_2_W', 'conv5_2_b', name='conv5_2')
		conv5_3 = conv_layer(conv5_2, 'conv5_3_W', 'conv5_3_b', name='conv5_3')
		pool5 = max_pool_layer(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool5')

		fc_1 = fully_collected_layer(pool5, 'fc_1', self.dropout)
		fc_2 = fully_collected_layer(fc_1, 'fc_2', self.dropout)
		fc_3 = fully_collected_layer(fc_2, 'fc_3', self.dropout)

		# Now we start upsampling and skip layer connections.
		img_shape = tf.shape(X)
		dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])
		upsample_1 = upsample_layer(fc_3, dconv3_shape, self.num_classes, 'upsample_1', 32)

		skip_1 = skip_layer_connection(pool4, 'skip_1', 512, stddev=0.00001)
		upsample_2 = upsample_layer(skip_1, dconv3_shape, self.num_classes, 'upsample_2', 16)

		skip_2 = skip_layer_connection(pool3, 'skip_2', 256, stddev=0.0001)
		upsample_3 = upsample_layer(skip_2, dconv3_shape, self.num_classes, 'upsample_3', 8)

		logits = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))

		reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
		reshaped_labels = tf.reshape(y, [-1, self.num_classes])

		prob = tf.nn.softmax(logits = reshaped_logits)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = reshaped_logits, labels = reshaped_labels)
		confusion_matrix = self.confusion_matrix(prob, reshaped_labels)

		return cross_entropy, prob, confusion_matrix

	def train(self, X_input, y_input, 
		model_name,
		train_mode = 0,
		batch_size = 8, 
		learning_rate = 1e-4, 
		epoch = 25, 
		dropout = 0.5, 
		restore = False,
		N_worst = 1e6,
		thre = 0.9
		):
		'''
		Basic training function could serve for various models
		train_mode == 0 step training
		train_mode != 0 nohup training
		'''
		
		# Construct the network
		X, y, label = self._get_input()
		cross_entropy, prob, confusion_matrix = self.build(X, label)
		loss, num_pixels = self._bootstrapping_loss(cross_entropy, prob, label)

		global_step = tf.get_variable(
        	'global_step', [],
        	initializer=tf.constant_initializer(0), trainable=False)

		lr = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.9, staircase = True)
		optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)

		saver = tf.train.Saver()

		# judge whether to restore from saved models
		if restore:
			print 'loading from model' + model_name
			start_epoch = np.int(model_name.split('_')[-1])
			saver.restore(self.sess, './models/' + model_name + '.ckpt')
		else:
			start_epoch = 0
			self.sess.run(tf.global_variables_initializer())

		# Start training 
		for e in xrange(start_epoch + 1, start_epoch + epoch + 1):

			X_train, y_train, X_val, y_val = select_val_set(X_input, y_input)
			X_train, y_train = remove_background_3d_with_label(X_train, y_train)
			X_val, y_val = remove_background_3d_with_label(X_val, y_val)

			print X_train.shape, y_train.shape
			print X_val.shape, y_val.shape

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
				y_train_sample = y_train[index]

				_, loss_val, label_val, prob_val, num_pixels_val = self.sess.run([
					optimizer, loss, label, prob, num_pixels], 
					feed_dict = {X: X_train_sample, y: y_train_sample, 
					self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training : True})

				now = time.time()
				interval = now - self.last
				self.last = time.time()

				if i % 5 == 0:
					print 'ite {0} finished, loss: {1:.3e}, time consumed: {2:.3f}s'.format(i, loss_val, interval),\
						num_pixels_val.astype('int32'), np.sum(label_val.reshape((-1, 5)), axis = 0).astype('int32')

			# Make sure momery available
			if X_val.shape[0] > 32:
				np.random.shuffle(indices_val)
				index_val = indices_val[:32]
				prob_val, label_val, confusion_matrix_val = self.sess.run([prob, label, confusion_matrix], feed_dict = {
					X: X_val[index_val], y:y_val[index_val], self.dropout: dropout, self.is_training: False})
				y_out = y_val[index_val]
			else:
				prob_val, label_val, confusion_matrix_val = self.sess.run([prob, label, confusion_matrix], feed_dict = {
					X: X_val, y: y_val, self.dropout: dropout, self.is_training: False})
				y_out = y_val
			pre, IU, dice = self.evaluate(prob_val, label_val.reshape((-1, self.num_classes)))
			y_bin = np.bincount(y_out.reshape(-1))

			print 'Presicion: ', pre, ' mean: ', np.mean(pre)
			print 'IU: ', IU, ' mean: ', np.mean(IU)
			print 'dice: ', dice, ' mean: ', np.mean(dice)
			print 'y: ', y_bin

			if len(y_bin) == 4:
				print confusion_matrix_val[:4] * 1.0 / y_bin.reshape(-1, 1)
			else:
				print confusion_matrix_val * 1.0 / y_bin.reshape(-1, 1)

			print 'epoch {0} finished'.format(e)


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

	def evaluate(self, prob, label):

		pre = Precision(prob, label)
		IU = Intersection_Over_Union(prob, label)
		dice = Dice(prob, label)

		return pre, IU, dice


	def predict(self, model_name, X_test, y_test = None, dropout = 0.5):

		# Reconstruct the network
		X, y, label = self._get_input()
		cross_entropy, prob, confusion_matrix = self.build(X, label)
		loss, num_pixels = self._bootstrapping_loss(cross_entropy, prob, label)

		global_step = tf.get_variable(
        	'global_step', [],
        	initializer=tf.constant_initializer(0), trainable=False)

		saver = tf.train.Saver()
		print 'loading from model' + model_name 
		saver.restore(self.sess, './models/' + model_name + '.ckpt')

		if y_test is not None:

			if len(X_test.shape) == 5:
				mean_pre = np.zeros([X_test.shape[0]])
				mean_IU = np.zeros([X_test.shape[0]])
				mean_dice = np.zeros([X_test.shape[0]])
				for i in xrange(X_test.shape[0]):
					label_val = np.empty([0, 5])
					prob_val = np.empty([0, 5])
					for j in xrange(X_test.shape[1] / 31):
						label_val_t, prob_val_t, confusion_matrix_val = self.sess.run([label, prob, confusion_matrix], feed_dict = {
							X: X_test[i, j * 31:(j+1) * 31], y: y_test[i, j * 31:(j+1) * 31], self.dropout: dropout, self.is_training: False})
						label_val_t = label_val_t.reshape((-1, self.num_classes))
						label_val = np.concatenate([label_val, label_val_t], axis = 0)
						prob_val = np.concatenate([prob_val, prob_val_t], axis = 0)

					pre, IU, dice = self.evaluate(prob_val, label_val)

					print i
					print pre, np.mean(pre)
					print IU, np.mean(IU)
					print dice, np.mean(dice)

					mean_pre[i] = np.mean(pre)
					mean_IU[i] = np.mean(IU)
					mean_dice[i] = np.mean(dice)

				print np.mean(mean_pre), '\n', np.mean(mean_IU), '\n', np.mean(mean_dice)

					
			elif len(X_test.shape) == 4:
				label_val, prob_val, confusion_matrix_val = self.sess.run([label, prob, confusion_matrix], feed_dict = {
					X: X_test, y: y_test, self.dropout: dropout, self.is_training: False})
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
				result = np.empty([N, D, W, H])
				for i in xrange(N):
					prob_val = self.sess.run(prob, feed_dict = {X: X_test[i], self.dropout: dropout, self.is_training: False})
					result[i] = np.argmax(prob_val.reshape(D, W, H, C), axis = -1)
				return result

			elif len(X_test.shape) == 4:
				prob_val = self.sess.run(prob, feed_dict = {
					X: X_test, self.dropout: dropout, self.is_training: False})
				result = np.argmax(prob_val.reshape(D, W, H, C), axis = -1)
				return result
			else:
				print "input's dimention error!"
				exit(1)


	def multi_gpu_train(self, X_input, y_input, 
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
							cross_entropy, prob, _ = self.build(X, label)
							loss, num_pixels = self._bootstrapping_loss(cross_entropy, prob, label)

							# add to collections
							tf.add_to_collection('Xs', X)
							tf.add_to_collection('ys', y)
							tf.add_to_collection('labels', tf.reshape(label, [-1, 5]))
							tf.add_to_collection('losses', loss)
							tf.add_to_collection('probs', prob)
							tf.add_to_collection('num_pixels', num_pixels)

							tf.get_variable_scope().reuse_variables()
							grads = optimizer.compute_gradients(loss)
							tower_grads.append(grads)

			total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
			total_prob = tf.concat(tf.get_collection('probs'), axis = 0, name = 'total_loss')
			total_label = tf.concat(tf.get_collection('labels'), axis = 0, name = 'total_label')
			total_confusion_matrix = self.confusion_matrix(total_prob, total_label)
			total_num_pixels = tf.add_n(tf.get_collection('num_pixels'), name = 'total_num_pixels')

			grads = self._average_gradients(tower_grads)
			apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

			variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
			variables_averages_op = variable_averages.apply(tf.trainable_variables())

			train_op = tf.group(apply_gradient_op, variables_averages_op)

			saver = tf.train.Saver()

			# judge whether to restore from saved models
			if restore:
				print 'loading from model' + model_name
				start_epoch = np.int(model_name.split('_')[-1])
				saver.restore(self.sess, './models/' + model_name + '.ckpt')
			else:
				start_epoch = 0
				self.sess.run(tf.global_variables_initializer())

			# Start training 
			for e in xrange(epoch):

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

					_, loss_val, prob_val, label_val, num_pixels_val = self.sess.run(
						[train_op, total_loss, total_prob, total_label, total_num_pixels], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_list_val), tuple(tf.get_collection('ys')): tuple(y_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: True})

					now = time.time()
					interval = now - self.last
					self.last = time.time()

					print 'ite {0} finished, loss: {1:.3e}, time consumed: {2:.3f}s'.format(i, loss_val, interval),\
						num_pixels_val.astype('int32'), np.sum(label_val.reshape((-1, self.num_classes)), axis = 0).astype('int32')

				# Now it's time to evaluate on the validation data set
				if X_val.shape[0] > 32 * num_gpu:
					X_val_list_val = []
					y_val_list_val = []
					for i in xrange(num_gpu):
						X_val_list_val.append(X_val[i * 32: (i + 1) * 32])
						y_val_list_val.append(y_val[i * 32: (i + 1) * 32])
					prob_val, label_val, confusion_matrix_val = self.sess.run([total_prob, total_label, total_confusion_matrix], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_val_list_val), tuple(tf.get_collection('ys')): tuple(y_val_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: False})
					y_out = y_val[index_val]
				else:
					X_val_list_val = []
					y_val_list_val = []
					for i in xrange(num_gpu):
						X_val_list_val.append(X_val[i * X_val.shape[0] // num_gpu:(i + 1) * X_val.shape[0] // num_gpu])
						y_val_list_val.append(y_val[i * y_val.shape[0] // num_gpu:(i + 1) * y_val.shape[0] // num_gpu])
					prob_val, label_val, confusion_matrix_val = self.sess.run([total_prob, total_label, total_confusion_matrix], feed_dict = {
						tuple(tf.get_collection('Xs')): tuple(X_val_list_val), tuple(tf.get_collection('ys')): tuple(y_val_list_val), 
						self.dropout: dropout, self.N_worst: N_worst, self.thre: thre, self.is_training: False})
					y_out = y_val
				
				pre, IU, dice = self.evaluate(prob_val, label_val.reshape((-1, self.num_classes)))
				y_bin = np.bincount(y_out.reshape(-1))

				print 'Presicion: ', pre, ' mean: ', np.mean(pre)
				print 'IU: ', IU, ' mean: ', np.mean(IU)
				print 'dice: ', dice, ' mean: ', np.mean(dice)
				print 'y: ', y_bin

				if len(y_bin) == 4:
					print confusion_matrix_val[:4] * 1.0 / y_bin.reshape(-1, 1)
				else:
					print confusion_matrix_val * 1.0 / y_bin.reshape(-1, 1)

				print 'epoch {0} finished'.format(e)

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

	def multi_gpu_predict(self, model_name, X_test, y_test = None, dropout = 0.5):

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
							cross_entropy, prob, _ = self.build(X, label)
							loss, num_pixels = self._bootstrapping_loss(cross_entropy, prob, label)

							# add to collections
							tf.add_to_collection('Xs', X)
							tf.add_to_collection('ys', y)
							tf.add_to_collection('labels', tf.reshape(label, [-1, 5]))
							tf.add_to_collection('losses', loss)
							tf.add_to_collection('probs', prob)
							tf.add_to_collection('num_pixels', num_pixels)

							tf.get_variable_scope().reuse_variables()
							grads = optimizer.compute_gradients(loss)
							tower_grads.append(grads)

			total_loss = tf.add_n(tf.get_collection('losses'), name = 'total_loss')
			total_prob = tf.concat(tf.get_collection('probs'), axis = 0, name = 'total_loss')
			total_label = tf.concat(tf.get_collection('labels'), axis = 0, name = 'total_label')
			total_confusion_matrix = self.confusion_matrix(total_prob, total_label)
			total_num_pixels = tf.add_n(tf.get_collection('num_pixels'), name = 'total_num_pixels')

			grads = self._average_gradients(tower_grads)
			apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

			variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
			variables_averages_op = variable_averages.apply(tf.trainable_variables())

			train_op = tf.group(apply_gradient_op, variables_averages_op)

			saver = tf.train.Saver()
			print 'loading from model' + model_name 
			saver.restore(self.sess, './models/' + model_name + '.ckpt')

			if y_test is not None:

				if len(X_test.shape) == 5:
					mean_pre = np.zeros([X_test.shape[0]])
					mean_IU = np.zeros([X_test.shape[0]])
					mean_dice = np.zeros([X_test.shape[0]])
					
					# caculate every 3D image 
					for i in xrange(X_test.shape[0]):
						label_val = np.empty([0, 5])
						prob_val = np.empty([0, 5])

						batch_size_test = X_val.shape[0] // num_gpu
						X_test_list_val = []
						y_test_list_val = []
						for j in xrange(num_gpu):
							X_test_list_val.append(X_test[i * batch_size_test: (i + 1) * batch_size_test])
							y_test_list_val.append(y_test[i * batch_size_test: (i + 1) * batch_size_test])
						prob_val_t, label_val_t, confusion_matrix_val = self.sess.run([
							total_prob, total_label, total_confusion_matrix], feed_dict = {
							tuple(tf.get_collection('Xs')): tuple(X_test_list_val), tuple(tf.get_collection('ys')): tuple(y_test_list_val), 
							self.dropout: dropout, self.is_training: False})

						label_val_t = label_val_t.reshape((-1, self.num_classes))
						label_val = np.concatenate([label_val, label_val_t], axis = 0)
						prob_val = np.concatenate([prob_val, prob_val_t], axis = 0)

						pre, IU, dice = self.evaluate(prob_val, label_val)

						print i
						print pre, np.mean(pre)
						print IU, np.mean(IU)
						print dice, np.mean(dice)

						mean_pre[i] = np.mean(pre)
						mean_IU[i] = np.mean(IU)
						mean_dice[i] = np.mean(dice)

					print np.mean(mean_pre), '\n', np.mean(mean_IU), '\n', np.mean(mean_dice)

						
				elif len(X_test.shape) == 4:
					batch_size_test = X_val.shape[0] // num_gpu
					X_test_list_val = []
					y_test_list_val = []
					for j in xrange(num_gpu):
						X_test_list_val.append(X_test[i * batch_size_test: (i + 1) * batch_size_test])
						y_test_list_val.append(y_test[i * batch_size_test: (i + 1) * batch_size_test])
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
							X_test_list.append(X_test[j * D // 3:(j + 1) * D // 3])
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


	def _tower_loss(self, X, label, scope):

		loss, _, _ = self.build(X, label)
		tf.add_to_collection('losses', loss)

		# Assemble all of the losses for the current tower only.
		losses = tf.get_collection('losses', scope)

		# Calculate the total loss for the current tower.
		total_loss = tf.add_n(losses, name='total_loss')

		return total_loss


	def _average_gradients(self, tower_grads):
		"""Calculate the average gradient for each shared variable across all towers.

		Note that this function provides a synchronization point across all towers.

		Args:
		tower_grads: List of lists of (gradient, variable) tuples. The outer list
		  is over individual gradients. The inner list is over the gradient
		  calculation for each tower.
		Returns:
		 List of pairs of (gradient, variable) where the gradient has been averaged
		 across all towers.
		"""
		average_grads = []
		for grad_and_vars in zip(*tower_grads):
			# Note that each grad_and_vars looks like the following:
			# ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
			grads = []
			for g, _ in grad_and_vars:
				# Add 0 dimension to the gradients to represent the tower.
				expanded_g = tf.expand_dims(g, 0)

				# Append on a 'tower' dimension which we will average over below.
				grads.append(expanded_g)

			# Average over the 'tower' dimension.
			grad = tf.concat(axis=0, values=grads)
			grad = tf.reduce_mean(grad, 0)

			# Keep in mind that the Variables are redundant because they are shared
			# across towers. So .. we will just return the first tower's pointer to
			# the Variable.
			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
		return average_grads

	def _get_input(self):

		# with tf.name_scope(name):	
		X = tf.placeholder(tf.float32, [None, self.W, self.H, self.C], name = 'X_train')
		y = tf.placeholder(tf.uint8, [None, self.W, self.H], name = 'y_trian')
		label = tf.one_hot(y, self.num_classes)
		return X, y, label

	def _weighted_loss(self, cross_entropy, label):

		count = tf.bincount(tf.cast(tf.reshape(self.y_train, [-1]), tf.int32), minlength = 5)
		class_weights = tf.constant([1, 1, 1, 1, 1], dtype = tf.float32)
		class_weights = class_weights / tf.cast(count, tf.float32) 
		weights = tf.reduce_sum(label * class_weights, axis = 1)

		loss = tf.reduce_sum(cross_entropy * weights) / 5
		return loss

	def _self_weighted_loss(self, cross_entropy, label):

		class_weights = tf.cast(tf.constant([1, 1, 1, 1, 1]), tf.float32)
		self.softmax_weights = unbalanced_layer(class_weights, self.num_classes, 'unbalance_1')
		weights = tf.reduce_sum(label * self.softmax_weights, axis = 1)
		
		loss = tf.reduce_mean(cross_entropy * weights * 5)
		return loss, cross_entropy * weights * 5


	def _bootstrapping_loss(self, cross_entropy, prob, label):
	
		'''
		try to find worst N pixel in every class and caculate the average loss
		'''
		loss_matrix = tf.reshape(cross_entropy, [-1, 1]) * tf.reshape(tf.cast(label, tf.float32), [-1, 5])
		loss_matrix_t = tf.transpose(loss_matrix)

		condition_matrix = tf.cast(prob < self.thre, tf.float32) * tf.reshape(label, [-1, 5])
		condition_matrix_t = tf.transpose(condition_matrix)

		worst_n_matrix, _ = tf.nn.top_k(loss_matrix_t * condition_matrix_t, k = self.N_worst)

		count = tf.reduce_sum(condition_matrix_t, axis = 1)
		num_pixels = tf.cast(tf.minimum(tf.cast(count, tf.int32), self.N_worst), tf.float32, name = 'minimum')

		# loss = tf.reduce_mean(tf.reduce_sum(worst_n_matrix, axis = 1) / self.num_pixels)
		loss = tf.reduce_sum(worst_n_matrix) / tf.reduce_sum(num_pixels)
		return loss, num_pixels


	def _print_variable_by_name(self, name):

		g = tf.get_default_graph()
		t = g.get_tensor_by_name(name)

		print name, self.sess.run(t)

	def _print_global_variables(self, ifall = False):

		vs = tf.global_variables()
		print [v.name for v in vs] 

		if ifall:
			print sess.run(vs)




class FCNN_BASIC(FCNN_2D):

	def build(self, X, y):

		conv1_1 = conv_layer_res(X, [3, 3, 4, 64], [1, 1, 1, 1], 'conv1_1', if_relu = True)
		conv1_2 = conv_layer_res(conv1_1, [3, 3, 64, 64], [1, 1, 1, 1], 'conv1_2', if_relu = True)
		pool1 = max_pool_layer(conv1_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool1')

		conv2_1 = conv_layer_res(pool1, [3, 3, 64, 128], [1, 1, 1, 1], 'conv2_1', if_relu = True)
		conv2_2 = conv_layer_res(conv2_1, [3, 3, 128, 128], [1, 1, 1, 1], 'conv2_2', if_relu = True)
		pool2 = max_pool_layer(conv2_2, [1, 2, 2, 1], [1, 2, 2, 1], name='pool2')

		conv3_1 = conv_layer_res(pool2, [3, 3, 128, 256], [1, 1, 1, 1], 'conv3_1', if_relu = True)
		conv3_2 = conv_layer_res(conv3_1, [3, 3, 256, 256], [1, 1, 1, 1], 'conv3_2', if_relu = True)
		conv3_3 = conv_layer_res(conv3_2, [3, 3, 256, 256], [1, 1, 1, 1], 'conv3_3', if_relu = True)
		pool3 = max_pool_layer(conv3_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool3')

		conv4_1 = conv_layer_res(pool3, [3, 3, 256, 512], [1, 1, 1, 1], 'conv4_1', if_relu = True)
		conv4_2 = conv_layer_res(conv4_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv4_2', if_relu = True)
		conv4_3 = conv_layer_res(conv4_2, [3, 3, 512, 512], [1, 1, 1, 1], 'conv4_3', if_relu = True)
		pool4 = max_pool_layer(conv4_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool4')

		conv5_1 = conv_layer_res(pool4, [3, 3, 512, 512], [1, 1, 1, 1], 'conv5_1', if_relu = True)
		conv5_2 = conv_layer_res(conv5_1, [3, 3, 512, 512], [1, 1, 1, 1], 'conv5_2', if_relu = True)
		conv5_3 = conv_layer_res(conv5_2, [3, 3, 512, 512], [1, 1, 1, 1], 'conv5_3', if_relu = True)
		pool5 = max_pool_layer(conv5_3, [1, 2, 2, 1], [1, 2, 2, 1], name='pool5')

		fc_1 = tf.nn.dropout(conv_layer_res(pool5, [3, 3, 512, 1024], [1, 1, 1, 1], 'fc_1', if_relu = True), self.dropout)
		fc_2 = tf.nn.dropout(conv_layer_res(fc_1, [1, 1, 1024, 1024], [1, 1, 1, 1], 'fc_2', if_relu = True), self.dropout)
		fc_3 = conv_layer_res(fc_2, [1, 1, 1024, self.num_classes], [1, 1, 1, 1], 'fc_3')

		# Now we start upsampling and skip layer connections.
		img_shape = tf.shape(X)
		dconv3_shape = tf.stack([img_shape[0], img_shape[1], img_shape[2], self.num_classes])
		upsample_1 = upsample_layer(fc_3, dconv3_shape, self.num_classes, 'upsample_1', 32)

		skip_1 = skip_layer_connection(pool4, 'skip_1', 512, stddev=0.00001)
		upsample_2 = upsample_layer(skip_1, dconv3_shape, self.num_classes, 'upsample_2', 16)

		skip_2 = skip_layer_connection(pool3, 'skip_2', 256, stddev=0.0001)
		upsample_3 = upsample_layer(skip_2, dconv3_shape, self.num_classes, 'upsample_3', 8)

		logits = tf.add(upsample_3, tf.add(2 * upsample_2, 4 * upsample_1))

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

	net = FCNN_BASIC(input_shape = (240, 240, 4), num_classes = 5)
	net.train(X, y, model_name = 'model_vggfcn_1_49',
	 batch_size = 20, learning_rate = 5e-5, epoch = 100, restore = True, N_worst = 5e5, thre = 0.7)


