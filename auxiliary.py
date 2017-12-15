import tensorflow as tf 
import numpy as np


def get_variable_from_model(model_path, name):

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_path + '.meta')
		saver.restore(sess, model_path)

		g = tf.get_default_graph()
		t = g.get_tensor_by_name(name)
		print sess.run(t)

def get_variables_from_model(model_path, if_print_content = True):

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_path + '.meta')
		saver.restore(sess, model_path)

		vs = tf.global_variables()
		print [v.name for v in vs] 

		if if_print_content:
			print sess.run(vs)

def three_to_two(X, y):

	print 'Start transposing...'
	X_load = X.reshape((-1, 240, 240, 4)).astype('float32')
	y_load = y.reshape((-1, 240, 240)).astype('int8')

	print X_load.shape, y_load.shape

	return X_load, y_load

def three_to_two_4_arg(X, y_1, y_2, y_3):

	print 'Start transposing...'
	X_load = X.reshape((-1, 240, 240, 4)).astype('float32')
	y_1_load = y_1.reshape((-1, 240, 240)).astype('int8')
	y_2_load = y_2.reshape((-1, 240, 240)).astype('int8')
	y_3_load = y_3.reshape((-1, 240, 240)).astype('int8')

	# print X_load.shape, y_load.shape

	return X_load, y_1_load, y_2_load, y_3_load


def select_val_set(X, y):

	print 'Select validation data sets'
	N = X.shape[0]
	val = np.random.randint(N)

	X_val = X[val].reshape(1, *(X[val].shape))
	y_val = y[val].reshape(1, *(y[val].shape))
	X_val, y_val = three_to_two(X_val, y_val)

	train = np.delete(np.arange(N), val)
	X_train = X[train]
	y_train = y[train]

	X_train, y_train = three_to_two(X_train, y_train)

	print X_train.shape, y_train.shape, X_val.shape, y_val.shape

	return X_train, y_train, X_val, y_val

def select_val_set_4_arg(X, y_1, y_2, y_3):

	print 'Select validation data sets'
	N = X.shape[0]
	val = np.random.randint(N)

	X_val = X[val].reshape(1, *(X[val].shape))
	y_1_val = y_1[val].reshape(1, *(y_1[val].shape))
	y_2_val = y_2[val].reshape(1, *(y_2[val].shape))
	y_3_val = y_3[val].reshape(1, *(y_3[val].shape))
	X_val, y_1_val, y_2_val, y_3_val = three_to_two_4_arg(X_val, y_1_val, y_2_val, y_3_val)

	train = np.delete(np.arange(N), val)
	X_train = X[train]
	y_1_train = y_1[train]
	y_2_train = y_2[train]
	y_3_train = y_3[train]

	X_train, y_1_train, y_2_train, y_3_train = three_to_two_4_arg(X_train, y_1_train, y_2_train, y_3_train)

	# print X_train.shape, y_1_train.shape, y_2_train.shape, y_3_train.shape, X_val.shape, y_1_val.shape, y_2_val.shape, y_3_val.shape

	return X_train, y_1_train, y_2_train, y_3_train, X_val, y_1_val, y_2_val, y_3_val


def select_val_set_3d(X, y):
	print 'Selecting validation data set...'
	N = X.shape[0]
	val = np.random.randint(N)

	X_val = X[val:val+1].transpose((0, 4, 2, 3, 1)).astype('float32')
	y_val = y[val:val+1].transpose((0, 3, 1, 2)).astype('float32')

	train = np.delete(np.arange(N), val)
	X_train = X[train].transpose((0, 4, 2, 3, 1)).astype('float32')
	y_train = y[train].transpose((0, 3, 1, 2)).astype('float32')

	return X_train, y_train, X_val, y_val


def Precision(prob, label):
	'''
	TP / (TP + FN)
	'''

	t_0 = np.array([1, 0, 0, 0, 0])
	t_1 = np.array([0, 1, 0, 0, 0])
	t_2 = np.array([0, 0, 1, 0, 0])
	t_3 = np.array([0, 0, 0, 1, 0])
	t_4 = np.array([0, 0, 0, 0, 1])
	index_t_back = np.where(np.all(label == t_0, axis = 1))
	index_t_comp = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_2, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_core = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_enh = np.where(np.all(label == t_4, axis = 1))

	# 1-D array of the shape (pixel_num, )
	argmax_back = np.argmax(prob[index_t_back], axis = 1)
	argmax_comp = np.argmax(prob[index_t_comp], axis = 1)
	argmax_core = np.argmax(prob[index_t_core], axis = 1)
	argmax_enh = np.argmax(prob[index_t_enh], axis = 1)

	acc_back = np.sum(argmax_back == 0) * 1.0 / len(argmax_back)
	acc_comp = (len(argmax_comp) - np.sum(argmax_comp == 0)) * 1.0 / len(argmax_comp)
	acc_core = (np.sum(argmax_core == 1) + np.sum(argmax_core == 3) + np.sum(argmax_core == 4)) * 1.0 / len(argmax_core)
	acc_enh = np.sum(argmax_enh == 4) * 1.0 / (len(argmax_enh) + 1)

	acc = np.array([acc_back, acc_comp, acc_core, acc_enh])
	return acc


def Intersection_Over_Union(prob, label):
	'''
	TP / (TP + FP + FN)
	'''
	
	t_0 = np.array([1, 0, 0, 0, 0])
	t_1 = np.array([0, 1, 0, 0, 0])
	t_2 = np.array([0, 0, 1, 0, 0])
	t_3 = np.array([0, 0, 0, 1, 0])
	t_4 = np.array([0, 0, 0, 0, 1])
	index_t_back = np.where(np.all(label == t_0, axis = 1))
	index_t_comp = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_2, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_core = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_not_core = np.where(np.all(label == t_0, axis = 1)|np.all(label == t_2, axis = 1))
	index_t_enh = np.where(np.all(label == t_4, axis = 1))
	index_t_not_enh = np.where(np.all(label != t_4, axis = 1))

	# 1-D array of the shape (pixel_num, )
	argmax_back = np.argmax(prob[index_t_back], axis = 1)
	argmax_comp = np.argmax(prob[index_t_comp], axis = 1)
	argmax_core = np.argmax(prob[index_t_core], axis = 1)
	argmax_not_core = np.argmax(prob[index_t_not_core], axis = 1)
	argmax_enh = np.argmax(prob[index_t_enh], axis = 1)
	argmax_not_enh = np.argmax(prob[index_t_not_enh], axis = 1)

	IU_back = np.sum(argmax_back == 0) * 1.0 / (len(argmax_back) + np.sum(argmax_comp == 0))
	IU_comp = np.sum(argmax_comp != 0) * 1.0 / (len(argmax_comp) + np.sum(argmax_back != 0))
	IU_core = np.sum((argmax_core == 1) | (argmax_core == 3) | (argmax_core == 4)) * 1.0 / (len(argmax_core) + np.sum((argmax_not_core != 0)
	 & (argmax_not_core != 2))) 
	IU_enh = np.sum(argmax_enh == 4) * 1.0 / (len(argmax_enh) + np.sum(argmax_not_enh == 4) + 1)

	IU = np.array([IU_back, IU_comp, IU_core, IU_enh])
	return IU


def Dice(prob, label):
	'''
	2 * TP / (2 * TP + FP + FN)
	'''
	
	t_0 = np.array([1, 0, 0, 0, 0])
	t_1 = np.array([0, 1, 0, 0, 0])
	t_2 = np.array([0, 0, 1, 0, 0])
	t_3 = np.array([0, 0, 0, 1, 0])
	t_4 = np.array([0, 0, 0, 0, 1])
	index_t_back = np.where(np.all(label == t_0, axis = 1))
	index_t_comp = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_2, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_core = np.where(np.all(label == t_1, axis = 1)|np.all(label == t_3, axis = 1)|np.all(label == t_4, axis = 1))
	index_t_not_core = np.where(np.all(label == t_0, axis = 1)|np.all(label == t_2, axis = 1))
	index_t_enh = np.where(np.all(label == t_4, axis = 1))
	index_t_not_enh = np.where(np.all(label != t_4, axis = 1))

	# 1-D array of the shape (pixel_num, )
	argmax_back = np.argmax(prob[index_t_back], axis = 1)
	argmax_comp = np.argmax(prob[index_t_comp], axis = 1)
	argmax_core = np.argmax(prob[index_t_core], axis = 1)
	argmax_not_core = np.argmax(prob[index_t_not_core], axis = 1)
	argmax_enh = np.argmax(prob[index_t_enh], axis = 1)
	argmax_not_enh = np.argmax(prob[index_t_not_enh], axis = 1)

	dice_back = np.sum(argmax_back == 0) * 2.0 / (len(argmax_back) + np.sum(argmax_back == 0) + np.sum(argmax_comp == 0))
	dice_comp = np.sum(argmax_comp != 0) * 2.0 / (len(argmax_comp) + np.sum(argmax_comp != 0) + np.sum(argmax_back != 0))
	dice_core = np.sum((argmax_core == 1) | (argmax_core == 3) | (argmax_core == 4)) * 2.0 /\
	(len(argmax_core) + np.sum((argmax_core == 1) | (argmax_core == 3) | (argmax_core == 4)) + 
		np.sum((argmax_not_core != 0) & (argmax_not_core != 2)))
	dice_enh = np.sum(argmax_enh == 4) * 2.0 / (len(argmax_enh) + np.sum(argmax_enh == 4) + np.sum(argmax_not_enh == 4) + 1)

	return np.array([dice_back, dice_comp, dice_core, dice_enh])


def binary_dice(prob, label):
	'''
	prob.shape = [num_pixel], label.shape = [num_pixel]	
	'''

	pred = (prob > 0.5).astype('int8')
	return np.sum(pred * label) * 2.0 / (np.sum(pred) + np.sum(label))

def binary_IoU(prob, label):
	
	pred = (prob > 0.5).astype('int8')
	return np.sum(pred * label) * 1.0 / (np.sum(pred) + np.sum(label) - np.sum(pred * label))

def binary_recall(prob, label):

	pred = (prob > 0.5).astype('int8')
	return np.sum(pred * label) * 1.0 / np.sum(label)



if __name__ == '__main__':

	get_variable_from_model('./models/model_VGG_unbal_labels.ckpt', 'unbal_1/weights:0')