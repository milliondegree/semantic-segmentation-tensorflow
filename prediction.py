import tensorflow as tf 
import numpy as np
from fcnn2d import FCNN_2D, Base, FCNN_BASIC
from auxiliary import *
from Unet_basic import UNET, WNET
from resnet import RESNET
from resunet import RES_UNET


if __name__ == '__main__':

	# print 'loading...'
	# f = np.load('./data/npz/LGG_new.npz')
	# X = f['X']
	# y_1 = f['y0']
	# y_2 = f['y1']
	# y_3 = f['y2']


	# net = WNET(input_shape = [240, 240, 4])
	# net.eval(X, y_1, y_2, y_3, 'model_wnet_1_74')


	print 'loading from LGG_train.npz'
	tmp = np.load(Base + '/LGG_train.npz')
	X = tmp['X']
	y = tmp['y']

	print X.shape, y.shape

	net = RES_UNET(input_shape = [240, 240, 4], num_classes = 5)
	net.predict('model_resunet_2_99', X, y)
	# net.predict(X, y, restore = True, model_name = 'model_vggfcn_1_99')
	# net.predict(X, y, restore = True, model_name = 'model_unet_1_99')

	# net = FCNN_2D(input_shape = [240, 240, 4], num_classes = 5)
	# predict = np.empty(shape = [0, 155, 240, 240], dtype = 'int16')
	# for i in xrange(X.shape[0]):
	# 	X_val, y_val = three_to_two(X[i].reshape(1, *(X[i].shape)), y[i].reshape(1, *(y[i].shape)))
	# 	acc_array, IU, dice, con, y_val, _ = net.predict(X_val, y_val, restore = True, model_name = 'model_VGG_LGG_boot_124')
	# 	print acc_array, np.mean(acc_array)
	# 	print IU, np.mean(IU)
	# 	print dice, np.mean(dice)
	# 	print np.bincount(y_val.reshape(-1)), i
	# 	print con
	# 	print 
	# 	result = net.predict(X_val, restore = True, model_name = 'model_VGG_LGG_boot_3_99')
	# 	predict = np.append(predict, result.astype('int16'))
	# 	print predict.shape

	# np.save(Base + '/LGG_predict0.npy', predict)
	