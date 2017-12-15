import numpy as np 
from image2array import get_file_lists, get_data

HGG_base = './data/Brain_Tumor/BRATS2015_Training/HGG/'
LGG_base = './data/Brain_Tumor/BRATS2015_Training/LGG/'


def remove_background_3d_with_label(img, label):
	'''
	Cut the pieces of image in (W. H, C) where all the pixels are labeled with 0 
	from the img in (N, W, H, C)
	And the labels are in shape (N, W, H)
	'''
	N, W, H, _ = img.shape
	zeros = np.zeros((W, H))

	exp = np.all(np.all(label == zeros, axis = 1), axis = 1)

	index = np.where(exp == False)[0]
	return img[index], label[index]

def remove_background_3d_with_label_4_arg(img, label_1, label_2, label_3):
	'''
	Cut the pieces of image in (W. H, C) where all the pixels are labeled with 0 
	from the img in (N, W, H, C)
	And the labels are in shape (N, W, H)
	'''
	N, W, H, _ = img.shape
	zeros = np.zeros((W, H))

	exp = np.all(np.all(label_1 == zeros, axis = 1), axis = 1)

	index = np.where(exp == False)[0]
	return img[index], label_1[index], label_2[index], label_3[index]



def remove_background_3d(img, width, height, depth):
	
	'''
	remove the voxels whose value is 0
	deprecated to call...
	'''

	W, H, D, _ = img.shape
	W_index, H_index, D_index, _ =	np.where(img > 0)
	
	W_max = np.amax(W_index)
	W_min = np.amin(W_index)
	H_max = np.amax(H_index)
	H_min = np.amin(H_index)
	D_max = np.amax(D_index)
	D_min = np.amin(D_index)

	print D_max, D_min

	if W_max - W_min > width or H_max - H_min > height or D_max - D_min > depth:
		print 'width or height or depth may be not proper'
		exit(1)
	else:
		W_center = (W_max + W_min) / 2
		H_center = (H_max + H_min) / 2
		D_center = (D_max + D_min) / 2
		return img[W_center - width / 2 : W_center + width / 2, H_center - height / 2 : H_center + height / 2, 0 : depth, :],\
			 W_center, H_center, D_center,

def remove_background_2d(img, width, height):
	'''
	remove the pixels whose value is 0 as more as possible
	'''
	W, H, _ = img.shape
	W_index, H_index, _ = np.where(img > 0)

	if len(W_index) == 0:
		return img[:width, :height, :]

	else:
		W_max = np.amax(W_index)
		W_min = np.amin(W_index)
		H_max = np.amax(H_index)
		H_min = np.amin(H_index)

		if W_max - W_min > width or H_max - H_min > height:
			print 'width or height or depth may be not proper'
			exit(1)

		else:
			W_center = (W_max + W_min) / 2
			H_center = (H_max + H_min) / 2

			return img[W_center - width / 2 : W_center + width / 2, H_center - height / 2 : H_center + height / 2, :]


def one_hot_label(label):
	dim = len(label.shape)
	label_0 = label == 0
	label_1 = label == 1
	label_2 = label == 2
	label_3 = label == 3
	label_4 = label == 4
	one_hot = np.stack((label_0, label_1, label_2, label_3, label_4), axis = dim)
	return one_hot


if __name__ == '__main__':

	file_lists, label_list = get_file_lists(HGG_base)
	data_np, label_np = get_data(file_lists[2:3], label_list[2:3])

	img, W_center, H_center, D_center = remove_back_ground(data_np[0], 180, 180, 136)
	print img.shape, W_center, H_center, D_center
	# img = data_np.transpose((0, 4, 1, 2, 3))[0, 0]
	# print img.shape
	# sitk_imshow(img)
