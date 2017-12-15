import numpy as np 
from medpy.io import load, header
import matplotlib.pyplot as plt

HGG_base = '/home/ff/data/Brain_Tumor/BRATS2015_Training/HGG/'
LGG_base = '/home/ff/data/Brain_Tumor/BRATS2015_Training/LGG/'

if __name__ == '__main__':

	# print mha.new(input_file = HGG_base + 'brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512/VSD.Brain.XX.O.MR_Flair.54512.mha').data.shape  
	data, _ = load(HGG_base + 'brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512/VSD.Brain.XX.O.MR_Flair.54512.mha')
	# header.get_pixel_spacing(meta)
	# header.get_offset(meta)
	# print data.shape
	# print data.dtype

	# x, y, c = np.where(data != 0)
	# for i in c:
	# 	print i

	# snap = data.transpose(2, 0, 1)[76:79]

	# print snap.shape

	# snap = snap.transpose(1, 2, 0)

	snap = data[:, :, 76]

	plt.figure(figsize = (8, 8))

	plt.subplot(2, 2, 1)
	plt.title('gray_r')
	plt.imshow((snap *1.0 / np.amax(snap) * 255).astype('int8'), cmap = 'gray_r')

	plt.subplot(2, 2, 2)
	plt.title('gray')
	plt.imshow((snap *1.0 / np.amax(snap) * 255).astype('int8'), cmap = 'gray')

	plt.subplot(2, 2, 3)
	plt.title('bone')
	plt.imshow((snap *1.0 / np.amax(snap) * 255).astype('int8'), cmap = 'bone')

	plt.subplot(2, 2, 4)
	plt.title('hot')
	plt.imshow((snap *1.0 / np.amax(snap) * 255).astype('int8'), cmap = 'hot')
	
	plt.show()
	