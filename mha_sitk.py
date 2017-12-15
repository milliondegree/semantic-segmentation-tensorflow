import sys
import numpy as np 
from skimage import io
import SimpleITK as sitk
import matplotlib.pyplot as plt

HGG_base = '/home/ff/data/Brain_Tumor/BRATS2015_Training/HGG/'
LGG_base = '/home/ff/data/Brain_Tumor/BRATS2015_Training/LGG/'


def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    # nda = nda.astype('int8')
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize = figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda[:, :, 76], extent=extent, interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()


def sitk_imshow(img, axis = 2, slice = [55, 75, 95, 105], cmap = 'gray'):

	data = img

	if len(slice) == 4:
		a, b, c, d = slice
	else:
		raise ValueError('Length of the slice list should be %d' % 4)

	if axis == 0:
		snap0 = data[a]
		snap1 = data[b]
		snap2 = data[c]
		snap3 = data[d]
	elif axis == 1:
		snap0 = data[:, a, :]
		snap1 = data[:, b, :]
		snap2 = data[:, c, :]
		snap3 = data[:, d, :]
	elif axis == 2:
		snap0 = data[:, :, a]
		snap1 = data[:, :, b]
		snap2 = data[:, :, c]
		snap3 = data[:, :, d]
	else:
		raise ValueError('Axis should be 0 or 1 or 2')


	# try:
	# 	plt.figure(figsize = (8, 8))

	# 	plt.subplot(2, 2, 1)
	# 	plt.title('0')
	# 	plt.imshow((snap0 *1.0 / np.amax(snap) * 255).astype('int8'), cmap = cmap)

	# 	plt.subplot(2, 2, 2)
	# 	plt.title('1')
	# 	plt.imshow((snap1 *1.0 / np.amax(snap) * 255).astype('int8'), cmap = cmap)

	# 	plt.subplot(2, 2, 3)
	# 	plt.title('2')
	# 	plt.imshow((snap2 *1.0 / np.amax(snap) * 255).astype('int8'), cmap = cmap)

	# 	plt.subplot(2, 2, 4)
	# 	plt.title('3')
	# 	plt.imshow((snap3 *1.0 / np.amax(snap) * 255).astype('int8'), cmap = cmap)
		
	# 	plt.show()

	# except:
	# 	print 'plt error'

	plt.figure(figsize = (8, 8))

	plt.subplot(2, 2, 1)
	plt.title('0')
	plt.imshow((snap0 *1.0 / np.amax(snap0) * 255).astype('int8'), cmap = cmap)

	plt.subplot(2, 2, 2)
	plt.title('1')
	plt.imshow((snap1 *1.0 / np.amax(snap1) * 255).astype('int8'), cmap = cmap)

	plt.subplot(2, 2, 3)
	plt.title('2')
	plt.imshow((snap2 *1.0 / np.amax(snap2) * 255).astype('int8'), cmap = cmap)

	plt.subplot(2, 2, 4)
	plt.title('3')
	plt.imshow((snap3 *1.0 / np.amax(snap3) * 255).astype('int8'), cmap = cmap)
	
	plt.show()



if __name__ == '__main__':

	image = sitk.ReadImage(HGG_base + 'brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517/VSD.Brain_3more.XX.O.OT.54517.mha')
	data = sitk.GetArrayFromImage(image)

	noto = data[data > 0]
	l = []
	for i in noto:
		if not i in l:
			print i
			l.append(i)

	# data = data.transpose((2, 1, 0))
	# snap = data[:, :, 76]

	# index = np.where(snap > 0)
	# print snap[index]
	# print np.amax(snap)
	
	# plt.imshow((snap * 1.0 / np.amax(snap) * 255).astype('int8'))
	# plt.show()
