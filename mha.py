"""
This class reads and writes mha files (images or vector fields)
Author: Paolo Zaffino  (p.zaffino@unicz.it)
Rev 19
NOT TESTED ON PYTHON 3
"""

import numpy as np

class new():
	
	"""
	PUBLIC PARAMETERS:
	
	data=3D/4D matrix
	size=3D/4D matrix size
	spacing=voxel size
	offset=spatial offset of data data
	data_type='short', 'float' or 'uchar'
	direction_cosines=direction cosines of the raw image/vf
	
	 
	CONSTRUCTOR OVERLOADING:
	
	img=mha.new() # All the public parameters will be set to None
	img=mha.new(input_file='img.mha')
	img=mha.new(data=matrix, size=[512, 512, 80], spacing=[0.9, 0.9, 5], offset=[-240, -240, -160], data_type='short', direction_cosines=[1, 0, 0, 0, 1, 0, 0, 0, 1])
	
	
	PUBLIC METHODS:
	
	img.read_mha('file_name.mha')
	img.write_mha('file_name.mha')
	"""
	
	data=None
	size=None
	spacing=None
	offset=None
	data_type=None
	direction_cosines=None
	
######################## CONSTRUCTOR - START - #########################
	def __init__ (self, input_file=None, data=None, size=None, spacing=None, offset=None, data_type=None, direction_cosines=None):
		
		if input_file!=None and data==None and size==None and spacing==None and offset==None and data_type==None and direction_cosines==None:
			self.read_mha(input_file)
			
		elif input_file==None and data!=None and size!=None and spacing!=None and offset!=None and data_type!=None and direction_cosines!=None:
			self.data=data
			self.size=size
			self.spacing=spacing
			self.offset=offset
			self.data_type=data_type
			self.direction_cosines=direction_cosines
		
		elif input_file==None and data==None and size==None and spacing==None and offset==None and data_type==None and direction_cosines==None:
			pass
######################## CONSTRUCTOR - END - ###########################	

######################## READ_MHA - START - ############################
	def read_mha(self, fn):
		
		"""
		This method reads a mha file and assigns the data to the object parameters
		
		INPUT PARAMETER:
		fn=file name
		"""	
		
		if fn.endswith('.mha'): ## Check if the file extension is ".mha"
			
			f = open(fn,'rb')
			data='img' ## On default the matrix is considered to be an image
	
			## Read mha header
			for r in range(20):
				
				row=f.readline()
				print type(row[0])
				
				if row.startswith('TransformMatrix ='):
					row=row.split('=')[1].strip()
					self.direction_cosines=self._cast2int(map(float, row.split()))
				elif row.startswith('Offset ='):
					row=row.split('=')[1].strip()
					self.offset=self._cast2int(map(float, row.split()))
				elif row.startswith('ElementSpacing ='):
					row=row.split('=')[1].strip()
					self.spacing=self._cast2int(map(float, row.split()))
				elif row.startswith('DimSize ='):
					row=row.split('=')[1].strip()
					self.size=map(int, row.split())
				elif row.startswith('ElementNumberOfChannels = 3'):
					data='vf' ## The matrix is a vf
					self.size.append(3)
				elif row.startswith('ElementType ='):
					data_type=row.split('=')[1].strip()
				elif row.startswith('ElementDataFile ='):
					break
			
			## Read raw data
			self.data=''.join(f.readlines())
			f.close()

			# print type(self.data)
			# print len(self.data)
			l = len(self.data)
			print self.data[l - 1]
			print self.data[0]
			
			## Raw data from string to array
			if data_type == 'MET_SHORT':
				self.data=np.fromstring(self.data, dtype=np.int16)
				self.data_type = 'short'
			elif data_type == 'MET_FLOAT':
				self.data=np.fromstring(self.data, dtype=np.float32)
				self.data_type = 'float'
			elif data_type == 'MET_UCHAR':
				self.data=np.fromstring(self.data, dtype=np.uint8)
				self.data_type = 'uchar'


			
			## Reshape array
			if data == 'img':
				self.data=self.data.reshape(self.size[2],self.size[1],self.size[0]).T
			elif data == 'vf':
				self.data=self.data.reshape(self.size[2],self.size[1],self.size[0],3)
				self.data=self._shiftdim(self.data, 3).T
			
		elif not fn.endswith('.mha'): ## Extension file is not ".mha". It returns all null values
			raise NameError('The input file is not a mha file!')
######################### READ_MHA - END - #############################

######################## WRITE_MHA - START - ###########################
	def write_mha (self,fn):
		
		"""
		This method writes the object parameters in a mha file
		
		INPUT PARAMETER:
		fn=file name
		"""
		
		if fn.endswith('.mha'): ## Check if the file extension is ".mha"

                        ## Order the matrix in the proper way
                        self.data = np.array(self.data, order = "F")
			
			## Check if the input matrix is an image or a vf
			if self.data.ndim == 3:
				data='img'
			elif self.data.ndim == 4:
				data='vf'
			
			f=open(fn, 'wb')
			
			## Write mha header
			f.write('ObjectType = Image\n')
			f.write('NDims = 3\n')
			f.write('BinaryData = True\n')
			f.write('BinaryDataByteOrderMSB = False\n')
			f.write('CompressedData = False\n')
			f.write('TransformMatrix = '+str(self.direction_cosines).strip('()[]').replace(',','')+'\n')
			f.write('Offset = '+str(self.offset).strip('()[]').replace(',','')+'\n')
			f.write('CenterOfRotation = 0 0 0\n')
			f.write('AnatomicalOrientation = RAI\n')
			f.write('ElementSpacing = '+str(self.spacing).strip('()[]').replace(',','')+'\n')
			f.write('DimSize = '+str(self.size).strip('()[]').replace(',','')+'\n')
			if data == 'vf':
				f.write('ElementNumberOfChannels = 3\n')
				self.data=self._shiftdim(self.data, 3) ## Shift dimensions if the input matrix is a vf
			if self.data_type == 'short':
				f.write('ElementType = MET_SHORT\n')
			elif self.data_type == 'float':
				f.write('ElementType = MET_FLOAT\n')
			elif self.data_type == 'uchar':
				f.write('ElementType = MET_UCHAR\n')
			f.write('ElementDataFile = LOCAL\n')
			
			## Write matrix
			f.write(self.data)
			f.close()
			
		elif not fn.endswith('.mha'): ## File extension is not ".mha"
			raise NameError('The input file name is not a mha file!')
######################## WRITE_MHA - END - #############################

############ UTILITY FUNCTIONS, NOT FOR PUBLIC USE - START - ###########
	def _cast2int (self, l):
		l_new=[]
		for i in l:
			if i.is_integer(): l_new.append(int(i))
			else: l_new.append(i)
		return l_new
		
	_shiftdim = lambda self, x, n: x.transpose(np.roll(range(x.ndim), -n))
############# UTILITY FUNCTIONS, NOT FOR PUBLIC USE - END - ############
