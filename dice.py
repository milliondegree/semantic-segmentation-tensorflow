import tensorflow as tf 
import numpy as np 
import time

from fcnn2d import FCNN_2D, Base
from resnet import RESNET
from resunet import RES_UNET
from layers import *
from auxiliary import *
from pre_process import *

class VGG_DICE(RES_UNET):

    def multi_binary_build(self, X, y):
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


        reshaped_logits = tf.reshape(logits, [-1, self.num_classes])
        reshaped_labels = tf.reshape(y, [-1, self.num_classes])
        prob = tf.nn.softmax(logits = reshaped_logits)
        prob_0 = tf.reduce_sum(prob[:, 1:5], axis = 1, keepdims = True)
        prob_1 = tf.reduce_sum(prob[:, 3:5], axis = 1, keepdims = True) + tf.reshape(prob[:, 1], [-1, 1])
        prob_2 = tf.reshape(prob[:, 4], [-1, 1])

        confusion_matrix = self.confusion_matrix(prob, reshaped_labels)
        # reshaped_logits_1 = tf.reshape(logits_1, [-1, 1])
        # reshaped_logits_2 = tf.reshape(logits_2, [-1, 1])
        # reshaped_logits_3 = tf.reshape(logits_3, [-1, 1])
        # reshaped_labels = tf.reshape(y, [-1, self.num_classes])

        # prob_1 = tf.sigmoid(reshaped_logits_1)
        # prob_2 = tf.sigmoid(reshaped_logits_2)
        # prob_3 = tf.sigmoid(reshaped_logits_3)

        return prob, prob_0, prob_1, prob_2, confusion_matrix
    

if __name__ == '__main__':
    print 'loading from HGG_train.npz...'
    f = np.load(Base + '/HGG_train2.npz')
    X = f['X']
    y = f['y']

    print X.shape, y.shape        

    # ans = raw_input('Do you want to continue? [y/else]: ')
    # if ans == 'y':
    net = RESNET(input_shape = (240, 240, 4), num_classes = 5)
    net.multi_gpu_train(X, y, model_name = 'model_vggdice_1', train_mode = 1, num_gpu = 1, 
     batch_size = 32, learning_rate = 5e-5, epoch = 100, restore = False, N_worst = 1e10, thre = 1.0)
 
