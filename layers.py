import tensorflow as tf 
import numpy as np 

# vgg_weights = np.load('./data/pretrained_weights/vgg16_weights.npz')


def bottle_layer(parent, channel_in, channel_1, channel_2, is_training, name):
    
    with tf.variable_scope(name):

        conv_1 = conv_layer_res(parent, [1, 1, channel_in, channel_1], [1, 1, 1, 1], 'conv_1')
        bn_1 = bn_layer(conv_1, is_training, 'bn_1')
        conv_1_relu = tf.nn.relu(bn_1, name = 'relu_1')
        conv_2 = conv_layer_res(conv_1_relu, [3, 3, channel_1, channel_1], [1, 1, 1, 1], 'conv_2')
        bn_2 = bn_layer(conv_2, is_training, 'bn_2')
        conv_2_relu = tf.nn.relu(bn_2, name = 'relu_2')
        conv_3 = conv_layer_res(conv_2_relu, [1, 1, channel_1, channel_2], [1, 1, 1, 1], 'conv_3')
        bn_3 = bn_layer(conv_3, is_training, 'bn_3')

        if channel_in != channel_2:
            shortcut = conv_layer_res(parent, [1, 1, channel_in, channel_2], [1, 1, 1, 1], 'shortcut')
        else:
            shortcut = parent

    return tf.nn.relu(bn_3 + shortcut)


def conv_norm_relu(parent, kernal_size, stride, is_training, name):
    
    conv = conv_layer_res(parent, kernal_size, stride, name)
    bn = bn_layer(conv, is_training, name)
    return tf.nn.relu(bn)


def stack_layer(parent, channel_in, channel_1, channel_2, is_training, name):

    with tf.variable_scope(name):

        conv_1 = conv_layer_res(parent, [3, 3, channel_in, channel_1], [1, 1, 1, 1], 'conv_1')
        bn_1 = bn_layer(conv_1, is_training, 'bn_1')
        conv_1_relu = tf.nn.relu(bn_1, name = 'relu_1')
        conv_2 = conv_layer_res(conv_1_relu, [3, 3, channel_1, channel_2], [1, 1, 1, 1], 'conv_2')
        bn_2 = bn_layer(conv_2, is_training, 'bn_2')

        if channel_in != channel_2:
            shortcut = conv_layer_res(parent, [1, 1, channel_in, channel_2], [1, 1, 1, 1], 'shortcut')
        else:
            shortcut = parent

    return tf.nn.relu(bn_2 + shortcut)


def stack_layer_3d(parent, channel_in, channel_1, channel_2, is_training, name):

    with tf.variable_scope(name):

        conv_1 = conv_layer_res_3d(parent, [3, 3, 3, channel_in, channel_1], [1, 1, 1, 1, 1], 'conv_1')
        bn_1 = bn_layer(conv_1, is_training, 'bn_1')
        conv_1_relu = tf.nn.relu(bn_1, name = 'relu_1')
        conv_2 = conv_layer_res_3d(conv_1_relu, [3, 3, 3, channel_1, channel_2], [1, 1, 1, 1, 1], 'conv_2')
        bn_2 = bn_layer(conv_2, is_training, 'bn_2')

        if channel_in != channel_2:
            shortcut = conv_layer_res_3d(parent, [1, 1, 1, channel_in, channel_2], [1, 1, 1, 1, 1], 'shortcut')
        else:
            shortcut = parent

    return tf.nn.relu(bn_2 + shortcut)




def stack_layer_not_act(parent, channel_in, channel_1, channel_2, is_training, name):

    with tf.variable_scope(name):

        conv_1 = conv_layer_res(parent, [3, 3, channel_in, channel_1], [1, 1, 1, 1], 'conv_1')
        bn_1 = bn_layer(conv_1, is_training, 'bn_1')
        conv_1_relu = tf.nn.relu(bn_1, name = 'relu_1')
        conv_2 = conv_layer_res(conv_1_relu, [3, 3, channel_1, channel_2], [1, 1, 1, 1], 'conv_2')
        bn_2 = bn_layer(conv_2, is_training, 'bn_2')

        if channel_in != channel_2:
            shortcut = conv_layer_res(parent, [1, 1, channel_in, channel_2], [1, 1, 1, 1], 'shortcut')
        else:
            shortcut = parent

    return bn_2 + shortcut



def conv_layer_res(parent, kernal_size, stride, name, if_bias = True, if_relu = False):

    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev = 0.0005)
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init)
        conv = tf.nn.conv2d(parent, weights, stride, padding = 'SAME')

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv 

        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias


def conv_layer_res_3d(parent, kernal_size, stride, name, if_bias = True, if_relu = False):

    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev = 0.0005)
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init)
        conv = tf.nn.conv3d(parent, weights, stride, padding = 'SAME')

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv 

        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias



def atrous_conv_layer(parent, kernal_size, rate, name, if_bias = True, if_relu = False):
    '''
    Implementation of atrous convolutional layer
    kernal_size = [H, W, in_C, out_C]
    '''
    with tf.variable_scope(name):
        init = tf.truncated_normal_initializer(stddev = 0.0005)
        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init)
        atrous_conv = tf.nn.atrous_conv2d(parent, weights, rate = rate, padding = 'SAME')

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init)
            conv_with_bias = tf.nn.bias_add(atrous_conv, bias)
        else:
            conv_with_bias = atrous_conv 

        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias

def atrous_tuning_layer(parent, kernal_name, bias_name, rate, name):
    '''
    Implementation of atrous convolutional layer with VGG16 parameters
    kernal_size = [H, W, in_C, out_C]
    '''
    with tf.variable_scope(name):
        weights = _get_kernel(kernel_name)
        init = tf.constant_initializer(value = kernel_weights, dtype=tf.float32)
        kernel = tf.get_variable(name = "weights", initializer=init, shape=kernel_weights.shape)
        atrous_conv = tf.nn.atrous_conv2d(parent, weights, rate = rate, padding='SAME')

        bias = _get_bias(bias_name)
        init = tf.constant_initializer(value=bias, dtype=tf.float32)
        biases = tf.get_variable(name="biases", initializer=init, shape=bias.shape)

        conv_with_bias = tf.nn.bias_add(atrous_conv, biases)
        conv_with_relu = tf.nn.relu(conv_with_bias)
    return conv_with_relu



def bn_layer(parent, is_training, name):

    with tf.variable_scope(name):
        shape = parent.shape
        param_shape = shape[-1:]

        pop_mean = tf.get_variable("mean", param_shape, initializer = tf.constant_initializer(0.0), trainable=False)
        pop_var = tf.get_variable("variance", param_shape, initializer = tf.constant_initializer(1.0), trainable=False)
        epsilon = 1e-4
        decay = 0.99

        scale = tf.get_variable('scale', param_shape, initializer = tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', param_shape, initializer = tf.constant_initializer(0.0))

        def True_fn():
            batch_mean, batch_var = tf.nn.moments(parent, list(range((len(shape) - 1))))

            train_mean = tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                output = tf.nn.batch_normalization(parent,
                batch_mean, batch_var, beta, scale, epsilon, name = name)
                return output

        def False_fn():
            output = tf.nn.batch_normalization(parent,
            pop_mean, pop_var, beta , scale, epsilon, name = name)
            return output

    return tf.cond(is_training, True_fn, False_fn)


def up_layer(parent, shape, output_channel, input_channel, upscale_factor, name):

    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], output_channel]
        filter_shape = [kernel_size, kernel_size, output_channel, input_channel]
        weights = _get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(parent, weights, output_shape,
                                        strides = strides, padding='SAME')

        bias_init = tf.constant(0.0, shape=[output_channel])
        bias = tf.get_variable('bias', initializer = bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)

    return dconv_with_bias


def up_layer_3d(parent, shape, output_channel, input_channel, upscale_factor, name):

    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], shape[3], output_channel]
        filter_shape = [kernel_size, kernel_size, kernel_size, output_channel, input_channel]
        weights = _get_bilinear_filter_3d(filter_shape, upscale_factor)
        deconv = tf.nn.conv3d_transpose(parent, weights, output_shape,
                                        strides = strides, padding='SAME')

        bias_init = tf.constant(0.0, shape=[output_channel])
        bias = tf.get_variable('bias', initializer = bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)

    return dconv_with_bias




def unbalanced_layer(parent, num_classes, name):
    '''
    
    '''
    with tf.variable_scope(name):
        W_init = np.ones((num_classes, ), dtype = 'float32')
        b_init = np.zeros((num_classes, ), dtype = 'float32')

        W = tf.get_variable(name = 'weights', initializer = tf.constant_initializer(value = W_init, dtype = tf.float32), shape = W_init.shape)
        b = tf.get_variable(name = 'bias', initializer = tf.constant_initializer(value = b_init, dtype = tf.float32), shape = b_init.shape)

        unbalanced_logits = parent * W + b
        softmax = tf.nn.softmax(unbalanced_logits)
    return softmax


def conv_layer_3d(parent, kernel_name, bias_name, name):
    with tf.variable_scope(name) as scope:
        kernel_weights = _get_kernel(kernel_name)
        if kernel_name == 'conv1_1_W':
            kernel_weights = np.concatenate((kernel_weights, kernel_weights[:, :, 2, :].reshape(3, 3, 1, 64)), axis = 2)
        kernel_weights = np.stack((kernel_weights, kernel_weights, kernel_weights))
        init = tf.constant_initializer(value = kernel_weights, dtype = tf.float32)
        kernel = tf.get_variable(name = 'weights', initializer = init, shape = kernel_weights.shape)
        conv = tf.nn.conv3d(parent, kernel, strides = [1, 1, 1, 1, 1], padding = 'SAME')

        bias = _get_bias(bias_name)
        init = tf.constant_initializer(value=bias, dtype=tf.float32)
        biases = tf.get_variable(name="biases", initializer=init, shape=bias.shape)

        conv_with_bias = tf.nn.bias_add(conv, biases)
        conv_with_relu = tf.nn.relu(conv_with_bias, name=scope.name)
    return conv_with_relu


def conv_layer(parent, kernel_name, bias_name, name):
    """
    This simple utility function create a convolution layer
    and applied relu activation.

    :param parent:
    :param kernel_name: Kernel weight tensor
    :param bias: Bias tensor
    :param name: Name of this layer
    :return: Convolution layer created according to the given parameters.
    """
    with tf.variable_scope(name) as scope:
        kernel_weights = _get_kernel(kernel_name)
        if kernel_name == 'conv1_1_W':
            kernel_weights = np.concatenate((kernel_weights, kernel_weights[:, :, 2, :].reshape(3, 3, 1, 64)), axis = 2)
        init = tf.constant_initializer(value=kernel_weights, dtype=tf.float32)
        kernel = tf.get_variable(name="weights", initializer=init, shape=kernel_weights.shape)
        conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')

        bias = _get_bias(bias_name)
        init = tf.constant_initializer(value=bias, dtype=tf.float32)
        biases = tf.get_variable(name="biases", initializer=init, shape=bias.shape)

        conv_with_bias = tf.nn.bias_add(conv, biases)
        conv_with_relu = tf.nn.relu(conv_with_bias, name=scope.name)
    return conv_with_relu


def max_pool_layer_3d(parent, kernel, stride, name, padding = 'SAME'):
    max_pool_3d = tf.nn.max_pool3d(parent, ksize = kernel, strides = stride, padding = padding , name = name)
    return max_pool_3d


def max_pool_layer(parent, kernel, stride, name, padding='SAME'):
    max_pool = tf.nn.max_pool(parent, ksize=kernel, strides=stride, padding=padding, name=name)
    return max_pool


def fully_collected_layer_3d(parent, name, dropout, num_classes = 5):
    with tf.variable_scope(name) as scope:
        if name == 'fc_1':
            w = vgg_weights['fc6_W']
            w_elements = 1
            for i in w.shape:
                w_elements *= i 
            w = w.reshape(w_elements)[:3 * 4 * 4 * 512 * 4096].reshape([3, 4, 4, 512, 4096])
            init = tf.constant_initializer(value = w, dtype = tf.float32)
            kernel = tf.get_variable(name = 'weights', initializer = init, shape = [3, 4, 4, 512, 4096])
            conv = tf.nn.conv3d(parent, kernel, [1, 1, 1, 1, 1], padding='SAME')
            bias = _get_bias('fc6_b')
            output = tf.nn.bias_add(conv, bias)
            output = tf.nn.relu(output, name=scope.name)
            return tf.nn.dropout(output, dropout)

        if name == 'fc_2':
            kernel = _reshape_fc_weights('fc7_W', [1, 1, 1, 4096, 4096])
            conv = tf.nn.conv3d(parent, kernel, [1, 1, 1, 1, 1], padding='SAME')
            bias = _get_bias('fc7_b')
            output = tf.nn.bias_add(conv, bias)
            output = tf.nn.relu(output, name=scope.name)
            return tf.nn.dropout(output, dropout)

        if name == 'fc_3':
            initial = tf.truncated_normal([1, 1, 1, 4096, num_classes], stddev=0.0001)
            kernel = tf.get_variable('kernel', initializer=initial)
            conv = tf.nn.conv3d(parent, kernel, [1, 1, 1, 1, 1], padding='SAME')
            initial = tf.constant(0.0, shape=[num_classes])
            bias = tf.get_variable('bias', initializer=initial)
            return tf.nn.bias_add(conv, bias)

        raise RuntimeError('{} is not supported as a fully connected name'.format(name))


def fully_collected_layer(parent, name, dropout, num_classes = 5):
    with tf.variable_scope(name) as scope:
        if name == 'fc_1':
            kernel = _reshape_fc_weights('fc6_W', [7, 7, 512, 4096])
            conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
            bias = _get_bias('fc6_b')
            output = tf.nn.bias_add(conv, bias)
            output = tf.nn.relu(output, name=scope.name)
            return tf.nn.dropout(output, dropout)

        if name == 'fc_2':
            kernel = _reshape_fc_weights('fc7_W', [1, 1, 4096, 4096])
            conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
            bias = _get_bias('fc7_b')
            output = tf.nn.bias_add(conv, bias)
            output = tf.nn.relu(output, name=scope.name)
            return tf.nn.dropout(output, dropout)

        if name == 'fc_3':
            initial = tf.truncated_normal([1, 1, 4096, num_classes], stddev=0.0001)
            kernel = tf.get_variable('kernel', initializer=initial)
            conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')
            initial = tf.constant(0.0, shape=[num_classes])
            bias = tf.get_variable('bias', initializer=initial)
            return tf.nn.bias_add(conv, bias)

        raise RuntimeError('{} is not supported as a fully connected name'.format(name))


def skip_layer_connection_3d(parent, name, num_input_layers, num_classes=5, stddev=0.0005):
    with tf.variable_scope(name) as scope:
        initial = tf.truncated_normal([1, 1, 1, num_input_layers, num_classes], stddev=stddev)
        kernel = tf.get_variable('kernel', initializer=initial)
        conv = tf.nn.conv3d(parent, kernel, [1, 1, 1, 1, 1], padding='SAME')

        bias_init = tf.constant(0.0, shape=[num_classes])
        bias = tf.get_variable('bias', initializer=bias_init)
        skip_layer = tf.nn.bias_add(conv, bias)

        return skip_layer


def skip_layer_connection(parent, name, num_input_layers, num_classes=5, stddev=0.0005):
    with tf.variable_scope(name) as scope:
        initial = tf.truncated_normal([1, 1, num_input_layers, num_classes], stddev=stddev)
        kernel = tf.get_variable('kernel', initializer=initial)
        conv = tf.nn.conv2d(parent, kernel, [1, 1, 1, 1], padding='SAME')

        bias_init = tf.constant(0.0, shape=[num_classes])
        bias = tf.get_variable('bias', initializer=bias_init)
        skip_layer = tf.nn.bias_add(conv, bias)

        return skip_layer


def upsample_layer_3d(bottom, shape, n_channels, name, upscale_factor, num_classes = 5):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], shape[3], num_classes]
        filter_shape = [kernel_size, kernel_size, kernel_size, n_channels, n_channels]
        weights = _get_bilinear_filter_3d(filter_shape, upscale_factor)
        deconv = tf.nn.conv3d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        bias_init = tf.constant(0.0, shape = [num_classes])
        bias = tf.get_variable('bias', initializer=bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)

    return dconv_with_bias

def upsample_layer(bottom, shape, n_channels, name, upscale_factor, num_classes = 5):
    kernel_size = 2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        output_shape = [shape[0], shape[1], shape[2], num_classes]
        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        weights = _get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        bias_init = tf.constant(0.0, shape=[num_classes])
        bias = tf.get_variable('bias', initializer=bias_init)
        dconv_with_bias = tf.nn.bias_add(deconv, bias)

    return dconv_with_bias

def _get_kernel(kernel_name):
    kernel = vgg_weights[kernel_name]
    return kernel


def _reshape_fc_weights(name, new_shape):
    w = vgg_weights[name]
    w = w.reshape(new_shape)
    init = tf.constant_initializer(value=w,
                                   dtype=tf.float32)
    var = tf.get_variable(name="weights", initializer=init, shape=new_shape)
    return var


def _get_bias(name):
    bias_weights = vgg_weights[name]
    return bias_weights


def _get_bilinear_filter_3d(filter_shape, upscale_factor):
    kernel_size = filter_shape[1]
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1], filter_shape[2]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            for z in range(filter_shape[2]):
                value = (1 - abs((x - centre_location) / upscale_factor)) * (
                    1 - abs((y - centre_location) / upscale_factor)) * (
                    1 - abs((z - centre_location) / upscale_factor))
                bilinear[x, y, z] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[3]):
        weights[:, :, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights


def _get_bilinear_filter(filter_shape, upscale_factor):
    kernel_size = filter_shape[1]
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
                1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        for j in range(filter_shape[3]):
            weights[:, :, i, j] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)

    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights

