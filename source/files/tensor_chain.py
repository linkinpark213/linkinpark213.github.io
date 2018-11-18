import tensorflow as tf


class TensorChain:
    def __init__(self, input: tf.Tensor):
        self.input_tensor = input
        self.output_tensor = input
        shape = input.shape.as_list()
        self.num_channels = shape[shape.__len__() - 1]
        self.layers_record = ['Input']
        self.identity_mapping_stack = []

    @staticmethod
    def _weights(shape: list, layer_name: str = None, sigma: float = 0.1, suffix: str = 'weights') -> tf.Variable:
        """
        Generate a tensorflow weights variable with the given shape and truncated normal distribution
        :param shape: The shape of the needed variable
        :param layer_name: The name of the layer utilizing this weight variable
        :param suffix: Suffix for the name of this weight variable
        :return: A tf.Variable object
        """
        if layer_name is None:
            return tf.Variable(tf.truncated_normal(shape, stddev=sigma), dtype=tf.float32)
        else:
            with tf.variable_scope(layer_name, default_name=''):
                return tf.Variable(tf.truncated_normal(shape, stddev=sigma), dtype=tf.float32, name=suffix)

    @staticmethod
    def _bias(shape: list, value: float = 0, layer_name: str = None, suffix: str = 'bias') -> tf.Variable:
        """
        Generate a tensorflow bias variable with the given shape, initialization is 0.1
        :param shape: The shape of the needed variable
        :param value: The initial value of the bias variable
        :param layer_name: The name of the layer utilizing this weight variable
        :param suffix: Suffix for the name of this weight variable
        :return: A tf.Variable object
        """
        if layer_name is None:
            return tf.Variable(tf.constant(value, shape=shape), dtype=tf.float32)
        else:
            with tf.variable_scope(layer_name, default_name=''):
                return tf.Variable(tf.constant(value, shape=shape), dtype=tf.float32, name=suffix)

    def _log_layer(self, log: str, disable: bool = False):
        """
        Add a layer record to the chain
        :param log: Log content
        :param disable: Set it True if you don't want it to be recorded
        """
        if disable == False:
            self.layers_record.append(log)

    def fully_connected_layer(self, output_dim: int, name: str = None, disable_log: bool = False):
        """
        Add a fully connected layer.
        :param output_dim: An integer indicating the dimension of the output
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        original_length = self.output_tensor.shape[1].value
        weights = self._weights([original_length, output_dim])
        bias = self._bias([output_dim])
        self.output_tensor = tf.matmul(self.output_tensor, weights) + bias
        self._log_layer('Fully connected layer, {} x {}'.format(original_length, output_dim), disable=disable_log)
        return self

    def max_pooling_layer_2d(self, kernel_size: int, stride=0, padding='SAME', name: str = None,
                             disable_log: bool = False):
        """
        Add a 2D max pooling layer
        :param kernel_size: Kernel size of the pooling layer
        :param stride: Stride of the pooling layer. If stride == 0, The stride would be same as kernel size
        :param padding: Padding pattern, 'SAME' by default, or 'VALID' alternatively
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        if stride == 0:
            self.output_tensor = tf.nn.max_pool(input, [1, kernel_size, kernel_size, 1],
                                                [1, kernel_size, kernel_size, 1], padding=padding,
                                                name=name)
        else:
            self.output_tensor = tf.nn.max_pool(input, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
                                                padding=padding,
                                                name=name)
        self._log_layer('2D Max pooling layer, kernel size = {}'.format(kernel_size, stride), disable=disable_log)
        return self

    def average_pooling_layer_2d(self, kernel_size: int, stride=0, padding='SAME', name: str = None,
                                 disable_log: bool = False):
        """
        Add an 2D average pooling layer
        :param kernel_size: Kernel size of the pooling layer
        :param stride: Stride of the pooling layer. If stride == 0, The stride would be same as kernel size
        :param padding: Padding pattern, 'SAME' by default, or 'VALID' alternatively
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        if stride == 0:
            self.output_tensor = tf.nn.avg_pool(input, [1, kernel_size, kernel_size, 1],
                                                [1, kernel_size, kernel_size, 1], padding=padding,
                                                name=name)
        else:
            self.output_tensor = tf.nn.avg_pool(input, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1],
                                                padding=padding,
                                                name=name)
        self._log_layer('2D Average pooling layer, kernel size = {}'.format(kernel_size, stride), disable=disable_log)
        return self

    def relu(self, name: str = None, disable_log: bool = False):
        """
        Add a Rectified Linear Unit
        :param name: name of this tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        self.output_tensor = tf.nn.relu(self.output_tensor, name=name)
        self._log_layer('Rectified Linear Unit', disable=disable_log)
        return self

    def convolution_layer_2d(self, filter_size: int, num_channels: int, stride: int = 1, name: str = None,
                             disable_log: bool = False):
        """
        Add a 2D convolution layer
        :param filter_size: Filter size(width and height) for this operation
        :param num_channels: Channel number of this filter
        :param stride: Stride for this convolution operation
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        filter = self._weights([filter_size, filter_size, self.num_channels, num_channels], layer_name=name,
                               suffix='filter')
        bias = self._bias([num_channels], layer_name=name)
        self.num_channels = num_channels
        self.output_tensor = tf.nn.conv2d(self.output_tensor, filter,
                                          [1, stride, stride, 1], 'SAME', name=name)
        self.output_tensor = tf.add(self.output_tensor, bias)
        self._log_layer(
            '2D Convolution layer, filter size = {}x{}, stride = {}, {} channels'.format(filter_size, filter_size,
                                                                                         stride,
                                                                                         num_channels),
            disable=disable_log)
        return self

    def batch_normalization(self, axis: int = -1, disable_log: bool = False):
        """
        Add a batch normalization operation
        :param axis: The axis to be normalized
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        self.output_tensor = tf.layers.batch_normalization(self.output_tensor, axis)
        self._log_layer('Batch Normalization on axis {}'.format(axis), disable=disable_log)
        return self

    def basic_block_2d(self, filter_size: int, num_channels: int, stride: int = 1, name: str = None,
                       disable_log: bool = False):
        """
        Add a 2D basic block, including a batch normalization operation, a ReLU operation and a convolution layer
        :param filter_size: Filter size(width, height and depth) of the convolution filter
        :param num_channels: Channel number of the convolution filter
        :param stride: Stride for the convolution operation
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        self.convolution_layer_2d(filter_size, num_channels, stride, name, disable_log=True)
        self.batch_normalization(3, disable_log=True)
        self.relu(disable_log=True)
        self.num_channels = num_channels
        self._log_layer('2D basic block, filter size = {}x{}, stride = {}, {} channels'.format(filter_size,
                                                                                               filter_size,
                                                                                               stride,
                                                                                               num_channels),
                        disable_log)
        return self

    def residual_block_2d(self, filter_size: int, num_channels: int, stride: int = 1, name: str = None,
                          pre_activation: bool = False,
                          disable_log: bool = False):
        """
        Add a 2D residual block, including a basic block, an identity mapping and the add operation
        :param filter_size: Filter size(width and height) of the convolution filter
        :param num_channels: Channel number of the convolution filter
        :param stride: Stride for the convolution operation
        :param name: The name of the tensor
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        self.branch_identity_mapping()
        if pre_activation:
            self.batch_normalization(disable_log=True)
            self.relu(disable_log=True)
            self.convolution_layer_2d(filter_size, num_channels, stride, name, disable_log=True)
            self.batch_normalization(disable_log=True)
            self.relu(disable_log=True)
            self.convolution_layer_2d(filter_size, num_channels, stride, name, disable_log=True)
            self.merge_identity_mapping_2d()
        else:
            self.convolution_layer_2d(filter_size, num_channels, stride, name, disable_log=True)
            self.batch_normalization(disable_log=True)
            self.relu(disable_log=True)
            self.convolution_layer_2d(filter_size, num_channels, stride, name, disable_log=True)
            self.batch_normalization(disable_log=True)
            self.merge_identity_mapping_2d()
            self.relu(disable_log=True)
        self._log_layer(
            '2D residual block, filter size = {}x{}, stride = {}, {} channels'.format(filter_size, filter_size, stride,
                                                                                      num_channels), disable_log)
        return self

    def reshape(self, new_shape: list, name: str = None, disable_log: bool = False):
        """
        Reshape the output tensor
        :param new_shape: The new shape of the tensor
        :param name: Name of the operation
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        self.output_tensor = tf.reshape(self.output_tensor, new_shape, name=name)
        self._log_layer('Reshape to {}'.format(new_shape), disable=disable_log)
        self.num_channels = new_shape[-1]
        return self

    def branch_identity_mapping(self):
        """
        Take an identity mapping at the current place
        :return: This object itself
        """
        self.identity_mapping_stack.append(self.output_tensor)
        return self

    def merge_identity_mapping_2d(self):
        """
        Add the identity mapping to the current output, 2D
        :return: This object itself
        """
        identity_mapping = self.identity_mapping_stack.pop()
        in_channel_num = identity_mapping.shape[3].value
        out_channel_num = self.num_channels
        if in_channel_num != out_channel_num:
            # Ignoring upsampling circumstances
            # If identity_mapping.shape[1].value > self.output_tensor.shape[1].value:
            stride = int(identity_mapping.shape[1].value / self.output_tensor.shape[1].value)
            # identity_filter = tf.constant(1, dtype=tf.float32, shape=[1, 1, in_channel_num, out_channel_num])
            identity_filter = self._weights([1, 1, in_channel_num, out_channel_num])
            identity_mapping = tf.nn.conv2d(identity_mapping, identity_filter, [1, stride, stride, 1], 'SAME')

        self.output_tensor = tf.add(self.output_tensor, identity_mapping)
        return self

    def flatten(self, name: str = None, disable_log: bool = False):
        """
        Flatten the output tensor
        :param name: Name of this operation
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return: This object itself
        """
        shape = self.output_tensor.shape
        prod = 1
        for i in range(1, shape.__len__()):
            prod *= shape[i].value
        self.reshape([-1, prod], name=name, disable_log=True)
        self._log_layer("Flatten to (batch_size, {})".format(prod), disable=disable_log)
        return self

    def dropout(self, keep_prob, name: str = None, disable_log: bool = False):
        """
        Add a dropout layer
        :param keep_prob: The probability that the network model don't drop each neuron
        :param name: Name of this operation
        :param disable_log: Set it True if you don't want this layer to be recorded
        """
        self.output_tensor = tf.nn.dropout(self.output_tensor, keep_prob=keep_prob, name=name)
        self._log_layer("Dropout layer, 'keep' probability = {})".format(keep_prob), disable=disable_log)
        return self

    def softmax(self, name: str = None, disable_log: bool = False):
        """
        Add a softmax operation
        :param name: Name of this operation
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return:
        """
        self.output_tensor = tf.nn.softmax(self.output_tensor, name=name)
        self._log_layer("Softmax", disable=disable_log)
        return self

    def sigmoid(self, name: str = None, disable_log: bool = False):
        """
        Add a sigmoid activation function
        :param name: Name of this operation
        :param disable_log: Set it True if you don't want this layer to be recorded
        :return:
        """
        self.output_tensor = tf.nn.sigmoid(self.output_tensor, name=name)
        self._log_layer("Sigmoid activation", disable=disable_log)
        return self
