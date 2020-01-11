from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf # TF2
assert tf.__version__.startswith('2')

l2 = tf.keras.regularizers.l2

def calc_from_depth(depth, num_blocks, bottleneck):
  if depth is None or num_blocks is None:
    raise ValueError("For 'from_depth' setup_mode, you need to specify the depth "
                     "and number of blocks.")

  if num_blocks != 3:
    raise ValueError(
        "Number of blocks must be 3 if setup_mode is 'from_depth'.")

  if (depth - 4) % 3 == 0:
    num_layers = (depth - 4) / 3
    if bottleneck:
      num_layers //= 2
    return [num_layers] * num_blocks
  else:
    raise ValueError("Depth must be 3N+4 if setup_mode is 'from_depth'.")

def calc_from_list(depth, num_blocks, layers_per_block):
  if depth is not None or num_blocks is not None:
    raise ValueError("You don't have to specify the depth and number of "
                     "blocks when setup_mode is 'from_list'")

  if layers_per_block is None or not isinstance(
      layers_per_block, list) or not isinstance(layers_per_block, tuple):
    raise ValueError("You must pass list or tuple when using 'from_list' setup_mode.")

  if isinstance(layers_per_block, list) or isinstance(layers_per_block, tuple):
    return list(layers_per_block)

def calc_from_integer(depth, num_blocks, layers_per_block):
  if depth is not None:
    raise ValueError("You don't have to specify the depth "
                     "when setup_mode is 'from_integer'")

  if num_blocks is None or not isinstance(layers_per_block, int):
    raise ValueError("You must pass number of blocks or an integer to "
                     "layers in each block")

  return [layers_per_block] * num_blocks

class ConvBlock(tf.keras.Model):
  def __init__(self, num_filters, data_format, bottleneck, weight_decay=1e-4,
               dropout_rate=0):
    super(ConvBlock, self).__init__()
    self.bottleneck = bottleneck

    axis = -1 if data_format == "channels_last" else 1
    inter_filter = num_filters * 4

    self.conv2 = tf.keras.layers.Conv3D(num_filters,
                                        (3, 3, 3),
                                        padding="same",
                                        use_bias=False,
                                        data_format=data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(weight_decay))
    self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

    if self.bottleneck:
      self.conv1 = tf.keras.layers.Conv3D(inter_filter,
                                          (1, 1, 1),
                                          padding="same",
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=l2(weight_decay))
      self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)

  def call(self, x, training=True):
    x = self.batchnorm1(x, training=training)

    if self.bottleneck:
      x = self.conv1(tf.nn.relu(x))
      x = self.batchnorm2(x, training=training)

    x = self.conv2(tf.nn.relu(x))
    x = self.dropout(x, training=training)

    return x

class TransitionBlock(tf.keras.Model):
  def __init__(self, num_filters, data_format,
               weight_decay=1e-4, dropout_rate=0):
    super(TransitionBlock, self).__init__()
    axis = -1 if data_format == "channels_last" else 1

    self.batchnorm = tf.keras.layers.BatchNormalization(axis=axis)
    self.conv = tf.keras.layers.Conv3D(num_filters,
                                       (1, 1, 1),
                                       padding="same",
                                       use_bias=False,
                                       data_format=data_format,
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=l2(weight_decay))
    self.avg_pool = tf.keras.layers.AveragePooling3D(data_format=data_format)

  def call(self, x, training=True):
    x = self.batchnorm(x, training=training)
    x = self.conv(tf.nn.relu(x))
    x = self.avg_pool(x)
    return x

class DenseBlock(tf.keras.Model):
  def __init__(self, num_layers, growth_rate, data_format, bottleneck,
               weight_decay=1e-4, dropout_rate=0):
    super(DenseBlock, self).__init__()
    self.num_layers = num_layers
    self.axis = -1 if data_format == "channels_last" else 1

    self.blocks = []
    for _ in range(int(self.num_layers)):
      self.blocks.append(ConvBlock(growth_rate,
                                   data_format,
                                   bottleneck,
                                   weight_decay,
                                   dropout_rate))

  def call(self, x, training=True):
    for i in range(int(self.num_layers)):
      output = self.blocks[i](x, training=training)
      x = tf.concat([x, output], axis=self.axis)

    return x

class DenseNet(tf.keras.Model):
  def __init__(self, setup_mode, growth_rate, output_classes, depth_of_model=None,
               num_of_blocks=None, num_layers_in_each_block=None,
               data_format="channels_last", bottleneck=True, compression=0.5,
               weight_decay=1e-4, dropout_rate=0., pool_initial=False,
               include_top=True):
    super(DenseNet, self).__init__()
    self.setup_mode = setup_mode
    self.depth_of_model = depth_of_model
    self.growth_rate = growth_rate
    self.num_of_blocks = num_of_blocks
    self.output_classes = output_classes
    self.num_layers_in_each_block = num_layers_in_each_block
    self.data_format = data_format
    self.bottleneck = bottleneck
    self.compression = compression
    self.weight_decay = weight_decay
    self.dropout_rate = dropout_rate
    self.pool_initial = pool_initial
    self.include_top = include_top

    self.padding = tf.keras.layers.ZeroPadding3D((1, 1, 1))

    if setup_mode == "from_depth":
      self.num_layers_in_each_block = calc_from_depth(
          self.depth_of_model, self.num_of_blocks, self.bottleneck)
    elif setup_mode == "from_list":
      self.num_layers_in_each_block = calc_from_list(
          self.depth_of_model, self.num_of_blocks,
          self.num_layers_in_each_block)
    elif setup_mode == "from_integer":
      self.num_layers_in_each_block = calc_from_integer(
          self.depth_of_model, self.num_of_blocks,
          self.num_layers_in_each_block)

    axis = -1 if self.data_format == "channels_last" else 1

    if self.pool_initial:
      init_filters = (7, 7, 7)
      stride = (2, 2, 2)
    else:
      init_filters = (3, 3, 3)
      stride = (1, 1, 1)

    self.num_filters = 2 * self.growth_rate

    self.conv1 = tf.keras.layers.Conv3D(self.num_filters,
                                        init_filters,
                                        strides=stride,
                                        padding="same",
                                        use_bias=False,
                                        data_format=self.data_format,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=l2(
                                            self.weight_decay))
    if self.pool_initial:
      self.pool1 = tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3),
                                                strides=(2, 2, 2),
                                                padding="same",
                                                data_format=self.data_format)
      self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=axis)
    
    self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=axis)
    
    num_filters_after_each_block = [self.num_filters]
    for i in range(1, self.num_of_blocks):
      temp_num_filters = num_filters_after_each_block[i-1] + (
          self.growth_rate * self.num_layers_in_each_block[i-1])
      temp_num_filters = int(temp_num_filters * compression)
      num_filters_after_each_block.append(temp_num_filters)
    
    self.dense_blocks = []
    self.transition_blocks = []
    for i in range(self.num_of_blocks):
      self.dense_blocks.append(DenseBlock(self.num_layers_in_each_block[i],
                                          self.growth_rate,
                                          self.data_format,
                                          self.bottleneck,
                                          self.weight_decay,
                                          self.dropout_rate))
      if i+1 < self.num_of_blocks:
        self.transition_blocks.append(
            TransitionBlock(num_filters_after_each_block[i+1],
                            self.data_format,
                            self.weight_decay,
                            self.dropout_rate))
    
    if self.include_top:
      self.last_pool = tf.keras.layers.GlobalAveragePooling3D(
          data_format=self.data_format)
      if self.output_classes>1:
        self.fc = tf.keras.layers.Dense(self.output_classes,activation='softmax')
      else:
        self.fc = tf.keras.layers.Dense(1)
    
  def call(self, x, training=True):
    x = self.padding(x)
    x = self.conv1(x)
    
    if self.pool_initial:
      x = self.batchnorm1(x, training=training)
      x = tf.nn.relu(x)
      x = self.pool1(x)
    
    for i in range(self.num_of_blocks - 1):
      x = self.dense_blocks[i](x, training=training)
      x = self.transition_blocks[i](x, training=training)
    
    x = self.dense_blocks[
        self.num_of_blocks -1](x, training=training)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.relu(x)
    
    if self.include_top:
      x = self.last_pool(x)
      x = self.fc(x)

    return x
