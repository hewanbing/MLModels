from absl import flags
import tensorflow as tf
assert tf.__version__.startswith('2')
import DataSet

FLAGS = flags.FLAGS

def define_flags():
  flags.DEFINE_integer('epochs', 2, '')
  flags.DEFINE_boolean('enable_function', False, '')
  flags.DEFINE_integer('batch_size', 40, '')
  flags.DEFINE_integer('shuffle_size', 200, '')
  flags.DEFINE_string('setup_mode', 'from_depth', '')
  flags.DEFINE_integer('growth_rate', 12, '')
  flags.DEFINE_integer('output_classes', 2, '')
  flags.DEFINE_integer('depth_of_model', 4+12*3, '')
  flags.DEFINE_integer('num_of_blocks', 3, '')
  flags.DEFINE_integer('num_layers_in_each_block', 4,'')
  flags.DEFINE_string('data_format', 'channels_last', '')
  flags.DEFINE_boolean('bottleneck', True,'')
  flags.DEFINE_float('compression', 0.5,'')
  flags.DEFINE_float('weight_decay', 1e-4,'')
  flags.DEFINE_float('dropout_rate', 0., '')
  flags.DEFINE_boolean('pool_initial', False,'')
  flags.DEFINE_boolean('include_top', True, '')
  flags.DEFINE_string('train_mode', 'keras_fit','')

def makeDataset():

  train=DataSet.dataSet('./data', 32, 32, 32, 3).makeDataSet('train')
  valid=DataSet.dataSet('./data', 32, 32, 32, 3).makeDataSet('validation')
  evalu=DataSet.dataSet('./data', 32, 32, 32, 3).makeDataSet('eval')

  return train, valid, evalu
  
def flags_dict():
  kwargs = {
    'epochs': FLAGS.epochs,
    'enable_function': FLAGS.enable_function,
    'batch_size': FLAGS.batch_size,
    'shuffle_size': FLAGS.shuffle_size,
    'setup_mode': FLAGS.setup_mode,
    'growth_rate': FLAGS.growth_rate,
    'output_classes': FLAGS.output_classes,
    'depth_of_model': FLAGS.depth_of_model,
    'num_of_blocks': FLAGS.num_of_blocks,
    'num_layers_in_each_block': FLAGS.num_layers_in_each_block,
    'data_format': FLAGS.data_format,
    'bottleneck': FLAGS.bottleneck,
    'compression': FLAGS.compression,
    'weight_decay': FLAGS.weight_decay,
    'dropout_rate': FLAGS.dropout_rate,
    'pool_initial': FLAGS.pool_initial,
    'include_top': FLAGS.include_top,
    'train_mode': FLAGS.train_mode
  }
  return kwargs
