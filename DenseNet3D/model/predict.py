from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import DenseNet3D
import DataSet
import numpy as np
import utils
from absl import app


def run_main(argv):
  del argv
  kwargs = utils.flags_dict()
  main(**kwargs)

def main(epochs,
         enable_function,
         batch_size,
         shuffle_size,
         setup_mode,
         growth_rate,
         output_classes,
         depth_of_model=None,
         num_of_blocks=None,
         num_layers_in_each_block=None,
         data_format='channels_last',
         bottleneck=True,
         compression=0.5,
         weight_decay=1e-4,
         dropout_rate=0.,
         pool_initial=False,
         include_top=True,
         train_mode='custom_fit'):

  model = DenseNet3D.DenseNet(setup_mode, growth_rate, output_classes, depth_of_model,
                            num_of_blocks, num_layers_in_each_block,
                            data_format, bottleneck, compression, weight_decay,
                            dropout_rate, pool_initial, include_top)
  train_dataset, test_dataset, eval_dataset = utils.makeDataset()

  if os.path.isfile('trainingCheckpoint/cp.ckpt.index'):
    model.load_weights('trainingCheckpoint/cp.ckpt')

  for value,tab in eval_dataset.take(10):
    prediction=model.predict(value[np.newaxis,:,:,:,:])
    print(prediction, tab.numpy())
  
if __name__ == '__main__':
  utils.define_flags()
  app.run(run_main)
