from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
import ResNet2D
import numpy as np

def main():
  NDim1=28
  NDim2=28
  Channel=1
  Classes=10
  model=ResNet2D.resnet50(Classes)
  if Classes > 1:
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
  else:
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])


  #check trained weights
  if os.path.isfile('trainingCheckpoint/cp.ckpt.index'):
    model.load_weights('trainingCheckpoint/cp.ckpt')
    
  model.build(input_shape=(None, NDim1, NDim2, Channel))
  print("Number of variables in the model :", len(model.variables))
  model.summary()

  #data
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # Preprocess the data (these are Numpy arrays)
  x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
  x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255
  
  y_train = y_train.astype('float32')
  y_test = y_test.astype('float32')
  
  # Reserve 10,000 samples for validation
  x_val = x_train[-10000:]
  y_val = y_train[-10000:]
  x_train = x_train[:-10000]
  y_train = y_train[:-10000]
    
  #evaluate
  scores = model.evaluate(x_test, y_test, batch_size=128)
  print("Final test loss and accuracy :", scores)

  predictions = model.predict(x_test[:100])
  idx=0
  for prediction in predictions:
    classes=np.argmax(prediction)
    difference= classes - y_test[idx]
    print('Prediction: ',classes,' Tag: ',y_test[idx],' Difference: ', difference)
    idx=idx+1
  
if __name__ == '__main__':
  main()
