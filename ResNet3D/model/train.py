from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import keras
import ResNet3D
import DataSet 
import numpy as np

def main():
  NDim1=32
  NDim2=32
  NDim3=32
  Channel=3
  Classes=2
  model=ResNet3D.resnet50(Classes)
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
    
  model.build(input_shape=(None, NDim1, NDim2, NDim3, Channel))
  print("Number of variables in the model :", len(model.variables))
  model.summary()

  #data
  trainData=DataSet.dataSet('./data', NDim1, NDim2, NDim3, Channel).makeDataSet('train')
  validData=DataSet.dataSet('./data', NDim1, NDim2, NDim3, Channel).makeDataSet('validation')
  evalData=DataSet.dataSet('./data', NDim1, NDim2, NDim3, Channel).makeDataSet('eval')

  preScores = model.evaluate(evalData.batch(10), verbose=1)
  print("Pre train loss and accuracy :", preScores)
  
  #train
  history=model.fit(trainData.shuffle(200).batch(10), epochs=1,
                    validation_data=validData.batch(10), verbose=1,
                    callbacks=[keras.callbacks.TensorBoard(log_dir='log',
                                                           histogram_freq=1,
                                                           write_graph=True,
                                                           write_images=True),
                               keras.callbacks.ModelCheckpoint(filepath='trainingCheckpoint/cp.ckpt',
                                                               save_weights_only=True,
                                                               verbose=1)
                    ])
  print(history.history)
  #evaluate
  scores = model.evaluate(evalData.batch(10), verbose=1)
  print("Final test loss and accuracy :", scores)
  #save
  model.save('./save/All')
  model.save_weights('./save/Weights')
  
if __name__ == '__main__':
  main()
