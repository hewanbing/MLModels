import os
import tensorflow as tf

class dataSet(object):

  def __init__(self, data_dir,dimension1,dimension2,dimension3,channel):
    self.data_dir = data_dir
    self.dim1=dimension1
    self.dim2=dimension2
    self.dim3=dimension3
    self.channel=channel
    
  def get_filenames(self):
    if self.subset in ['train', 'validation', 'eval']:
      fileNames=[]
      for root, dirs, files in os.walk(self.data_dir):
        for file in files:
          if file.find(self.subset) >= 0:
            if os.path.splitext(file)[1] == '.tfrecords':
              fileNames.append(os.path.join(root, file))

      return fileNames
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def makeDataSet(self,subset):
    self.subset = subset
    filenames = self.get_filenames()
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(self.parser)
    return dataset

  def parser(self, serialized_example):
    features = tf.io.parse_single_example(
      serialized_example,
      features={
        'image':tf.io.FixedLenFeature([self.dim1,self.dim2,self.dim3,self.channel],tf.float32),
        'label':tf.io.FixedLenFeature([],tf.int64),
        'nBinX':tf.io.FixedLenFeature([],tf.int64),
        'nBinY':tf.io.FixedLenFeature([],tf.int64),
        'nBinZ':tf.io.FixedLenFeature([],tf.int64),
        'nChannel':tf.io.FixedLenFeature([],tf.int64),
      })
    image = features['image']
    label = tf.cast(features['label'], tf.int32)

    return image, label
