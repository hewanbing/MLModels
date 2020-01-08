import argparse
from ROOT import TFile, gSystem,TChain
from ROOT import gRandom,TCanvas,TH3D,TH2D,TH1D,gDirectory
import tensorflow as tf
import numpy as np

def histogram3D2Matrix(histogram3D,dimensionZ,dimensionY,dimensionX):
  matrix=np.zeros((dimensionZ,dimensionY,dimensionX))

  for z in range(dimensionZ):
    for y in range(dimensionY):
      for x in range(dimensionX):
        matrix[z][y][x]=histogram3D.GetBinContent(x+1,y+1,z+1)

  return matrix

def convert2tfrecords(args):

  nChannel=3
  h1 = TH3D('h1','h1',\
            args.nBinX,args.rangeXL,args.rangeXH,\
            args.nBinY,args.rangeYL,args.rangeYH,\
            args.nBinZ,args.rangeZL,args.rangeZH)
  h2 = TH3D('h2','h2',\
            args.nBinX,args.rangeXL,args.rangeXH,\
            args.nBinY,args.rangeYL,args.rangeYH,\
            args.nBinZ,args.rangeZL,args.rangeZH)
  h3 = TH3D('h3','h3',\
            args.nBinX,args.rangeXL,args.rangeXH,\
            args.nBinY,args.rangeYL,args.rangeYH,\
            args.nBinZ,args.rangeZL,args.rangeZH)
  chainCME0=TChain('AMPT')
  chainCME0.Add(args.input0)
  chainCME1=TChain('AMPT')
  chainCME1.Add(args.input1)
  totalNumber0=chainCME0.GetEntries()
  totalNumber1=chainCME1.GetEntries()
  
  with tf.io.TFRecordWriter(args.output) as record_writer:

    writeOutNumber=0
    for (event0,event1) in zip(chainCME0,chainCME1):
      for i in range(event0.mult):
        if(event0.ID[i]==211):
          h1.Fill(event0.Px[i],event0.Py[i],event0.Pz[i])
          continue
        if(event0.ID[i]==111):
          h2.Fill(event0.Px[i],event0.Py[i],event0.Pz[i])
          continue
        if(event0.ID[i]==-211):
          h3.Fill(event0.Px[i],event0.Py[i],event0.Pz[i])
          continue

      channel1=histogram3D2Matrix(h1,args.nBinZ,args.nBinY,args.nBinX)
      channel2=histogram3D2Matrix(h2,args.nBinZ,args.nBinY,args.nBinX)
      channel3=histogram3D2Matrix(h3,args.nBinZ,args.nBinY,args.nBinX)
      
      channel1=channel1[:,:,:,np.newaxis]
      channel2=channel2[:,:,:,np.newaxis]
      channel3=channel3[:,:,:,np.newaxis]
      event=np.hstack((channel1,channel2,channel3))
      event=event.reshape(args.nBinX*args.nBinY*args.nBinZ*nChannel)
      example=tf.train.Example(features=tf.train.Features(
        feature={
          'image': tf.train.Feature(float_list=tf.train.FloatList(value=event)),
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
          'nBinX': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinX])),
          'nBinY': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinY])),
          'nBinZ': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinZ])),
          'nChannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nChannel])),
        }))
      record_writer.write(example.SerializeToString())
      h1.Reset()
      h2.Reset()
      h3.Reset()

      for i in range(event1.mult):
        if(event1.ID[i]==211):
          h1.Fill(event1.Px[i],event1.Py[i],event1.Pz[i])
          continue
        if(event1.ID[i]==111):
          h2.Fill(event1.Px[i],event1.Py[i],event1.Pz[i])
          continue
        if(event1.ID[i]==-211):
          h3.Fill(event1.Px[i],event1.Py[i],event1.Pz[i])
          continue
        
      channel1=histogram3D2Matrix(h1,args.nBinZ,args.nBinY,args.nBinX)
      channel2=histogram3D2Matrix(h2,args.nBinZ,args.nBinY,args.nBinX)
      channel3=histogram3D2Matrix(h3,args.nBinZ,args.nBinY,args.nBinX)
      
      channel1=channel1[:,:,:,np.newaxis]
      channel2=channel2[:,:,:,np.newaxis]
      channel3=channel3[:,:,:,np.newaxis]
      event=np.hstack((channel1,channel2,channel3))
      event=event.reshape(args.nBinX*args.nBinY*args.nBinZ*nChannel)
      example=tf.train.Example(features=tf.train.Features(
        feature={
          'image': tf.train.Feature(float_list=tf.train.FloatList(value=event)),
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
          'nBinX': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinX])),
          'nBinY': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinY])),
          'nBinZ': tf.train.Feature(int64_list=tf.train.Int64List(value=[args.nBinZ])),
          'nChannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nChannel])),
        }))
      record_writer.write(example.SerializeToString())
      h1.Reset()
      h2.Reset()
      h3.Reset()
      writeOutNumber+=1
      print('{}\r'.format('   Tag0:%d, Tag1:%d, %d writen into %s' %(totalNumber0,totalNumber1,writeOutNumber,args.output)),end="",flush=True)
      
def main(args):
  print('Creating TFRecord files...')
  convert2tfrecords(args)
  print('')
  print('Done!')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input0',
      type=str,
      default='afterART_0.root')
  parser.add_argument(
      '--input1',
      type=str,
      default='afterART_1.root')
  parser.add_argument(
      '--output',
      type=str,
      default='train.tfrecords')
  parser.add_argument(
      '--nBinX',
      type=int,
      default=32)
  parser.add_argument(
      '--nBinY',
      type=int,
      default=32)
  parser.add_argument(
      '--nBinZ',
      type=int,
      default=32)
  parser.add_argument(
      '--rangeXL',
      type=float,
      default=-3)
  parser.add_argument(
      '--rangeXH',
      type=float,
      default=3)
  parser.add_argument(
      '--rangeYL',
      type=float,
      default=-3)
  parser.add_argument(
      '--rangeYH',
      type=float,
      default=3)
  parser.add_argument(
      '--rangeZL',
      type=float,
      default=-20)
  parser.add_argument(
      '--rangeZH',
      type=float,
      default=20)
  
  args = parser.parse_args()
  main(args)
