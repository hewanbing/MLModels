from absl import app
import tensorflow as tf
import os
import DenseNet3D
import utils
assert tf.__version__.startswith('2')

class Train(object):
  def __init__(self, epochs, enable_function, model, output_classes):
    self.epochs = epochs
    self.enable_function = enable_function
    self.autotune = tf.data.experimental.AUTOTUNE
    
    self.optimizer = tf.keras.optimizers.Adam(0.001)

    self.classes=output_classes

    if output_classes > 1:
      self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      self.acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    else:
      self.loss = tf.keras.losses.MeanSquaredError()
      self.acc_metric = tf.keras.metrics.MeanSquaredError()

    self.callbacks=[tf.keras.callbacks.TensorBoard(log_dir='log',
                                                   histogram_freq=1,
                                                   write_graph=True,
                                                   write_images=True),
                    tf.keras.callbacks.ModelCheckpoint(filepath='trainingCheckpoint/cp.ckpt',
                                                       save_weights_only=True,
                                                       verbose=1),
                    tf.keras.callbacks.LearningRateScheduler(self.decay)
    ]
    
    self.model = model

  def decay(self, epoch):
    if epoch < 150:
      return 0.1
    if epoch >= 150 and epoch < 225:
      return 0.01
    if epoch >= 225:
      return 0.001

  def keras_fit(self, train_dataset, test_dataset):
    self.model.compile(optimizer=self.optimizer,
                       loss=self.loss,
                       metrics=[self.acc_metric])
      
    if os.path.isfile('trainingCheckpoint/cp.ckpt.index'):
        self.model.load_weights('trainingCheckpoint/cp.ckpt')
        
    #self.model.build(input_shape=(None, 32, 32, 32, 3))
    #print("Number of variables in the model :", len(self.model.variables))
    #self.model.summary()

    history = self.model.fit(train_dataset,
                             epochs=self.epochs,
                             validation_data=test_dataset,
                             verbose=1,
                             callbacks=self.callbacks)
    return (history.history),

  def train_step(self, image, label):
    with tf.GradientTape() as tape:
      predictions = self.model(image, training=True)
      loss = self.loss(label, predictions)
      loss += sum(self.model.losses)
      
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))

    self.acc_metric(label, predictions)

  def test_step(self, image, label):
    predictions = self.model(image, training=False)
    loss = self.loss(label, predictions)

    self.acc_metric(label, predictions)

  def custom_fit(self, train_dataset, test_dataset):
    if os.path.isfile('trainingCheckpoint/cp.ckpt.index'):
        self.model.load_weights('trainingCheckpoint/cp.ckpt')
        
    self.model.build(input_shape=(None, 32, 32, 32, 3))
    print("Number of variables in the model :", len(self.model.variables))
    self.model.summary()
    
    if self.enable_function:
      self.train_step = tf.function(self.train_step)
      self.test_step = tf.function(self.test_step)

    for epoch in range(self.epochs):
      self.optimizer.learning_rate = self.decay(epoch)

      for image, label in train_dataset:
        self.train_step(image, label)

      for test_image, test_label in test_dataset:
        self.test_step(test_image, test_label)

      template = ('Epoch: {}, Train Accuracy: {}')

      print(
          template.format(epoch, self.acc_metric.result()))

      if epoch != self.epochs - 1:
        self.acc_metric.reset_states()

    return (self.acc_metric.result().numpy())

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
  train_obj = Train(epochs, enable_function, model, output_classes)
  train_dataset, test_dataset, eval_dataset = utils.makeDataset()

  print('Training...')
  if train_mode == 'custom_fit':
    return train_obj.custom_fit(train_dataset.shuffle(shuffle_size).batch(batch_size),
                                 test_dataset.batch(batch_size))
  elif train_mode == 'keras_fit':
    return train_obj.keras_fit(train_dataset.shuffle(shuffle_size).batch(batch_size),
                               test_dataset.batch(batch_size))

if __name__ == '__main__':
  utils.define_flags()
  app.run(run_main)
