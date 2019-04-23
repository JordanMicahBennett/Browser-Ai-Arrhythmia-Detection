import numpy as np
import keras
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPool2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

from keras.callbacks import ModelCheckpoint
filepath = input("Enter the filename you want your model to be saved as: ")

train_path = input("Enter the directory of the training images: ")
valid_path = input("Enter the directory of the validation images: ")

checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

batch_size = 32

IMAGE_SIZE = [128, 128]





from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = IMAGE_SIZE + [3],kernel_initializer='glorot_uniform'))


model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Flatten())

model.add(Dense(2048))


model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

gen = ImageDataGenerator()

test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)

train_gen = gen.flow_from_directory(train_path, target_size=IMAGE_SIZE)

train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
callbacks_list = [checkpoint]

r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=50,
  steps_per_epoch=356702//batch_size,
  validation_steps=39634//batch_size,callbacks=callbacks_list
)