import os
import tensorflow as tf
import numpy as np
import cv2
import glob2 as glob
import random
import libs.preprocessing as pre

def blurring(img):
      """_summary_

      Args:
          img (MatLike): The image to transform.

      Returns:
          MatLike: The blurred image.
      """      
      blur = cv2.GaussianBlur(img.copy(),(5,5),1.)
      blur = cv2.GaussianBlur(blur,(7,7),(7.-1.)/4.)
      
      return blur

# Set data_dir to the directory of error-free images:  
data_dir = 'path_to_error_free_data'
checkpoint_filepath = './auto_encoder_{epoch:02d}-{loss:.4f}-{val_loss:.4f}.keras'

batch_size = 32
img_height = 256
img_width = 256
num_filter = 32
seed_value = 42
num_epochs = 2000

pathes = glob.glob(data_dir + '/*')
x_set = []
y_set = []
split = 0.1

# Data Augmentation with blurring
for p in pathes:
  x = cv2.resize(cv2.imread(p),(img_height,img_width))
  blur = blurring(x.copy())

  x_i = random.randint(0,int(img_height*3/4))
  y_i = random.randint(0,int(img_width*3/4))
  l_h = random.randint(int(1/8 * img_height), int(2/8*img_height))
  l_w = random.randint(int(1/8 * img_width), int(2/8*img_width))
  patch_x = x.copy()
  patch_x[x_i:x_i + l_h,y_i:y_i + l_w] = blur[x_i:x_i + l_h,y_i:y_i + l_w]

 
  x = pre.preprocessing_resnet50v2(x.copy())
  patch_x = pre.preprocessing_resnet50v2(patch_x)
  x_set.append(x)
  y_set.append(x)
  x_set.append(patch_x)
  y_set.append(x)


h = int((1-split) * len(x_set))
zipped = list(zip(x_set,y_set))
random.shuffle(zipped)
x_set, y_set = zip(*zipped)

x_train = np.array(x_set[:h])
y_train = np.array(y_set[:h])

x_val = np.array(x_set[h:])
y_val = np.array(y_set[h:])

train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
val_ds =  tf.data.Dataset.from_tensor_slices((x_val,y_val))

train_ds = train_ds.shuffle(1000,seed_value)
train_ds = train_ds.batch(32)
val_ds = val_ds.shuffle(1000,seed_value)
val_ds = val_ds.batch(len(y_val))

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3),name='encoder_input'),
    tf.keras.layers.Conv2D(filters = num_filter << 0, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters = num_filter << 1, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters = num_filter << 2, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters = num_filter << 3, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2D(filters = num_filter << 4, kernel_size= 3, activation='relu',strides=2,padding='same'),
    #tf.keras.layers.Conv2D(filters = num_filter << 5, kernel_size= 3, activation='relu',strides=2,padding='same'),
  ])

encoder.summary()

decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(8,8,512),name='decoder_input'),
    #tf.keras.layers.Conv2DTranspose(filters = num_filter << 5, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = num_filter << 4, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = num_filter << 3, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = num_filter << 2, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = num_filter << 1, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = num_filter << 0, kernel_size= 3, activation='relu',strides=2,padding='same'),
    tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size=1, activation='tanh',strides = 1, padding='same')
  ])

decoder.summary()
x = tf.keras.Input(shape=(img_height, img_width, 3),name='res_input')
y = encoder(x)
y = decoder(y)

auto_encoder_model = tf.keras.Model(inputs=x,outputs=y)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose = 1)

auto_encoder_model.summary()
print('=========== begin training auto encoder ===========')
auto_encoder_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.MeanAbsoluteError())

auto_encoder_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=num_epochs,
  callbacks=[model_checkpoint_callback]
)



