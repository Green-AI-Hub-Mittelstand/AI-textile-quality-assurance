import tensorflow as tf
import libs.preprocessing as helper

# Set data_dir to your data set directory directory with labeld subdirectories:
# data_dir:
#   - label_00
#   - label_01
data_dir = 'path_to_folder_with_labeled_sub_directories'

# Model Checkpoint name strategy
checkpoint_filepath = './transfer_net_{epoch:04d}-{loss:.4f}-{val_loss:.4f}-{accuracy:.4f}-{val_accuracy:.4f}.keras'

# Parameters
batch_size = 32
img_height = 224
img_width = 224
seed_value = 42
num_epochs = 100
num_classes = 2
label_mode_val = 'categorical'

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode=label_mode_val,
  validation_split=0.2,
  subset="training",
  seed=seed_value,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  label_mode=label_mode_val,
  subset="validation",
  seed=seed_value,
  image_size=(img_height, img_width),
  batch_size=batch_size)

train_ds = train_ds.map(lambda x, y: (helper.preprocessing_resnet50v2(x),y))
val_ds = val_ds.map(lambda x, y: (helper.preprocessing_resnet50v2(x),y))

backbone = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None
)
backbone.trainable = False

output =  tf.keras.Sequential([
    tf.keras.Input(shape=(7,7,2048)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')])
output.summary()

x = tf.keras.Input(shape=(224, 224, 3),name='res_input')
y = backbone(x)
y = output(y)

res50model = tf.keras.Model(inputs=x,outputs=y)

res50model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose = 1)

print('===========begin training res_net_50_v2===========')
res50model.compile(
  optimizer='adam',
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'])

res50model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=num_epochs,
   callbacks = [model_checkpoint_callback]
)



