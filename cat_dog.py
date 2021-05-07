import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# init necessary variables
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

# 1. Download dataset
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# 2. Data augmentation
image_gen = ImageDataGenerator(tf.keras.applications.mobilenet_v2.preprocess_input,
                               rescale=1./255, rotation_range=40,
                               width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.2, zoom_range=0.2,
                               horizontal_flip=True, fill_mode='nearest')

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir,
                                               shuffle=True, target_size=IMG_SIZE, class_mode='binary')

val_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=validation_dir,
                                             shuffle=True, target_size=IMG_SIZE, class_mode='binary')

# 3. Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

# Freeze base model
base_model.trainable = False

# Connect new predict output to base model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(base_model.input, outputs)

# Set up the learning process
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
# 4. Train
initial_epochs = 10

# save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint('cat vs dog/best.h5', monitor='val_loss',
                                                save_best_only=True, mode='auto')
callback_list = [checkpoint]

# transfer learning
history = model.fit(train_data_gen, epochs=initial_epochs,
                    validation_data=val_data_gen)

# fine tune
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

# unfreeze base model
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

history_fine = model.fit(train_data_gen, epochs=total_epochs, initial_epoch=history.epoch[-1],
                         validation_data=val_data_gen, callbacks=callback_list)

