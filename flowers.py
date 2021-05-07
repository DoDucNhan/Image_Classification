import os
import glob
import shutil
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# init necessary variables
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
num_classes = 5

# 1. Download dataset
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
path_to_zip = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
base_dir = os.path.join(os.path.dirname(path_to_zip), 'flower_photos')
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

# create train and validation set
for cl in classes:
    img_path = os.path.join(base_dir, cl)
    images = glob.glob(img_path + '/*.jpg')
    # print("{}: {} Images".format(cl, len(images)))
    num_train = int(round(len(images)*0.8))
    train, val = images[:num_train], images[num_train:]

    for t in train:
        if not os.path.exists(os.path.join(base_dir, 'train', cl)):
            os.makedirs(os.path.join(base_dir, 'train', cl))
        shutil.move(t, os.path.join(base_dir, 'train', cl))

    for v in val:
        if not os.path.exists(os.path.join(base_dir, 'val', cl)):
            os.makedirs(os.path.join(base_dir, 'val', cl))
        shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# 2. Data augmentation
image_gen = ImageDataGenerator(tf.keras.applications.mobilenet_v2.preprocess_input,
                               rescale=1./255, rotation_range=45,
                               width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.3, zoom_range=0.5, horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=train_dir,
                                               shuffle=True, target_size=IMG_SIZE, class_mode='sparse')

val_data_gen = image_gen.flow_from_directory(batch_size=BATCH_SIZE, directory=val_dir,
                                             shuffle=True, target_size=IMG_SIZE, class_mode='sparse')

# 3. Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE,
                                                            weights='imagenet', include_top=False)
# Freeze base model
base_model.trainable = False

# Connect new predict output to base model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(base_model.inputs, outputs)

# Set up the learning process
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Train
epochs = 10
# save best model
checkpoint = tf.keras.callbacks.ModelCheckpoint('flowers/best.h5', monitor='val_loss',
                                                save_best_only=True, mode='auto')
callback_list = [checkpoint]

history = model.fit(train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=callback_list)
