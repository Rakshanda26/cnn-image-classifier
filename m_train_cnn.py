import tensorflow as tf 
from tensorflow.keras import layers , models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128,128)
BATCH_SIZE = 16


train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)


train_data = train_datagen.flow_from_directory(
    'data/train',
    target_size = IMG_SIZE,
    BATCH_SIZE = BATCH_SIZE,
    class_mode = 'categorical'
)

val_data = val_datagen.flow_from_directory(
    'data/val',
    target_size = IMG_SIZE,
    BATCH_SIZE = BATCH_SIZE,
    class_mode = 'categorical'
)

num_class = len(train_data.class_indices)

models.Sequential(
    layers.Conv2d()
)