'''
Defines and trains an autoencoder model that compresses RNA dot plots found in /content/data/dots*.npz
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import tensorflow as tf
from tensorflow.keras import layers

# Define the autoencoder model

def downsample(x, filters, kernel_size, activation = tf.nn.relu, padding = 'same'):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2, 2), activation=activation, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def upsample(x, filters, kernel_size, activation = tf.nn.relu, padding = 'same'):
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=(2, 2), activation=activation, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x

def make_model():
    '''inp = tf.keras.Input(shape=(2048, 2048, 1))
    x = downsample(inp, 8, (3, 3))     # (8, 1024, 1024)
    x = downsample(x, 16, (3, 3))      # (16, 512, 512)
    x = downsample(x, 32, (3, 3))      # (32, 256, 256)
    x = downsample(x, 64, (3, 3))      # (64, 128, 128)
    x = downsample(x, 128, (3, 3))     # (128, 64, 64)
    x = downsample(x, 256, (3, 3))     # (256, 32, 32)
    x = downsample(x, 128, (3, 3))     # (128, 16, 16)
    x = downsample(x, 64, (3, 3))      # (64, 8, 8)

    down_model = tf.keras.Model(inp, x)
    down_model.summary()

    x_up = upsample(x, 128, (3, 3))       # (128, 16, 16)
    x = upsample(x_up, 256, (3, 3))       # (256, 32, 32)
    x = upsample(x, 128, (3, 3))       # (128, 64, 64)
    x = upsample(x, 64, (3, 3))        # (64, 128, 128)
    x = upsample(x, 32, (3, 3))        # (32, 256, 256)
    x = upsample(x, 16, (3, 3))        # (16, 512, 512)
    x = upsample(x, 8, (3, 3))         # (8, 1024, 1024)
    x = upsample(x, 1, (3, 3))         # (1, 2048, 2048)
    out = tf.keras.layers.Conv2D(1, (3, 3), activation=tf.nn.sigmoid, padding='same')(x)

    up_model = tf.keras.Model(x_up, out)
    model = tf.keras.Model(inp, out)'''

    down_model = tf.keras.Sequential([
        layers.Input((2048, 2048, 1)),
        layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=2),              # 1024, 1024, 8
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),             # 512, 512, 16
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),             # 256, 256, 64
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),             # 128, 128, 32
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),             # 64, 64, 16
        layers.BatchNormalization(),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),              # 32, 32, 8
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),                        # 32, 32, 16
        layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),              # 16, 16, 8
    ])
    up_model = tf.keras.Sequential([
        layers.Input((16, 16, 8)),
        layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),     # 32, 32, 8
        layers.BatchNormalization(),
        layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),    # 64, 64, 16
        layers.BatchNormalization(),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),                         # 64, 64, 8
        layers.BatchNormalization(),
        layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),    # 128, 128, 16
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),    # 256, 256, 32
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),    # 512, 512, 64
        layers.BatchNormalization(),
        layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),    # 1024, 1024, 32
        layers.BatchNormalization(),
        layers.Conv2DTranspose(2, (3, 3), activation='relu', padding='same', strides=2),     # 2048, 2048, 4
        layers.Conv2D(1, (3,3), activation='relu', padding='same')                           # 2048, 2048, 1
    ])

    inp = tf.keras.Input(shape=(2048, 2048, 1))
    model = tf.keras.Model(inp, up_model(down_model(inp)))

    return model, down_model, up_model

model, down_model, up_model = make_model()
model.summary()
down_model.summary()
up_model.summary()

# Train the model (custom training loop)

DATAPATH = "./archiveII/output"
BATCH_SIZE = 5
# just absolute error
def loss_fn(x, y):
  return tf.reduce_sum(tf.abs(tf.cast(x, dtype=tf.float32) - tf.cast(y, dtype=tf.float32)) ** 2) / BATCH_SIZE

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(model, data, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        x = model(data, training=True)
        loss = loss_fn(data, x)
        del x
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    del gradients
    return loss

import random
def train(epochs):
    files = os.listdir(DATAPATH)
    print(files)
    for epoch in range(epochs):
        TotalLoss = 0
        random.shuffle(files)
        filenum = 1
        for file in files:
            FileLoss = 0
            if file.endswith('.npz'):
                npdata = np.load(DATAPATH + "/" + file)['arr_0']
                dataset = tf.data.Dataset.from_tensor_slices(npdata)
                data = dataset.batch(BATCH_SIZE)

                for batch in data:
                    FileLoss += train_step(model, tf.expand_dims(batch, axis=-1), loss_fn, optimizer)
                TotalLoss += FileLoss / len(data)
                print(f'\rFile {filenum}/63', end='', flush=True)
                filenum += 1
                del data
                del dataset
                del npdata
        print('')
        print(f"Epoch {epoch}: {TotalLoss}")


model.load_weights("./dotplotencoder/checkpoints/weights/model")
down_model.load_weights("./dotplotencoder/checkpoints/weights/down_model")
up_model.load_weights("./dotplotencoder/checkpoints/weights/up_model")


# Evaluate model on random training data
def Evaluate(num):
  files = os.listdir(DATAPATH)
  random.shuffle(files)

  data = tf.data.Dataset.from_tensor_slices(np.load(DATAPATH + "/" + files[0])['arr_0'])
  for i in range(num):
    for example in data.take(1):
      # execute the model
      output = model(tf.expand_dims(tf.expand_dims(example, axis=0), axis=-1))
      plt.imshow(example, cmap="gray", interpolation="none")
      plt.imshow(tf.squeeze(tf.squeeze(output, axis = 0), axis=-1), cmap="gray", interpolation="none")
      plt.show()
      
Evaluate(5)