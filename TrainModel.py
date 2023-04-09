'''
Defines and trains an autoencoder model that compresses RNA dot plots found in /archiveII/output/dots*.npz
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

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
    inp = tf.keras.Input(shape=(2048, 2048, 1))
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
    model = tf.keras.Model(inp, out)

    return model, down_model, up_model

model, down_model, up_model = make_model()
model.summary()
down_model.summary()
up_model.summary()

# Train the model (custom training loop)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(model, data, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        x = model(data)
        loss = loss_fn(data, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

DATAPATH = "./archiveII/output"
import random
def train(epochs):
    files = os.listdir(DATAPATH)
    print(files)
    for epoch in range(epochs):
        TotalLoss = 0
        random.shuffle(files)
        for file in files:
            FileLoss = 0
            if file.endswith('.npz'):
                data = np.load(DATAPATH + "/" + file)['arr_0']
                for plot in data:
                    FileLoss += train_step(model, tf.expand_dims(tf.expand_dims(plot, axis=-1), axis=0), loss_fn, optimizer)
                TotalLoss += FileLoss / len(data)
                print('.', end='', flush=True)
        print('')
        print(f"Epoch {epoch}: {TotalLoss}")


train(10)