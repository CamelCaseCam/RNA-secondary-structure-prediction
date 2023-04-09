'''
Defines and trains an autoencoder model that compresses RNA dot plots found in /archiveII/output/dots*.npz
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

# Define the autoencoder model

NUM_BASES_INPUT = 2048

def make_model():
    inp = tf.keras.Input(shape=(NUM_BASES_INPUT))
    x = tf.keras.layers.Dense(NUM_BASES_INPUT, activation=tf.nn.relu)(inp)
    x = tf.keras.layers.Dense(1024 / 2, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)
    #x = tf.keras.layers.Dense(256 / 2, activation=tf.nn.relu)(x)

    down_model = tf.keras.Model(inp, x)
    down_model.summary()

    #x_up = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)         # 256
    x_up = tf.keras.layers.Dense(512, activation=tf.nn.relu)(x)      # 512
    out = tf.keras.layers.Dense(1024 / 2, activation=tf.nn.relu)(x_up) # 1024
    out = tf.keras.layers.Dense(NUM_BASES_INPUT, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dense(NUM_BASES_INPUT)(out)

    up_model = tf.keras.Model(x_up, out)
    model = tf.keras.Model(inp, out)

    return model, down_model, up_model

model, down_model, up_model = make_model()
model.summary()
down_model.summary()
up_model.summary()

# Train the model (custom training loop)
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(model, data, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        x = model(data)
        loss = loss_fn(data, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

DATAPATH = "./archiveII/output_nums"
BATCH_SIZE = 32
STATS = "D:\Development\RNAFolding\Stats.csv"

import random
def train(epochs):
    files = os.listdir(DATAPATH)
    print(files)
    stats_file = open(STATS, 'w')
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
                    FileLoss += tf.reduce_sum(train_step(model, batch, loss_fn, optimizer))
                TotalLoss += FileLoss / len(data)
                print(f'\rFile {filenum}/8', end='', flush=True)
                filenum += 1
                del data
                del dataset
                del npdata
        print('')
        print(f"Epoch {epoch}: {TotalLoss}")
        stats_file.write(f"{TotalLoss}\n")


train(500)

model.save_weights("./checkpoints/model")
down_model.save_weights("./checkpoints/down_model")
up_model.save_weights("./checkpoints/up_model")


# Test the model
# Evaluate model on random training data
def Evaluate(num):
  files = os.listdir(DATAPATH)
  random.shuffle(files)

  data = tf.data.Dataset.from_tensor_slices(np.load(DATAPATH + "/" + files[0])['arr_0'])
  for i in range(num):
    for example in data.take(1):
        output = model(tf.expand_dims(example, 0))
        print(example)
        print(output)
        print(loss_fn(example, output))
        print('_________________________')