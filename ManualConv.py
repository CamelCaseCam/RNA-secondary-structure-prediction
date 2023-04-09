'''
Defines and trains an autoencoder model that compresses RNA dot plots found in /archiveII/output/dots*.npz
'''

from typing import Literal
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

# keep computer awake
class WindowsInhibitor:
    '''Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

def PreventSleep():
    import ctypes
    print("Preventing Windows from going to sleep")
    ctypes.windll.kernel32.SetThreadExecutionState(
        WindowsInhibitor.ES_CONTINUOUS | \
        WindowsInhibitor.ES_SYSTEM_REQUIRED)

def AllowSleep():
    import ctypes
    print("Allowing Windows to go to sleep")
    ctypes.windll.kernel32.SetThreadExecutionState(
        WindowsInhibitor.ES_CONTINUOUS)


# Define the autoencoder model

NUM_BASES_WINDOW = 16
LATENT_DIM = 10

def make_model():
    inp = tf.keras.Input(shape=(NUM_BASES_WINDOW))
    '''x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 2, activation=tf.nn.leaky_relu)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)'''
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    '''x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)'''
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(LATENT_DIM)(x)

    down_model = tf.keras.Model(inp, x)
    down_model.summary()

    x_up = tf.keras.layers.Dense(LATENT_DIM)(x)
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x_up)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    '''x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 40, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)'''
    x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 50, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    '''x = tf.keras.layers.Dense(NUM_BASES_WINDOW * 2, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2)(x)'''
    out = tf.keras.layers.Dense(NUM_BASES_WINDOW)(x)

    up_model = tf.keras.Model(x_up, out)
    model = tf.keras.Model(inp, out)

    return model, down_model, up_model

model, down_model, up_model = make_model()
model.summary()
down_model.summary()
up_model.summary()

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

def train_step(model, data, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        x = model(data, training=True)
        loss = loss_fn(data, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train_on_example(model, example, loss_fn, optimizer):
    ExampleTotalLoss = 0
    batches = tf.reshape(example, (-1, NUM_BASES_WINDOW))
    return train_step(model, batches, loss_fn, optimizer)

DATAPATH = "./archiveII/output_nums"
BATCH_SIZE = 64
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
                data = np.load(DATAPATH + "/" + file)['arr_0']

                for batch in data:
                    FileLoss += tf.reduce_mean(train_on_example(model, batch, loss_fn, optimizer))
                TotalLoss += FileLoss / len(data)
                print(f'\rFile {filenum}/8', end='', flush=True)
                filenum += 1
        print('')
        print(f"Epoch {epoch}: {TotalLoss}")
        stats_file.write(f"{TotalLoss}\n")


#train(40)

# load the data for model.fit()
data = np.load(DATAPATH + "/dots.npz")['arr_0']
data = tf.reshape(data, (-1, BATCH_SIZE, NUM_BASES_WINDOW))
#TrainData = tf.data.Dataset.from_tensor_slices([data[0:int(len(data) * 0.8)], data[0:int(len(data) * 0.8)]]).shuffle(int(len(data) * 0.8))

#TestData = tf.data.Dataset.from_tensor_slices([data[int(len(data) * 0.8):], data[int(len(data) * 0.8):]]).shuffle(int(len(data) * 0.2))

test_data = data[int(len(data) * 0.8):]
data = data[0:int(len(data) * 0.8)]

PreventSleep()

NUM_EPOCHS = 50

CONTINUE_TRAINING = False
if CONTINUE_TRAINING and __name__ == '__main__':
    model.load_weights("./checkpoints_manual/model")
    down_model.load_weights("./checkpoints_manual/down_model")
    up_model.load_weights("./checkpoints_manual/up_model")

    NUM_EPOCHS = 10


#model.fit(data, data, epochs=NUM_EPOCHS, validation_split=0.1)
#stats_file = open(STATS, 'w')
#stats_file.write('')
#stats_file.close()
if __name__ == "__main__":
    stats_file = open(STATS, 'a')
    for epoch in range(NUM_EPOCHS):
        TotalLoss = 0
        TotalTestLoss = 0
        index = 0
        for step in data:
            TotalLoss += tf.reduce_mean(train_step(model, step, loss_fn, optimizer))
            index += 1
            if index % 100 == 0:
                print(f'\r{index}/{len(data)}', end='', flush=True)
        index = 0
        print('')
        for step in test_data:
            TotalTestLoss += tf.reduce_mean(tf.keras.losses.MeanAbsoluteError()(step, model(step, training=False)))
            index += 1
            if index % 100 == 0:
                print(f'\r{index}/{len(test_data)}', end='', flush=True)
        print('')
        print(f"Epoch {epoch}: {TotalLoss/len(data)}, {TotalTestLoss/len(test_data)}")
        stats_file.write(f"{TotalLoss/len(data)},{TotalTestLoss/len(test_data)}\n")

    model.save_weights("./checkpoints_manual/model")
    down_model.save_weights("./checkpoints_manual/down_model")
    up_model.save_weights("./checkpoints_manual/up_model")

AllowSleep()