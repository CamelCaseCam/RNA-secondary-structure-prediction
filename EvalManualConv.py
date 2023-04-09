'''
Defines and trains an autoencoder model that compresses RNA dot plots found in /archiveII/output/dots*.npz
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import gc
import tensorflow as tf

# Define the autoencoder model

NUM_BASES_WINDOW = 16

from ManualConv import make_model

model, down_model, up_model = make_model()
model.summary()
down_model.summary()
up_model.summary()

loss_fn = tf.keras.losses.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


model.load_weights("./checkpoints_manual/model")
down_model.load_weights("./checkpoints_manual/down_model")
up_model.load_weights("./checkpoints_manual/up_model")

DATAPATH = "./archiveII/output_nums"
BATCH_SIZE = 32
STATS = "D:\Development\RNAFolding\Stats.csv"

SHIFT_DATA = True

import random
def Evaluate(num, show_output = False):
  data = np.load('./archiveII/output_nums/dots.npz')['arr_0']
  random.shuffle(data)
  data = tf.reshape(data, (-1, 1, NUM_BASES_WINDOW))

  for i in range(num):
    example = data[i]
    output = model(example)
    if show_output:
        print(zip(example, output).__next__())
    print(loss_fn(example, output))
    print(tf.reduce_mean(example))
    print(tf.reduce_mean(output))
    print('_________________________')

Evaluate(2, True)