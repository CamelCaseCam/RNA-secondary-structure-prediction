'''
Model that predicts RNA secondary structures (archiveII/output_directional/dots.npz) from RNA sequences 
(archiveII/output_directional/seqs.npz). This model will use a combination of convolutional and dense layers
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load data
dots = np.load('archiveII/output_directional/dots.npz')['arr_0']
seqs = np.load('archiveII/output_directional/seqs.npz')['arr_0']

# Split data into training and testing sets
train_dots = dots[:int(len(dots) * 0.8)]
train_seqs = seqs[:int(len(seqs) * 0.8)]
test_dots = dots[int(len(dots) * 0.8):]
test_seqs = seqs[int(len(seqs) * 0.8):]

# shuffle the training data
train_dots, train_seqs = tf.data.Dataset.from_tensor_slices((train_dots, train_seqs)).shuffle(10000).batch(32)


DOTS_STDDEV = 0.05
SEQS_STDDEV = 0.15
# add noise to the training data
def add_noise(x, stddev):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=tf.float32)
    return x + noise

train_dots = add_noise(train_dots, DOTS_STDDEV)
train_seqs = add_noise(train_seqs, SEQS_STDDEV)


def DownBlock(x, filters, kernel_size, strides, padding, activation, dropout):
    x = layers.Conv1D(filters, kernel_size, 1, padding, activation=activation)(x)
    x = layers.Conv1D(filters, kernel_size, strides, padding, activation=activation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    return x

def UpBlock(x, x_skip, filters, kernel_size, strides, padding, activation, dropout):
    x = layers.Conv1D(filters, kernel_size, 1, padding, activation=activation)(x)
    x = layers.Conv1DTranspose(filters, kernel_size, strides, padding, activation=activation)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Concatenate()([x, x_skip])
    return x


THRESHOLD = 0.25
NONZERO_WEIGHT = 5
ZERO_WEIGHT = 1
VARIANCE_WEIGHT = 1.0
# Define loss function
def loss(x, y_true, y_pred):
    '''
    This loss function works like mean absolute error, but it has different weights for where y is 0 and where y is nonzero.
    '''
    nonzero_mask = tf.where(abs(y_true) > THRESHOLD, tf.ones_like(y_true), tf.zeros_like(y_true))
    zero_mask = tf.where(abs(y_true) <= THRESHOLD, tf.ones_like(y_true), tf.zeros_like(y_true))

    nonzero_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * nonzero_mask)
    zero_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * zero_mask)

    # Add a term for variance between zero and nonzero values
    variance = tf.abs(tf.reduce_mean(tf.abs(y_pred) * nonzero_mask) + tf.reduce_mean(tf.abs(y_pred) * zero_mask))

    return NONZERO_WEIGHT * nonzero_loss + ZERO_WEIGHT * zero_loss + VARIANCE_WEIGHT * variance


# Make model
DATA_SHAPE = (2048, 6)
# Create model
def make_model():
    inp = layers.Input(shape=DATA_SHAPE)
    noisy_x = layers.GaussianNoise(0.25)(inp)

    # Head 1
    x1 = DownBlock(noisy_x, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    x2 = DownBlock(x1, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x3 = DownBlock(x2, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x4 = DownBlock(x3, 128, 3, 2, 'same', 'relu', 0.4)    # 128
    x5 = DownBlock(x4, 256, 3, 2, 'same', 'relu', 0.4)    # 64
    x6 = DownBlock(x5, 128, 3, 2, 'same', 'relu', 0.4)    # 32
    x7 = DownBlock(x6, 256, 5, 2, 'same', 'relu', 0.4)    # 16
    x8 = DownBlock(x7, 128, 5, 2, 'same', 'relu', 0.4)    # 8
    x9 = DownBlock(x8, 256, 5, 2, 'same', 'relu', 0.4)    # 4

    x10 = DownBlock(x9, 512, 5, 2, 'same', 'relu', 0.4)    # 2

    x_up_1 = UpBlock(x10, x9, 256, 5, 2, 'same', 'relu', 0.4)    # 4
    x_up_2 = UpBlock(x_up_1, x8, 128, 5, 2, 'same', 'relu', 0.4)    # 8
    x_up_3 = UpBlock(x_up_2, x7, 256, 5, 2, 'same', 'relu', 0.4)    # 16
    x_up_4 = UpBlock(x_up_3, x6, 128, 3, 2, 'same', 'relu', 0.4)    # 32
    x_up_5 = UpBlock(x_up_4, x5, 256, 3, 2, 'same', 'relu', 0.4)    # 64
    x_up_6 = UpBlock(x_up_5, x4, 128, 3, 2, 'same', 'relu', 0.4)    # 128
    x_up_7 = UpBlock(x_up_6, x3, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x_up_8 = UpBlock(x_up_7, x2, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x_up_9 = UpBlock(x_up_8, x1, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    Head1Out = UpBlock(x_up_9, noisy_x, 8, 3, 2, 'same', 'relu', 0.1)    # 2048


    # Head 2
    x1_2 = DownBlock(noisy_x, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    x1_2 = layers.Concatenate()([x1_2, x_up_9])
    x2_2 = DownBlock(x1_2, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x2_2 = layers.Concatenate()([x2_2, x_up_8])
    x3_2 = DownBlock(x2_2, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x3_2 = layers.Concatenate()([x3_2, x_up_7])
    x4_2 = DownBlock(x3_2, 128, 3, 2, 'same', 'relu', 0.4)    # 128
    x4_2 = layers.Concatenate()([x4_2, x_up_6])
    x5_2 = DownBlock(x4_2, 256, 3, 2, 'same', 'relu', 0.4)    # 64
    x5_2 = layers.Concatenate()([x5_2, x_up_5])
    x6_2 = DownBlock(x5_2, 128, 3, 2, 'same', 'relu', 0.4)    # 32
    x6_2 = layers.Concatenate()([x6_2, x_up_4])

    x7_2 = DownBlock(x6_2, 256, 5, 2, 'same', 'relu', 0.4)    # 16

    x_up_21 = UpBlock(x7_2, x6_2, 128, 3, 2, 'same', 'relu', 0.4)    # 32
    x_up_22 = UpBlock(x_up_21, x5_2, 256, 3, 2, 'same', 'relu', 0.4)    # 64
    x_up_23 = UpBlock(x_up_22, x4_2, 128, 3, 2, 'same', 'relu', 0.4)    # 128
    x_up_24 = UpBlock(x_up_23, x3_2, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x_up_25 = UpBlock(x_up_24, x2_2, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x_up_26 = UpBlock(x_up_25, x1_2, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    Head2Out = UpBlock(x_up_26, noisy_x, 8, 3, 2, 'same', 'relu', 0.1)    # 2048

    # Head 3
    x1_3 = DownBlock(noisy_x, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    x1_3 = layers.Concatenate()([x1_3, x_up_26])
    x2_3 = DownBlock(x1_3, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x2_3 = layers.Concatenate()([x2_3, x_up_25])
    x3_3 = DownBlock(x2_3, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x3_3 = layers.Concatenate()([x3_3, x_up_24])
    x4_3 = DownBlock(x3_3, 128, 3, 2, 'same', 'relu', 0.4)    # 128

    x5_3 = DownBlock(x4_3, 256, 3, 2, 'same', 'relu', 0.4)    # 64

    x_up_31 = UpBlock(x5_3, x4_3, 128, 3, 2, 'same', 'relu', 0.4)    # 128
    x_up_32 = UpBlock(x_up_31, x3_3, 64, 3, 2, 'same', 'relu', 0.4)    # 256
    x_up_33 = UpBlock(x_up_32, x2_3, 32, 3, 2, 'same', 'relu', 0.4)    # 512
    x_up_34 = UpBlock(x_up_33, x1_3, 16, 3, 2, 'same', 'relu', 0.4)    # 1024
    Head3Out = UpBlock(x_up_34, noisy_x, 8, 3, 2, 'same', 'relu', 0.1)    # 2048

    # Concatenate heads
    x = layers.Concatenate()([Head1Out, Head2Out, Head3Out])
    x = layers.Conv1D(8, 3, padding='same', activation='linear')(x)


    out = layers.Conv1D(1, 1, 1, 'same', activation='tanh')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss=loss)
    return model

model = make_model()

# Train model
model.fit(train_seqs, train_dots, validation_data=(test_seqs, test_dots), epochs=50, batch_size=32)
