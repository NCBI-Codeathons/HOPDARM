#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import keras
from pathlib import Path
from numpy.fft import fft, ifft

parser = ArgumentParser()
parser.add_argument('--latest', action='store_true')
args = parser.parse_args()

BATCH_SIZE = 10
BUFFER_SIZE = 100
EPOCHS = 10

def get_npz(base):
    csv = Path(base + '.csv')
    npz = Path('/tmp/hackers00/' + base + '.npz')
    if npz.is_file():
        retval = np.load(npz)
        retval = retval[retval.files[0]]
    else:
        retval = np.genfromtxt(
            csv,
            delimiter=',',
            skip_header=1,
            usecols=range(1,902),
        )
        np.savez(npz, retval)
    return retval

def cepstrum(x, *args):
    spectrum = fft(x, *args)
    log_spectrum = np.log(np.abs(spectrum))
    return ifft(log_spectrum).real

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

enc_data = get_npz('encoding')
ret_data = get_npz('retrieval')

enc_sig = sigmoid(cepstrum(enc_data[:,:-1]))
enc_lab = enc_data[:,-1].astype(int)
ret_sig = sigmoid(cepstrum(ret_data[:,:-1]))
ret_lab = ret_data[:,-1].astype(int)

train_ds = tf.data.Dataset.from_tensor_slices((enc_sig, enc_lab))
test_ds = tf.data.Dataset.from_tensor_slices((ret_sig, ret_lab))

train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

strategy = tf.distribute.MirroredStrategy()
ckpt_dir = Path('/tmp/hackers00/ckpt')
ckpt_fmt = str(ckpt_dir / 'ckpt_{epoch}')

def decay_function(epoch):
    return 1e-2/(epoch+1)

class LearningRateCallback (keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch+1}: learning rate - {model.optimizer.lr.numpy()}')

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=ckpt_fmt, save_weights_only=True, verbose=1),
    keras.callbacks.LearningRateScheduler(decay_function),
    LearningRateCallback(),
]

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(900, input_shape=(900,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

with strategy.scope():
    model = build_model()

    if not args.latest:
        model.fit(
            train_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
        model.save_weights(ckpt_fmt.format(epoch=0))
    else:
        model.load_weights(tf.train.latest_checkpoint(ckpt_dir))
    model.evaluate(test_ds, verbose=2)
