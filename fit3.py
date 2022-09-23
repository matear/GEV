# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# ## Module for DL model used to fit the GEV output
#
# modifying the file to turn it into a module.
# -

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
from numpy.random import seed
#tf.__version__
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.list_physical_devices()
dev = '/gpu:0'


# + [markdown] tags=[]
# # Configure Deep Learning Model 

# +
# build and fit model and plot diagnostics
def build_and_compile_model(norm,loss,reg,learn,layers):
  print(layers.shape)
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(layers[0], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)),
      tf.keras.layers.Dense(layers[1], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)),
      tf.keras.layers.Dense(layers[2], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(learn))
  return model

def dnn(loss,reg,learn,epochs,layers,xt,yt):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(xt))
    print(normalizer.mean.numpy())
    with tf.device(dev):
        dnn_model = build_and_compile_model(normalizer,loss,reg,learn,layers)
        history = dnn_model.fit(xt,yt,validation_split=0.5, verbose=0, epochs=epochs)
    
    return dnn_model,history


# +
hidden = 128

with tf.device(dev):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hidden, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(hidden, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# +
model = tf.keras.models.Sequential()
model.add(norm,tf.keras.layers.Dense(layers[0], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)) )
model.add(tf.keras.layers.Dense(layers[1], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)) )
model.add(tf.keras.layers.Dense(layers[2], activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(reg)) )
model.add(tf.keras.layers.Dense(1))


model1 = tf.keras.Sequential()
# -


