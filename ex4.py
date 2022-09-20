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
# #  Using DL model to fit the GEV output
# -

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
tf.__version__

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

tf.config.list_physical_devices()

# # Load the data 

# + tags=[]
loc=10.0
file1='loc'+str(loc)+'.nc'
da=xr.open_dataset(file1)
file1
# -

da

test_results = {}


# +
# use a random normal set of x's to define the data
xt=np.random.randn(400,2)
xt.shape

yt=xt[:,0]*0.
yt[:]= 1.9*xt[:,0]  + xt[:,1]*.5 + xt[:,0]*xt[:,1]**2#

print(yt.shape,xt.shape)

plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.plot(xt[:,0],label='x1')
plt.plot(xt[:,1],label='x2')
plt.plot(yt,label='y')
plt.legend()

plt.subplot(1,4,2)
plt.hist(xt, bins=10)
plt.xlabel('Value ')
plt.ylabel('Count')

plt.subplot(1,4,3)
plt.scatter(xt[:,0],xt[:,1] )
plt.xlabel('X1')
plt.ylabel('X2')

plt.subplot(1,4,4)
plt.hist(yt, bins=10)
plt.xlabel('Value ')
plt.ylabel('Count')

# -

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(xt))
print(normalizer.mean.numpy())


# + tags=[]
#help(normalizer)
# -

linear_model = tf.keras.Sequential([
    normalizer, tf.keras.layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# %%time
history = linear_model.fit(
    xt,yt,epochs=100, verbose=0, validation_split = 0.2)


# +
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss ')
  plt.legend()
  plt.grid(True)


plot_loss(history)
# -

test_results['linear_model'] = linear_model.evaluate(xt,yt, verbose=0)
test_results

linear_model.get_weights()
#linear_model.summary()

# +
yp = linear_model.predict(xt).flatten()

plt.scatter(yp, yt)
plt.plot(yt,yt,'k')
plt.xlabel('Predicted')
plt.ylabel('Data')
plt.legend()

print(yp.shape,yt.shape)
# -


error = yp - yt
plt.hist(error, bins=20)
plt.xlabel('Prediction Error ')
plt.ylabel('Count')

# # Organise the data to use in the DL model 

# + tags=[]
asol=da.asol    # % return value error
t_ret=da.t_ret  # true solution
i_ret=da.i_ret  # solution for each iteration
aerr=i_ret.std(axis=3)/i_ret.mean(axis=3) # std/mean from iterations

n=asol.coords['size'].values
ashp=asol.coords['shape'].values*(-1)  # GEV code uses the opposite for shape parameter
ascl=asol.coords['scale'].values
ari=asol.coords['ari'].values
rtmp=aerr*100*2

# +
# rtmp is the relative error as a function of 
# (scale,shape,Sample size, return period)
# for a varying scale, shape, return period the relative error for the return period

nret=1; nsh=6; nsc=0
for j in range(11):
#    plt.plot(xscale("log"))
    rtmp[nsc,j,:,1].plot.line(xscale="log",label=str(ashp[j]))

plt.legend()


# -

rtmp

import itertools


# ##  Unroll the 4d rtmp into a 1d array

# +
# recreate the 3 d array
def unroll(yroll,rtmp):
    nsize=rtmp.shape
    atmp=rtmp.copy()
    l=-1
#    atmp=np.zeros([nsize[0],nsize[1],nsize[2]])
    for i,j,k,m in itertools.product(range(1),range(nsize[1]),range(nsize[2]),range(nsize[3])):
        l=l+1 #print(i,j,k)
        atmp[i,j,k,m]=yroll[l]
        
    return atmp

# create the data
def xroll(rtmp):
    nsize=rtmp.shape
    xdata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3],4])
    ydata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3]])
    x1=rtmp.coords['scale'].values
    x2=rtmp.coords['shape'].values *(-1) # convert to stand GEV convention 
    x3=rtmp.coords['size'].values
    x4=rtmp.coords['ari'].values
    l=-1
    for i,j,k,m in itertools.product(range(1),range(nsize[1]),range(nsize[2]),range(nsize[3])):
        l=l+1 #print(i,j,k)
        xdata[l,0]=x1[i] #np.log(x1[i])
        xdata[l,1]=x2[j] # np.log(x2[j])
        xdata[l,2]=x3[k]
        xdata[l,3]=x4[m]
        ydata[l]=rtmp[i,j,k,m].values
        
    return xdata,ydata


# -

xtmp,ytmp=xroll(rtmp)
print(xtmp.shape)

# + tags=[]
i=sample(range(0, 3300), 3300)
print(i[0:100])

# +
xt=np.copy(xtmp)
yt=np.copy(ytmp)
xt[:,0:4]=xtmp[i[:],0:4]
yt[:]=ytmp[i[:]]

print(xt.shape)
# -

j=3
plt.plot(xt[:,j],yt[:],'rx')
plt.plot(xtmp[:,j],ytmp[:],'bo')


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Small 

# +
# build and fit model and plot diagnostics
def dnn(normalizer,loss,xt,yt):
    dnn_model = build_and_compile_model(normalizer,loss)
    history = dnn_model.fit(xt,yt,validation_split=0.5, verbose=0, epochs=400)
    plot_loss(history) 
    
    return dnn_model,history

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss ')
  plt.legend()
  plt.grid(True)

    
def plot_scatter(dnn_model,xt,yt):    
    yp = dnn_model.predict(xt).flatten()
    error= yp -yt
    plt.figure(figsize=(15,5))
#    
    plt.subplot(1,2,1)
    plt.scatter(yp, yt, )
    plt.plot(yt,yt,'k')
    plt.xlabel('Predicted')
    plt.ylabel('Data')
    plt.legend()
    print('Predicted max=',np.max(yp))
    print('Data max =', np.max(yt))
#
    plt.subplot(1,2,2)
    plt.hist(error, bins=20)
    plt.xlabel('Prediction Error ')
    plt.ylabel('Count')
    return
    



# +
# %%time
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(xt))
print(normalizer.mean.numpy())

loss='mean_squared_error'
loss='mean_absolute_error'

def build_and_compile_model(norm,loss):
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(4, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(0.004))
  return model

s1,h1=dnn(normalizer,loss,xt,yt)
plot_scatter(s1,xt,yt)
test_results['small'] = s1.evaluate(xt,yt, verbose=0)


# -

# # Small 2 

# +

def build_and_compile_model(norm,loss):
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(4, activation='relu'),
      tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(0.004))
  return model

s2,hs2=dnn(normalizer,loss,xt,yt)
plot_scatter(s2,xt,yt)
test_results['small 2'] = s2.evaluate(xt,yt, verbose=0)


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Medium

# + tags=[]
def build_and_compile_model(norm,loss):
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(8, activation='relu'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(0.004))
  return model

m1,hm1=dnn(normalizer,loss,xt,yt)
plot_scatter(m1,xt,yt)
test_results['medium'] = m1.evaluate(xt,yt, verbose=0)


# + [markdown] tags=[]
# # Medium with regularisation 

# +
def build_and_compile_model(norm,loss):
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(8, activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(0.00000)),
      tf.keras.layers.Dense(4, activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(0.00000)),
      tf.keras.layers.Dense(8, activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(0.00000)),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(0.004))
  return model

mr1,hmr1=dnn(normalizer,loss,xt,yt)
plot_scatter(mr1,xt,yt)
test_results['medium with reg'] = mr1.evaluate(xt,yt, verbose=0)


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Medium with dropout 

# +
def build_and_compile_model(norm,loss):
  model = tf.keras.Sequential([
      norm,tf.keras.layers.Dense(16, activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(0.0000)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(64, activation='relu',
      kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(1)
  ])
  model.compile(loss=loss,
                optimizer=tf.keras.optimizers.Adam(0.004))
  return model

mrd1,hmrd1=dnn(normalizer,loss,xt,yt)
plot_scatter(mrd1,xt,yt)
test_results['medium with reg and dropout'] = mrd1.evaluate(xt,yt, verbose=0)
# -

# # Results 

# + tags=[]
test_results
# -

s2.summary()
#s2.get_weights()
#m1.summary()
mr1.summary()

#plot_loss(h1)
plot_loss(hs2)
#plot_loss(hm1)
plot_loss(hmr1)
#plot_loss(hmrd1)

# #
# {'small': 0.06398718059062958,
#  'small 2': 0.05341784283518791,
#  'medium': 0.05726228281855583,
#  'medium with reg': 0.0721876472234726,
#  'medium with reg and dropout': 0.16216889023780823} 


