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
# ## Callable version of a DL model to fit the GEV output
#
# a version strip of plotting output
# -

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
#from numpy.random import seed
from random import seed
#tf.__version__
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.list_physical_devices()

# %load_ext autoreload
# %autoreload 2

# +
import fit_lib 
import sys

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

from sys import argv

if is_interactive():
    params = [0, 0.001, 0.0042, 400, 8,16,16 ]
    params = [1, 0.000, 0.0040, 160, 8,16,8 ]
    params = [1, 0.000, 0.0040, 161, 64,64,64 ]
else:
    print(argv)
    ss=argv[1:]
    print(ss)
    params = [float(i) for i in ss]

if params[0] == 0:
    loss='mean_squared_error'
else:
    loss='mean_absolute_error'
    
print('params ',params)
print('loss=', loss)
print(is_interactive())
# -

reg = params[1]*1.
learn=params[2]*1.
epochs=int(params[3])
layers= np.array(params[4:]) 
print(reg,learn,epochs,layers,loss)

# remove layers with zero nodes
i=np.argwhere( layers[:] > 0  ) [:,0]
ll=layers[i]
layers=ll

# # Load the data 

# + tags=[]
loc=0.0
file1='loc'+str(loc)+'.nc'
da=xr.open_dataset(file1)
file1
# -

test_results = {}


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
# redefine relative error based on the (95% - 5% / 50% ) values times 100 to get percent 
a50=i_ret.reduce(np.percentile,axis=3,q=50)
a5=i_ret.reduce(np.percentile,axis=3,q=5)
a95=i_ret.reduce(np.percentile,axis=3,q=95)
rtmp=100*((a95-a5)/a50) *.5  # to reflect +/-

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##  Unroll the 4d rtmp into a 1d array

# +
# turn data into form suitable for DL
seed(2)
xtmp,ytmp=fit_lib.xroll1(rtmp)
print(xtmp.shape)
nt=ytmp.size
i=sample(range(0, nt), nt)

# resample data for training and validation
xt=np.copy(xtmp[:,0:4])
yt=np.copy(ytmp)
xt[:,0:4]=xtmp[i[:],0:4]
yt[:]=ytmp[i[:]]

print(i[0:10])
print(xt.shape,yt.shape)

# + [markdown] tags=[]
# # Deep Learning Model 
#
# configure and fit

# + tags=[]
# #%%time
#loss='mean_squared_error'
#loss='mean_absolute_error'
seed(1)
#reg=0.0
#learn=0.0035
#epochs=160
#layers=np.array([8,4,8])
#layers=np.array([16,16])
print(reg,learn,epochs,layers,loss)

from keras.callbacks import LearningRateScheduler

def schedule(epoch,lr):
    if epoch < 50:
        return lr
    elif epoch > 121:
        return 0.0001*.1
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(schedule,0)


s1,h1=fit_lib.dnn(loss,reg,learn,epochs,layers, xt,yt, 1, lr_scheduler)

# basic plotting output
fit_lib.plot_loss(h1) 
error=fit_lib.plot_scatter(s1,xt,yt)
#
a=s1.evaluate(xt,yt, verbose=0)
test_results['small1'] = a
print('fit=', a)
print('max error',np.max(error),np.min(error))
# -


learn=0.0001
s1.compile(loss=loss,  optimizer=tf.keras.optimizers.Adam(learn) )
history = s1.fit(xt,yt,validation_split=0.2, verbose=0, epochs=20)
fit_lib.plot_loss(history) 
s1.optimizer.get_config()
a=s1.evaluate(xt,yt, verbose=0)
print('fit=', a)

# + tags=[]
learn=0.0001*.1
s1.compile(loss=loss,  optimizer=tf.keras.optimizers.Adam(learn) )
history = s1.fit(xt,yt,validation_split=0.2, verbose=0, epochs=30)
fit_lib.plot_loss(history) 
s1.optimizer.get_config()
a=s1.evaluate(xt,yt, verbose=0)
print('fit=', a)
print('save model ')
s1.save('model_0' )
# -

error=fit_lib.plot_scatter(s1,xt,yt)

# +
aa=(s1.get_weights())
sum=0;sum1=0
for m in range(0,len(aa)) :
    sum=aa[m].size +sum
    
for m in range(3,len(aa)) :
    sum1=aa[m].size +sum1

# total and trainable parameters
print (sum,sum1)
number_unknowns=sum1

# +
original_stdout = sys.stdout # Save a reference to the original standard output
print(reg,learn,epochs,layers,a,np.max(error),np.min(error))

if ( layers.size < 3):
    ltmp=np.append(layers,[0])
    layers=ltmp
#print(layers[0],layers[1],layers[2])

with open('his.txt', 'a') as fout:
    sys.stdout = fout # Change the standard output to the file we created.
    print(reg,learn,epochs,layers,a,np.max(error),np.min(error),number_unknowns)
    sys.stdout = original_stdout # Reset the standard output to its ori
# -

s1.summary()

s1.get_weights()[2] # number of samples


