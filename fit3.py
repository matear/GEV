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
from numpy.random import seed
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
    params = [0.001, 0.0042, 400, 8,16,0 ]
else:
    print(argv)
    ss=argv[1:]
    print(ss)
    params = [float(i) for i in ss]
   
    
print('params loss,reg,learn,epochs,layers',params)
print(is_interactive())
# -

reg = params[0]*1.
learn=params[1]*1.
epochs=int(params[2])
layers= np.array(params[3:]) 
print(reg,learn,epochs,layers)

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

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ##  Unroll the 4d rtmp into a 1d array

# +
# turn data into form suitable for DL
seed(2)
xtmp,ytmp=fit_lib.xroll(rtmp)
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
loss='mean_squared_error'
loss='mean_absolute_error'
seed(1)
#reg=0.0
#learn=0.0035
#epochs=200
#layers=np.array([8,4,8])
#layers=np.array([16,16])
print(reg,learn,epochs,layers)

s1,h1=fit_lib.dnn(loss,reg,learn,epochs,layers, xt,yt)

# basic plotting output
fit_lib.plot_loss(h1) 
error=fit_lib.plot_scatter(s1,xt,yt)
#
a=s1.evaluate(xt,yt, verbose=0)
test_results['small1'] = a
print('fit=', a)
print('max error',np.max(error),np.min(error))


# +
original_stdout = sys.stdout # Save a reference to the original standard output
print(reg,learn,epochs,layers,a,np.max(error),np.min(error))

with open('his.txt', 'a') as fout:
    sys.stdout = fout # Change the standard output to the file we created.
    print(reg,learn,epochs,layers,a,np.max(error),np.min(error))
    sys.stdout = original_stdout # Reset the standard output to its ori
