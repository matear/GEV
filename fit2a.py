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
# -

npp=500
plt.figure(figsize=(8,10))
plt.subplot(2,2,1)
plt.hist(xtmp[:,0],10)  #,yt[0:npp],'x')
plt.subplot(2,2,2)
plt.hist(xtmp[:,1],11)  #,yt[0:npp],'x')
plt.subplot(2,2,3)
plt.hist(xtmp[:,2],21)  #,yt[0:npp],'x')
plt.subplot(2,2,4)
plt.hist(xtmp[:,3],11)  #,yt[0:npp],'x')
#plt.hist(yt)  #,yt[0:npp],'x')

# check the resampling
j=2
plt.plot(xt[:,j],yt[:],'rx')
plt.plot(xtmp[:,j],ytmp[:],'bo')

# +
nl=3
(rtmp[2,:,:,nl]).plot(levels=np.arange(10,20, 1))
rtmp[2,:,:,nl].plot.contour(levels=[10],colors='white')
rtmp[6,:,:,nl].plot.contour(levels=[10],colors='grey')
rtmp[8,:,:,nl].plot.contour(levels=[10],colors='blue')

print(asol.coords['shape'].values)
print(ashp)

# using the xarray the shape coordindate has the wrong sign!

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

# + [markdown] tags=[]
# # Useful plots of fitted model 
# -

yp = s1.predict(xtmp).flatten()

atmp=fit_lib.unroll(yp,rtmp)
btmp=fit_lib.unroll(ytmp,rtmp)

ns=0; na=0
plt.figure(figsize=(12,6))
plt.subplot(1,3,1)
(atmp[ns,:,:,na]-btmp[ns,:,:,na]).plot(levels=10)
plt.subplot(1,3,2)
btmp[ns,:,:,na].plot(levels=10)
plt.subplot(1,3,3)
atmp[ns,:,:,na].plot(levels=10)

i=np.argwhere( xtmp[:,2] < 15.0 )[:,0]
print(i.shape)
xsel=xtmp[i,:]
xsel.shape
ysel=ytmp[i]
yselp = s1.predict(xsel).flatten()
#plt.plot(yp,yt,'x')
plt.plot(yselp,ysel,'o')

# ##  Calculate the number of samples

# +
#n, ashp,ascl,ari, rtmp
# make new dataset
erel=np.array([10.])  # desired uncertainty o
eari=np.array([10,50,100,200]) # desired return period
xp = fit_lib.rroll(ascl,ashp,erel,eari)

print(ashp)

# -

ya = s1.predict(xp).flatten()

# +
etmp=fit_lib.runroll(ascl,ashp,erel,eari,ya)
print(etmp.shape)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,2],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.colorbar(label='Samples / ARI')
plt.contour(cs,colors='k')
plt.title('ARI=100')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(1,2,2)
cs=plt.contourf(ashp*(1),ascl,etmp[:,:,0,3],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.title('ARI=200')
plt.colorbar(label='Samples / ARI')
plt.contour(cs, colors='k')
plt.xlabel('Shape')
plt.ylabel('Scale')

m=5
print(ashp[m])
for i in range(10):
    print(ascl[i],etmp[i,m])
# -

yres=ya.reshape([10,11,4])
xx=xp[:,0] *1.
xres=xx.reshape([10,11,4])

#plt.pcolor(ashp,ascl,yres)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.contourf(ashp,ascl,yres[:,:,2])
plt.colorbar()
plt.subplot(1,2,2)
yy=yres[:,:,3]/yres[:,:,2]
plt.contourf(yy)
plt.colorbar()

# +
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
l=-1
for i in eari:
    print(i)
    l=l+1
    plt.plot(ashp,yres[1,:,l]*i,label=str(i)+' ARI')
plt.title('Sampling for 10% return value uncertainty')
plt.xlabel('Shape')
plt.ylabel('Number of Samples')
plt.legend()

plt.subplot(1,2,2)
plt.plot(ashp,(yres[1,:,:]))
plt.xlabel('Shape')
plt.ylabel('Number of Samples/ARI')
# -


