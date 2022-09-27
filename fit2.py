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

# +
import sys

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

from sys import argv

if is_interactive():
    params = [0.001, 0.0042, 400, 64,64,64 ]
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


# -

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
    print(x1,x2,x3,x4)
    for i,j,k,m in itertools.product(range(nsize[0]),range(nsize[1]),range(nsize[2]),range(nsize[3])): #range(nsize[3])):
        l=l+1 #print(i,j,k)
        xdata[l,0]=x1[i] #np.log(x1[i])
        xdata[l,1]=x2[j] # np.log(x2[j])
        xdata[l,2]=x3[k]
        xdata[l,3]=x4[m]
        ydata[l]=rtmp[i,j,k,m].values
# predict number of samples for given relative error
    tmpx=np.copy(ydata)
    tmpy=np.copy(xdata[:,2])
    xdata[:,2]=np.copy(tmpx)
    ydata=tmpy[:]/xdata[:,3]  # number of samples / ARI
#    ydata=np.log(ydata)
#    xdata[:,3]=np.log(xdata[:,3])
    return xdata,ydata


# +
# turn data into form suitable for DL
seed(2)
xtmp,ytmp=xroll(rtmp)
print(xtmp.shape)
nt=ytmp.size
i=sample(range(0, nt), nt)

# resample data for training and validation
xt=np.copy(xtmp[:,0:4])
yt=np.copy(ytmp)
xt[:,0:4]=xtmp[i[:],0:4]
yt[:]=ytmp[i[:]]
npp=500
print(xt.shape,yt.shape)
plt.figure(figsize=(8,10))
plt.subplot(2,2,1)
plt.hist(xtmp[:,0],10)  #,yt[0:npp],'x')
plt.subplot(2,2,2)
plt.hist(xtmp[:,1],11)  #,yt[0:npp],'x')
plt.subplot(2,2,3)
plt.hist(xtmp[:,2],21)  #,yt[0:npp],'x')
plt.subplot(2,2,4)
plt.hist(xtmp[:,3],11)  #,yt[0:npp],'x')

print(xtmp[:,2])
#plt.hist(yt)  #,yt[0:npp],'x')
# -

# check the resampling
j=2
plt.plot(xt[:,j],yt[:],'rx')
plt.plot(xtmp[:,j],ytmp[:],'bo')

rtmp

# +
nl=3
(rtmp[2,:,:,nl]).plot(levels=np.arange(10,20, 1))
rtmp[2,:,:,nl].plot.contour(levels=[10],colors='white')
rtmp[6,:,:,nl].plot.contour(levels=[10],colors='grey')
rtmp[8,:,:,nl].plot.contour(levels=[10],colors='blue')

print(asol.coords['shape'].values)
print(ashp)

# using the xarray the shape coordindate has the wrong sign!
# -

# # some plotting routines 

# +
    
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
    return error 
    
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss ')
  plt.legend()
  plt.grid(True)


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
    dnn_model = build_and_compile_model(normalizer,loss,reg,learn,layers)
    history = dnn_model.fit(xt,yt,validation_split=0.5, verbose=0, epochs=epochs)
    plot_loss(history) 
    
    return dnn_model,history


# + tags=[]
# #%%time
loss='mean_squared_error'
loss='mean_absolute_error'
seed(1)
#reg=0.0
#learn=0.0035
#epochs=200
#layers=np.array([8,4,8])
print(reg,learn,epochs,layers)

s1,h1=dnn(loss,reg,learn,epochs,layers, xt,yt)
error=plot_scatter(s1,xt,yt)
a=s1.evaluate(xt,yt, verbose=0)
test_results['small1'] = a
print('fit=', a)
print('max error',np.max(error),np.min(error))


# + tags=[]
test_results
# -

layers

# +
original_stdout = sys.stdout # Save a reference to the original standard output
print(reg,learn,epochs,layers,a,np.max(error),np.min(error))

with open('his.txt', 'a') as fout:
    sys.stdout = fout # Change the standard output to the file we created.
    print(reg,learn,epochs,layers,a,np.max(error),np.min(error))
    sys.stdout = original_stdout # Reset the standard output to its ori


# + [markdown] tags=[]
# # Do some analysis of the results 

# +

def rroll(ascl,ashp,erel,eari):
    n1=ascl.size
    n2=ashp.size
    n3=erel.size
    n4=eari.size
    xdata=np.zeros([n1*n2*n3*n4,4])
    x1=ascl
    x2=ashp
    x3=erel
    x4=eari
    l=-1
    for i,j,k,m in itertools.product(range(n1),range(n2),range(n3),range(n4)):
        l=l+1 
#        print(i,j,k,m)
        xdata[l,0]=ascl[i] 
        xdata[l,1]=ashp[j]
        xdata[l,2]=erel[k]
        xdata[l,3]=eari[m]
    return xdata

def runroll(ascl,ashp,erel,eari,yp):
    n1=ascl.size
    n2=ashp.size
    n3=erel.size
    n4=eari.size
    xdata=np.zeros([n1,n2,n3,n4])
    l=-1
    for i,j,k,m in itertools.product(range(n1),range(n2),range(n3),range(n4)):
        l=l+1 
#        print(i,j,k,m)
        xdata[i,j,k,m]=yp[l] 
    return xdata


# -

yp = s1.predict(xt).flatten()

atmp=unroll(yp,rtmp)
btmp=unroll(yt,rtmp)

ns=6; na=3
plt.subplot(1,2,1)
(atmp[:,ns,:,na]-btmp[:,ns,:,na]).plot()
plt.subplot(1,2,2)
btmp[:,ns,:,na].plot()

i=np.argwhere(yt[:] < 50.0)[:,0]
print(i.shape)
xsel=xt[i,:]
xsel.shape
ysel=yt[i]
yselp = s1.predict(xsel).flatten()
#plt.plot(yp,yt,'x')
plt.plot(yselp,ysel,'o')

# ##  Calculate the number of samples

# +
#n, ashp,ascl,ari, rtmp
# make new dataset
erel=np.array([10.])  # desired uncertainty o
eari=np.array([100,200]) # desired return period
xp = rroll(ascl,ashp,erel,eari)

print(ashp)

# -

ya = s1.predict(xp).flatten()

# +
etmp=runroll(ascl,ashp,erel,eari,ya)
print(etmp.shape)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,0],levels=np.arange(2,40,4),cmap='coolwarm',extend='both')
plt.colorbar(label='Samples / ARI')
plt.contour(cs,colors='k')
plt.title('ARI=100')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(1,2,2)
cs=plt.contourf(ashp*(1),ascl,etmp[:,:,0,1],levels=np.arange(2,40,4),cmap='coolwarm',extend='both')
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

yres=ya.reshape([10,11,2])
xx=xp[:,0] *1.
xres=xx.reshape([10,11,2])

#plt.pcolor(ashp,ascl,yres)
plt.contourf(ashp,ascl,yres[:,:,0])
plt.colorbar()

yy=yres[:,:,1]/yres[:,:,0]
plt.contourf(yy)
plt.colorbar()

test_results

import fit_lib 


# +

help(fit_lib)



# -


