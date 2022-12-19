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
# solving the uncertainty of Return Value as a function of GEV parameters, Sample size and ARI
#
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
##    params = [0.000, 0.004, 400, 16,16,8 ]
##    params = [0.000, 0.004, 400, 8,32,4 ]
#    params = [0.000, 0.004, 400, 8,8,16 ]
    params = [0.000, 0.004, 400, 8,16,8 ]
#    params = [0.001, 0.0042, 600, 8,16,8 ]
#    params = [0.001, 0.0042, 400, 8,16,8 ]
#    params = [0.0001, 0.0040, 400, 8,16,8 ]
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
# redefine relative error based on the (95% - 5% / 50% ) values times 100 to get percent 
a50=i_ret.reduce(np.percentile,axis=3,q=50)
a5=i_ret.reduce(np.percentile,axis=3,q=5)
a95=i_ret.reduce(np.percentile,axis=3,q=95)
rtmp=100*((a95-a5)/a50) *.5  # to reflect +/-

# +

print(rtmp[0,10,0,:].values)
for rr in range(4):
    plt.hist(i_ret[0,10,4,:,rr],label='ARI ='+str(int((ari[rr]+.1))) )
    plt.plot([a5[0,10,4,rr],a50[0,10,4,rr],a95[0,10,4,rr]],[rr,rr,rr],'bx-')
    plt.title('Shape ='+str(ashp[10])+ ' Scale = ' + str(ascl[0]) )
    plt.xlabel('Return Value')
    plt.ylabel('Number')
    plt.legend()
#plt.hist(i_ret[0,10,4,:,1])
#plt.plot([a5[0,10,4,1],a50[0,10,4,1],a95[0,10,4,1]],[10,10,10],'gx-')
plt.savefig('figf3b.png')
# -

da

rtmp[1,:,1,:].plot()

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
(rtmp[2,:,:,nl]).plot(levels=np.arange(5,20, 1))
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
n_save=1

s1,h1=fit_lib.dnn(loss,reg,learn,epochs,layers, xt,yt, n_save)

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

if ( layers.size < 3):
    ltmp=np.append(layers,[0])
    layers=ltmp
#print(layers[0],layers[1],layers[2])

with open('his1.txt', 'a') as fout:
    sys.stdout = fout # Change the standard output to the file we created.
    print(reg,learn,epochs,layers,a,np.max(error),np.min(error))
    sys.stdout = original_stdout # Reset the standard output to its ori

# +
#h1.history['loss'][-10:]
# -

# #  Load Multiple models 

# load all models into memory
members = fit_lib.load_all_models(0,n_save) # 20) #n_save)
print('Loaded %d models' % len(members))
# prepare an array of equal weights
n_models = len(members)
weights = [1/n_models for i in range(1, n_models+1)]
# create a new model with the weighted average of all model weights
model = fit_lib.model_weight_ensemble(loss,learn,members, weights)
# summarize the created model
#model.summary()

# +
for i in range(n_save):
    a=members[i].evaluate(xt,yt, verbose=0)
    print(a)
    
a=model.evaluate(xt,yt, verbose=0)
#a=s1.evaluate(xt,yt, verbose=0)
test_results['small1'] = a
print('fit=', a)
print('max error',np.max(error),np.min(error))

model.save('model_' + str(i) )



# + [markdown] tags=[]
#
# # Useful plots of fitted model 

# + tags=[]
yp = s1.predict(xtmp).flatten()
yp = model.predict(xtmp).flatten()
# -

atmp=fit_lib.unroll(yp,rtmp)
btmp=fit_lib.unroll(ytmp,rtmp)

ns=0; na=0
plt.figure(figsize=(13,6))
plt.subplot(1,3,1)
(atmp[ns,:,:,na]-btmp[ns,:,:,na]).plot(levels=10)
plt.subplot(1,3,2)
btmp[ns,:,:,na].plot(levels=10)
plt.subplot(1,3,3)
atmp[ns,:,:,na].plot(levels=10)

print(xtmp.shape)
print(xtmp[0:10,:],ytmp[0:10].T)
plt.plot(xtmp[0:10,3],ytmp[0:10],'x')
              

# +
#plt.plot(xtmp[0:100:5,2]) 
ii=np.argwhere( (ytmp[:] < 100) &  (ytmp[:] > 0) )
print(ii.size)
plt.plot((yp[ii]))
plt.plot(ytmp[ii])

plt.plot(ytmp[ii]- yp[ii])
#plt.ylim([19,21])
#plt.plot(yselp[0:1000])
#print(yp[0:100])
# -

i=np.argwhere( xtmp[:,3] < 20000.0 )[:,0]
print(i.shape)
xsel=xtmp[i,:]
xsel.shape
ysel=ytmp[i]
yselp = s1.predict(xsel).flatten()
#plt.plot(yp,yt,'x')
plt.plot(yselp,ysel,'o')
plt.xlim([0, 200])
plt.ylim([0, 200])

# ##  Calculate the number of samples

# +
#n, ashp,ascl,ari, rtmp
# make new dataset
erel=np.array([500.])  # desired uncertainty o
eari=np.array([20,50,100,200]) # desired return period
xp = fit_lib.rroll(ascl,ashp,erel,eari)

print(ashp)
Num_samples=int(erel)
print(Num_samples)

# -

ya = s1.predict(xp).flatten()
plt.plot(ya)

# +
etmp=fit_lib.runroll(ascl,ashp,erel,eari,ya)
print(etmp.shape)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,0],levels=np.arange(4,44,4),cmap='Reds',extend='both')
plt.colorbar(label='Relative Uncertainty (%)')
plt.contour(cs,colors='k')
plt.title('ARI=20')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,2)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,1],levels=np.arange(4,44,4),cmap='Reds',extend='both')
plt.colorbar(label='Relative Uncertainty (%)')
plt.contour(cs,colors='k')
plt.title('ARI=50')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,3)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,2],levels=np.arange(4,44,4),cmap='Reds',extend='both')
plt.colorbar(label='Relative Uncertainty (%)')
plt.contour(cs,colors='k')
plt.title('ARI=100')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,4)
cs=plt.contourf(ashp*(1),ascl,etmp[:,:,0,3],levels=np.arange(4,44,4),cmap='Reds',extend='both')
plt.title('ARI=200')
plt.colorbar(label='Relative Uncertainty (%)')
plt.contour(cs, colors='k')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.savefig('figf1b.png')

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
plt.contourf(ashp,ascl,yres[:,:,0])
plt.colorbar()
print(yres[0:10,:,0].mean(axis=0),yres[0:10,:,1].mean(axis=0))

# +
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
l=-1
#plt.yscale("log")  
for i in eari:
    print(i)
    l=l+1
    plt.plot(ashp,yres[0:10,:,l].mean(axis=0),label=str(i)+' ARI')
plt.title('Uncertainty for '+str(Num_samples)+' samples')
plt.xlabel('Shape') 
plt.ylabel('Return Value relative Uncertainty (%)')
plt.plot(ashp,np.zeros(ashp.size)+10,'--k')
plt.legend()

plt.subplot(1,2,2)
plt.yscale("log")  
plt.plot(ashp,(yres[2,:,:]))
plt.title('Uncertainty for '+str(Num_samples)+' samples')
plt.xlabel('Shape')
plt.ylabel('Return Value relative Uncertainty (%)')
plt.plot(ashp,np.zeros(ashp.size)+10,'--k')

plt.savefig('figf2b.png')

ef=np.zeros(ashp.size)+10
# -
# dd=!date
dd1=dd[0]
dd2=dd1[19:25]
print(dd2)
model.save('modelb_' + str(dd2) )


# # End 


