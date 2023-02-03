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
# ##  Used saved DL model to produce figures

# + tags=[]
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
import fit_lib 
# -

# %load_ext autoreload
# %autoreload 2

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
plt.savefig('figf3a.png')
# -

da

rtmp[1,:,1,:].plot()

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
(rtmp[2,:,:,nl]).plot(levels=np.arange(5,20, 1))
rtmp[2,:,:,nl].plot.contour(levels=[10],colors='white')
rtmp[6,:,:,nl].plot.contour(levels=[10],colors='grey')
rtmp[8,:,:,nl].plot.contour(levels=[10],colors='blue')

print(asol.coords['shape'].values)
print(ashp)

# using the xarray the shape coordindate has the wrong sign!
# -

# #  Load Multiple models 

# +
loss='mean_absolute_error'
learn=0.004
#loss='mean_squared_error'

# load all models into memory
n_save=1 
members = fit_lib.load_all_models('amodel1',0,n_save)
print('Loaded %d models' % len(members))
# prepare an array of equal weights
n_models = len(members)
weights = [1/n_models for i in range(1, n_models+1)]
# create a new model with the weighted average of all model weights
model = fit_lib.model_weight_ensemble(loss,learn,members, weights)
# summarize the created model
model.summary()
model.optimizer.get_config()


# + tags=[]
#model=fit_lib.load_1_model('old_1/amodel1_9')
#model=fit_lib.load_1_model('amodel1_0')
model=fit_lib.load_1_model('model_0')

a=model.evaluate(xt,yt, verbose=0)
print('fit=', a)
model.summary()
model.optimizer.get_config()

# + [markdown] tags=[]
# #
# # solve for with reduced learning
# learn=0.00005
# learn=0.0001
# #learn=0.001
# #learn=0.0005
# #learn=0.004
# model.compile(loss=loss,  optimizer=tf.keras.optimizers.Adam(learn) )
# history = model.fit(xt,yt,validation_split=0.2, verbose=0, epochs=20)
# fit_lib.plot_loss(history) 
# model.optimizer.get_config()
# a=model.evaluate(xt,yt, verbose=0)
# print('fit=', a)
# -

# #
# # reduced learing again
# learn=0.0001
# learn=0.00005
# learn=0.00001*1
# #learn=0.000001
# model.compile(loss=loss,  optimizer=tf.keras.optimizers.Adam(learn) )
# history = model.fit(xt,yt,validation_split=0.2, verbose=0, epochs=10)
# fit_lib.plot_loss(history) 
# #model.optimizer.get_config()
# a=model.evaluate(xt,yt, verbose=0)
# print('fit=', a)
# #print(history.history['val_loss'])

# + [markdown] tags=[]
# #### 
# -

# # turn into markdown because it is not used
#
# for i in range(n_save):
#     a=members[i].evaluate(xt,yt, verbose=0)
#     print(a)
#     
# a=model.evaluate(xt,yt, verbose=0)
# #a=s1.evaluate(xt,yt, verbose=0)
# test_results['small1'] = a
# print('fit=', a)
#
# model.save('amodel_ave' )
#
# #print(members[1].loss)
#
# for i in range(0,n_save):
#     print(i)
#     print(members[i].loss)
#


# + [markdown] tags=[]
#
# # Useful plots of fitted model 

# + [markdown] tags=[]
# ##  Compare the DL model to the original data

# + tags=[]
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

# +
#plt.plot(xtmp[0:100:5,2]) 
ii=np.argwhere( (ytmp[:] < 1.1) &  (ytmp[:] > 0.9) )
print(ii.size)
plt.plot((yp[ii]))
plt.plot(ytmp[ii])

plt.plot(ytmp[ii]- yp[ii])
#plt.ylim([19,21])
#plt.plot(yselp[0:1000])
#print(yp[0:100])
# -

i=np.argwhere( xtmp[:,2] < 15.0 )[:,0]
print(i.shape)
xsel=xtmp[i,:]
xsel.shape
ysel=ytmp[i]
yselp = model.predict(xsel).flatten()
#plt.plot(yp,yt,'x')
plt.plot(yselp,ysel,'o')
plt.xlim([0, 100])
plt.ylim([0, 100])

# ##  Use the DL to Calculate the number of samples

# +
#n, ashp,ascl,ari, rtmp
# make new dataset
erel=np.array([10.])  # desired uncertainty o
eari=np.array([20,50,100,200]) # desired return period
xp = fit_lib.rroll(ascl,ashp,erel,eari)

print(ashp)

# -

ya = model.predict(xp).flatten()

# +
etmp=fit_lib.runroll(ascl,ashp,erel,eari,ya)
# dont need my own function because reshape works too.
yres=ya.reshape([10,11,4])
xx=xp[:,0] *1.
xres=xx.reshape([10,11,4])
print(etmp.shape,yres.shape)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,0],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.colorbar(label='Samples / ARI')
plt.contour(cs,colors='k')
plt.title('ARI=20')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,2)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,1],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.colorbar(label='Samples / ARI')
plt.contour(cs,colors='k')
plt.title('ARI=50')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,3)
cs=plt.contourf(ashp,ascl,etmp[:,:,0,2],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.colorbar(label='Samples / ARI')
plt.contour(cs,colors='k')
plt.title('ARI=100')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.subplot(2,2,4)
cs=plt.contourf(ashp*(1),ascl,etmp[:,:,0,3],levels=np.arange(2,44,4),cmap='coolwarm',extend='both')
plt.title('ARI=200')
plt.colorbar(label='Samples / ARI')
plt.contour(cs, colors='k')
plt.xlabel('Shape')
plt.ylabel('Scale')

plt.savefig('figf1a.png')

m=5
print(ashp[m])
for i in range(10):
    print(ascl[i],etmp[i,m])
# -

#plt.pcolor(ashp,ascl,yres)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.contourf(ashp,ascl,yres[:,:,2])
plt.colorbar()
plt.subplot(1,2,2)
yy=yres[:,:,3]
plt.contourf(ashp,ascl,yy)
plt.colorbar()

# +
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
l=-1
plt.yscale("log")  
for i in eari:
    print(i)
    l=l+1
#    plt.plot(ashp,yres[0:10,:,l].mean(axis=0)*i,label=str(i)+' ARI')
    plt.plot(ashp,yres[1,:,l]*i,label=str(i)+' ARI')
plt.title('Sampling for 10% return value uncertainty')
plt.xlabel('Shape')
plt.ylabel('Number of Samples')
plt.legend()

plt.subplot(1,2,2)
plt.plot(ashp,(yres[1,:,:]))
plt.xlabel('Shape')
plt.ylabel('Number of Samples/ARI')

plt.savefig('figf2a.png')
# -
# # End 
