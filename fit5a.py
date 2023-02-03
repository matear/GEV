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
plt.savefig('figf0a.png')
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
n_save=50 
members = fit_lib.load_all_models('casea1/amodel1',0,n_save)
print('Loaded %d models' % len(members))
# prepare an array of equal weights
members[0].summary()
# -


imember=0; a1=np.zeros(len(members))
for n in members :
    a1[imember]=n.evaluate(xt,yt,verbose=0)
    print(a1[imember])
    imember=imember+1


ibest=13
ibest=23
print(a1[ibest])

# + [markdown] tags=[]
#
# # Useful plots of fitted model 

# + [markdown] tags=[]
# ##  Compare the DL model to the original data

# + tags=[]
yp = members[ibest].predict(xtmp).flatten()
# -

yp1=yp.reshape([10,11,8,5])
ytmp1=ytmp.reshape([10,11,8,5])

# +
atmp=fit_lib.unroll(yp,rtmp)
btmp=fit_lib.unroll(ytmp,rtmp)

atmp=yp1 #+rtmp*0
btmp=ytmp1# +rtmp*0
ns=1; na=4
# -

# ## ns=1; na=4
# plt.figure(figsize=(13,6))
# plt.subplot(1,3,1)
# (atmp[ns,:,:,na]-btmp[ns,:,:,na]).plot(levels=10)
# plt.subplot(1,3,2)
# btmp[ns,:,:,na].plot(levels=10)
# plt.subplot(1,3,3)
# atmp[ns,:,:,na].plot(levels=10)

# + tags=[]
plt.contourf(atmp[ns,:,:,na]-btmp[ns,:,:,na])
plt.colorbar()
# -

# Noticed it was when sigma equalled 0.5 (first index) model was having problems fitting the data
# same for both fits
for ns in range(0,3):
    plt.plot(ashp,atmp[ns,:,4,na])
    plt.plot(ashp,btmp[ns,:,4,na],'o')
plt.xlabel('Shape')
plt.ylabel('Samples/ARI')

for ns in range(0,5):
    plt.plot(atmp[ns,5,:,na])
    plt.plot(btmp[ns,5,:,na],'o')
plt.xlabel('Sample Index')
plt.ylabel('Samples/ARI')

# ##  Use the DL to Calculate the number of samples

# +
#n, ashp,ascl,ari, rtmp
# make new dataset
erel=np.array([10.])  # desired uncertainty o
eari=np.array([20,50,100,200]) # desired return period
xp = fit_lib.rroll(ascl,ashp,erel,eari)

print(ashp)

# -

ya = members[ibest].predict(xp).flatten()

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
# # Plot the uncertainty using all the members

# +
imember=0; a=np.zeros([len(ya),len(members)])
print(a.shape)


for n in members :
    a[:,imember]=n.predict(xp).flatten()
    imember=imember+1
# -

yall=a[:,:].reshape(10,11,4,50)
yres1=yall[:,:,:,:].min(axis=3)
yres2=yall[:,:,:,:].max(axis=3)
yres=yall[:,:,:,ibest]


# +
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
l=-1
plt.yscale("log")  
for i in eari:
    print(i)
    l=l+1
    plt.plot(ashp,yres[0:10,:,l].mean(axis=0)*i,label=str(i)+' ARI')
    plt.fill_between(ashp,yres1[0:10,:,l].mean(axis=0)*i, yres2[0:10,:,l].mean(axis=0)*i,alpha=.4)
#    plt.plot(ashp,yres[1,:,l]*i,label=str(i)+' ARI')
plt.title('Sampling for 10% return value uncertainty')
plt.xlabel('Shape')
plt.ylabel('Number of Samples')
plt.legend()

l=-1
plt.subplot(1,2,2)
for i in eari:
    print(i)
    l=l+1
#plt.plot(ashp,(yres[1,:,:]))
    plt.plot(ashp,yres[0:10,:,l].mean(axis=0), label=str(i)+' ARI')
    plt.fill_between(ashp,yres1[0:10,:,l].mean(axis=0), yres2[0:10,:,l].mean(axis=0),alpha=.4)
plt.title('Sampling for 10% return value uncertainty')
plt.xlabel('Shape')
plt.ylabel('Number of Samples/ARI')
plt.legend()

plt.savefig('figf3a.png')
# -
# # end 
