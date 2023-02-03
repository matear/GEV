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

# + [markdown] Collapsed="false" tags=[] jp-MarkdownHeadingCollapsed=true
# # Analysis of the hyperparameter simulations

# + Collapsed="false" tags=[]
import numpy as np
#import lmoments_rm as lr
import scipy
import matplotlib.pyplot as plt
import itertools
import xarray as xr
import pandas as pd

# + [markdown] tags=[]
# # Load the data on the fit

# +
#
data1 = pd.read_csv('caseb4/obhis1.txt',  delim_whitespace=True)
data1 = pd.read_csv('casea1/oahis1.txt',  delim_whitespace=True)

n1 = data1['j'].array
n2=n1[0:150]

# sort data according to cost value
err=data1['g'].array

#i=np.argsort(n2,kind='stable')
#j=np.sort(n2,kind='stable')

i1=np.argsort(err,kind='stable')
j1=np.sort(err,kind='stable')

# list best 20 fits
i1s=i1[0:20]

layers=data1[['d','e','f']]
alayers=layers.to_numpy()

for m in range(0,1):
    print(alayers[i1[m]],j1[m],n2[i1[m]],data1['h'][m],data1['i'][m]) 

# -

#plt.plot(j1[0:20],n2[i1[0:20]],'o')
plt.plot(n2[i1[0:20]],j1[0:20],'o')
print(i1)

# +
# for shape parameter
plt.figure(1, figsize=(10, 10))

plt.subplot(221); cplot1(1,1)
plt.subplot(222); cplot1(1,2)
plt.subplot(223); cplot1(1,3)
plt.subplot(224); cplot1(1,4)

# +
# Example figure
plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
# global
tl1='Natural'
tl2="Total"
t1='Global'
flxplt(totd*(-1e-15),tota*(-1e-15),gstd_d.std(axis=1)*1e-15,gstd_a.std(axis=1)*1e-15)
plt.subplot(2,2,2)
t1='Southern Ocean'
flxplt(totsd*(-1e-15),totsa*(-1e-15),sstd_d.std(axis=1)*1e-15,sstd_a.std(axis=1)*1e-15)

plt.subplot(2,2,4)
plt.xlim(1982,2020)
plt.ylim(-2.5,0)
t1='Southern Ocean'
flxplt(totsd*(-1e-15)*0,totsa*(-1e-15),sstd_d.std(axis=1)*1e-15*0,sstd_a.std(axis=1)*1e-15)

plt.plot(time2,sflx.sum(axis=(1,2)).rolling(mtime=365, center=True).mean(),color='blue')

plt.savefig('fco2.pdf',dpi=600)

# +
# for scale parameter
plt.figure(1, figsize=(10, 10))

plt.subplot(221); cplot2(5,1)
plt.subplot(222); cplot2(5,2)
plt.subplot(223); cplot2(5,3)
plt.subplot(224); cplot2(5,4)
# -
# In the two figures above the white line denotes 10% error in ARI. For the 10% error, one needs 100s of samples and it approaches 1000s for a ARI of 1 in 100 year event.  

plt.hist(j1,10)

plt.plot(j1[0:20])


