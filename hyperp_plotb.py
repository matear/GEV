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

# +
import sys

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

from sys import argv

if is_interactive():
    params = [0]
else:
    print(argv)
    ss=argv[1:]
    print(ss)
    params = [float(i) for i in ss]
   
    
print(params)
print(is_interactive())


# + [markdown] tags=[]
# # Load the data on the fit

# +
#
data1 = pd.read_csv('bhis1_mse.txt',  delim_whitespace=True)
data2 = pd.read_csv('tt2.txt',  delim_whitespace=True)
print(data1['a'][0])

n1 = data2['b'].array
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

for m in range(0,20):
    print(alayers[i1[m]],j1[m],n2[i1[m]],data1['h'][m],data1['i'][m]) 
# -


plt.plot(j1[:],n2[i1[:]],'o')

# +
#
data1 = pd.read_csv('bhis0_mse.txt',  delim_whitespace=True)
data2 = pd.read_csv('tt2.txt',  delim_whitespace=True)
print(data1['a'][0])

n1 = data2['b'].array
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

for m in range(0,20):
    print(alayers[i1[m]],j1[m],n2[i1[m]],data1['h'][m],data1['i'][m]) 
# -


plt.plot(j1[:],n2[i1[:]],'o')


