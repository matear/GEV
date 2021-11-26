# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Simple file to show GEV functions

# +
import numpy as np
import lmoments3
import lmoments
import lmoments_rm as lr
import scipy
import matplotlib.pyplot as plt

# import GEV class for demo
from scipy.stats import genextreme as gev
# -

from dask.distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)
client


cluster = LocalCluster()  # Create a local cluster  
cluster  

cluster.scale(3)

cluster

loc=10; scl=1
fshp=np.array([1,.5,.4,.2,0])
for i in range(fshp.size):
    cdf = gev.cdf(np.linspace(loc,100,100),fshp[i], 
              loc=loc, scale=scl, )
    plt.plot(np.linspace(loc,100,100),1/(1-cdf+1e-10), label=str(fshp[i]))
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1,1000)
    plt.xlabel('Box value')
    plt.ylabel('Return Interval')
    plt.ylim(1,1000)
    plt.legend(loc='lower right')

cdf

loc=10; scl=2
for i in range(fshp.size):
    cdf = gev.pdf(np.linspace(0,100,100),fshp[i], 
              loc=loc, scale=scl)
    plt.plot(np.linspace(0,100,100),cdf,label=str(fshp[i]))
 #   plt.xscale("log")
#    plt.yscale("log")  
    plt.xlabel('Box value')
    plt.xlim(0,30)
    plt.legend(loc='upper right')


