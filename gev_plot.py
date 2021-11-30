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

# + [markdown] Collapsed="false" tags=[] jp-MarkdownHeadingCollapsed=true
# # Analysis looking at GEV 

# + Collapsed="false" tags=[]
import numpy as np
import lmoments3
import lmoments
#import lmoments_rm as lr
import scipy
import matplotlib.pyplot as plt
import itertools
import xarray as xr
from numba import jit

# + [markdown] tags=[]
# # Scipy version
# -

# import GEV class for demo
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
import numpy as np

# +
import sys

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

from sys import argv

if is_interactive():
    params = [10]
else:
    print(argv)
    ss=argv[1:]
    print(ss)
    params = [float(i) for i in ss]
   
    
print(params)
print(is_interactive())


# -


# The shape parameter sets determine the family of curves and the scale parameter just capture the same relationship.  As shape parameter becomes more positive the tail of the distribution shrinks and the ARI collapses, which helps make the determination easier with less samples.

loc=10
file1='loc10.nc'
da=xr.open_dataset(file1)

da

asol=da.__xarray_dataarray_variable__
n=asol.coords['size'].values
ashp=asol.coords['shape'].values*(-1)
ascl=asol.coords['scale'].values
ari=asol.coords['ari'].values


# + tags=[]
def cplot1(l):
    cc=asol[1,:,:,l]
    plt.contourf(n,ashp,cc,cmap='RdBu_r',levels=[.1,1,5,10,25,50,75,100,200],
             extend='max',)
    plt.xlabel('Sample Number')
    plt.ylabel('Shape Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(ari[l])+' year' + 
              '  Scale={:.1f}'.format(ascl[1]))
    plt.colorbar(label='% Error',shrink=0.8)
    plt.contour(n,ashp,cc,levels=([10]),colors='white',linewidths=3)
    return

def cplot2(l):
    cc=asol[:,7,:,l]
    plt.contourf(n,ascl,cc,cmap='RdBu_r',levels=[.1,1,5,10,25,50,75,100,200],
             extend='max')
    plt.xlabel('Sample Number')
    plt.ylabel('Scale Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(ari[l])+' year' +
             '  Scale={:.1f}'.format(ashp[7]))
    plt.colorbar()
    plt.contour(n,ascl,cc,levels=([10]),colors='white',linewidths=3)
    return

def cplot3(l,m):
    cc=asol[l,m,:,:]*1
    plt.contourf(ari,n,cc,cmap='RdBu_r',levels=[.1,1,5,10,25,50,75,100,200],
             extend='max')
    plt.xlabel('Return Period')
    plt.ylabel('Number of Samples')
    plt.yscale("log")
    plt.title('Loc='+str(loc)+' Scale= {:.1f}'.format(ascl[l])+
              ' Shape={:.2f}'.format(ashp[m]))
   # plt.xscale("log")
    plt.colorbar()
    plt.contour(ari,n,cc,levels=[10],colors='white',linewidths=3)


# -

cplot3(1,6)

# +
# for shape parameter
plt.figure(1, figsize=(10, 10))

plt.subplot(221); cplot1(1)
plt.subplot(222); cplot1(2)
plt.subplot(223); cplot1(3)
plt.subplot(224); cplot1(4)

# +
# for scale parameter
plt.figure(1, figsize=(10, 10))

plt.subplot(221); cplot2(1)
plt.subplot(222); cplot2(2)
plt.subplot(223); cplot2(3)
plt.subplot(224); cplot2(4)
# -
ashp

# save output as a xarray dataset
#da = xr.DataArray(data=asol*100, dims=["scale","shape","size","ari"],
#                coords=dict(shape=ashp, scale =ascl, size=n, ari=aep),
 #                 attrs=dict(description="Return Period Uncertainty",units="percentage",),
 #                )




# In the two figures above the white line denotes 10% error in ARI. For the 10% error, one needs 100s of samples and it approaches 1000s for a ARI of 1 in 100 year event.  




