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
# # Analysis looking at GEV 

# + Collapsed="false" tags=[]
import numpy as np
#import lmoments3
#import lmoments
#import lmoments_rm as lr
import scipy
import matplotlib.pyplot as plt
import itertools
import xarray as xr
#from numba import jit

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
    params = [0]
else:
    print(argv)
    ss=argv[1:]
    print(ss)
    params = [float(i) for i in ss]
   
    
print(params)
print(is_interactive())


# -


# The shape parameter sets determine the family of curves and the scale parameter just capture the same relationship.  As shape parameter becomes more positive the tail of the distribution shrinks and the ARI collapses, which helps make the determination easier with less samples.

# + [markdown] tags=[]
# # Analytical relationship

# + [markdown] tags=[]
# Plot some example GEV as pdf and CDFs
#
#

# +
loc=33; scl=2.5
fshp=np.array([.5,.4,.2,0,-.2,-.4])
r=np.linspace(0,50,100)

plt.figure(figsize=(8, 11))
plt.subplot(2,1,2)

for i in range(fshp.size):
    cdf = gev.cdf(r,fshp[i], 
              loc=loc, scale=scl, )
    p=1/(1-cdf)
    yp=-np.log(cdf)
    plt.plot(-np.log(yp),r, label=str(fshp[i]*(-1)))
# could plot the returm period in non log units
#    plt.plot((1/yp),r, label=str(fshp[i]*(-1)))
#    plt.xscale("log")
########
#    plt.yscale("log")
    plt.xlim(0,3)
    plt.ylabel('Return value')
    plt.xlabel('Log of Return Interval')
#   plt.ylim(1,500)
    plt.legend(loc='upper left')

    
plt.subplot(2,1,1)    
for i in range(fshp.size):
    cdf = gev.pdf(r,fshp[i], 
              loc=loc, scale=scl)
    plt.plot(r,cdf,label=str(fshp[i]*(-1)))
 #   plt.xscale("log")
#    plt.yscale("log")  
    plt.xlabel('Box value')
    plt.ylabel('Probability')
  #  plt.xlim(0,30)
    plt.legend(loc='upper right')
    

# -

print(fshp[i],loc,scl)
print(-np.log(yp[0:25]))
print(r[0:25])
plt.plot(-np.log(yp),r)
plt.plot([0,5],[1,1])
#plt.ylim(0.9,1.5)

# # Testing the impact of scale and loc
# first do the solution with a set scale and location value and plot
# redo the calculation by doubling the scale and location value and plot
#
# then scale the return value the double scale value
#

# +
loc=10; scl=1.0
fshp=np.array([.5,.4,.2,0,-.2,-.4])
r=np.linspace(0,20,100)

plt.figure(figsize=(8, 11))
plt.subplot(2,2,1)
r=np.linspace(0,20,100)

for i in range(fshp.size):
    cdf = gev.cdf(r,fshp[i], 
              loc=loc, scale=scl, )
    p=1/(1-cdf)
    yp=-np.log(cdf)
    plt.plot(-np.log(yp),r, label=str(fshp[i]*(-1)))
    plt.xlim(0,4)
    plt.ylabel('Return value')
    plt.xlabel('Log of Return Interval')
#   plt.ylim(1,500)
    plt.legend(loc='upper left')

r=np.linspace(0,40,100)
loc=20; scl=2.0
#r=r/scl
#loc=loc/scl
#scl=1
plt.subplot(2,2,2)

for i in range(fshp.size):
    cdf = gev.cdf(r,fshp[i], 
              loc=loc, scale=scl, )
    p=1/(1-cdf)
    yp=-np.log(cdf)
    plt.plot(-np.log(yp),r, label=str(fshp[i]*(-1)))
    plt.xlim(0,4)
    plt.ylabel('Return value')
    plt.xlabel('Log of Return Interval')
#   plt.ylim(1,500)
    plt.legend(loc='upper left')

# now show the equivalent plot with rescaling    
plt.subplot(2,2,1)

#plt.subplot(2,2,3)
r=(r/scl) #-loc/scl)-1
plt.plot(-np.log(yp),r, label=str(fshp[i]*(-1)))
plt.xlim(0,4)
plt.ylabel('Return value')
plt.xlabel('Log of Return Interval')
plt.legend(loc='upper left')

# get same result with rescaling!!!

# +
shp=fshp[i]*(-1)
loc=1
scl=1
tau=np.array([0,.90,.99,.999])
ari=1./(1-tau)

ret= loc - scl/shp * (1 - (-1*np.log(tau))**(-1*shp) )
#ret=  (1 - (-1*np.log(tau))**(-1*shp) )
#ret=  ( (-1*np.log(tau))**(-1*shp) )

print(shp,loc,scl)
print(ret)
# -

eari=np.array([20,50,100,200]) # desired return period
ep=1./eari
gev.isf(ep, -shp,loc,scl)


