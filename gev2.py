# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Simple file to show GEV functions

# +
import numpy as np
import lmoments3
import lmoments
#import lmoments_rm as lr
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

# + [markdown] tags=[]
# # Analytical relationship
# -

loc=0; scl=1.5
fshp=np.array([.5,.4,.2,0,-.2,-.4])
r=np.linspace(0,5,10)
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
#    plt.xlim(1,100)
    plt.ylabel('Return value')
    plt.xlabel('Log of Return Interval')
#   plt.ylim(1,500)
    plt.legend(loc='upper left')

loc=0; scl=1
r=np.linspace(0,5,100)
for i in range(fshp.size):
    cdf = gev.pdf(r,fshp[i], 
              loc=loc, scale=scl)
    plt.plot(r,cdf,label=str(-1*fshp[i]))
 #   plt.xscale("log")
#    plt.yscale("log")  
    plt.xlabel('Box value')
#    plt.xlim(0,30)
    plt.legend(loc='upper right')

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# # Analytical relationship

# + [markdown] tags=[]
# Testing the analytical results for shape=0.
#
# The linear relationship is for z_p = -log (y_p) where y_p = -log(CDF)
#
#

# +
aloc=3.87; scl=.198
fshp=np.array([.2,0,-.2])
#fshp=np.array([-0.00])
r=np.linspace(loc,5,100)

for i in range(fshp.size):
    cdf = gev.cdf(r,fshp[i], 
              loc=aloc, scale=scl, )
    p=1/(1-cdf+1e-30)
    yp=-np.log(cdf)
    plt.plot(-np.log(yp),r, label=str(fshp[i]*(-1)))
 #   plt.xscale("log")
 #   plt.yscale("log")
    logyp=np.linspace(0,4,4)
    zp=aloc + scl*logyp
#    plt.plot(yp1,zp)
    plt.xlim(-1,4)
    plt.ylabel('Return value')
    plt.xlabel('Log of Yp')
    plt.ylim(3.5,5)
    plt.legend(loc='upper left')   


# -

# now showing the -log(yp) = log(p)
#
# good assumption except at small p values (return period less than 10^1)

# +
r=np.linspace(0,3,100)
cdf = gev.cdf(r,0, loc=0, scale=1, )
p=1/(1-cdf)
yp=-np.log(cdf)
plt.xlabel('log Yp')
plt.ylabel('Log of Return Interval')

plt.plot(-np.log(yp),np.log(p))
plt.ylim(0,3)
plt.xlim(0,3)





#plt.plot(1/yp,p)

# + active=""
# as p (return period) goes to long return periods (p->0) then log(p) = -log(yp)
# -


