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

# + [markdown] Collapsed="false" tags=[] jp-MarkdownHeadingCollapsed=true tags=[]
# # Analysis looking at GEV 
#
# trying to fit the ARI with an equation
#
#

# + [markdown] toc-hr-collapsed=true
# # Setup Libraries
# -

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
# ## Scipy version
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
# # GEV analysis for sample size
# -
loc=10.0
file1='loc'+str(loc)+'.nc'
da=xr.open_dataset(file1)
file1

# + tags=[]
da

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

# +
ae=i_ret.std(axis=3)
aerel=ae/ae[:,:,5,:] # std relative to the 5000 sample case

plt.figure(figsize=(8, 11))
plt.subplot(2,2,1)
(np.log(aerel[1,:,0,:])).plot(cmap='OrRd')

plt.subplot(2,2,2)
(np.log(aerel[1,5,:,:])).plot(cmap='OrRd')

plt.subplot(2,2,3)
(np.log(aerel[1,:,:,3])).plot(cmap='OrRd')

plt.subplot(2,2,4)
(np.log(aerel[:,5,:,3])).plot(cmap='OrRd')

#(aerel[:,:,:,:].mean(axis=(0,1))).plot()

plt.savefig('fig4_'+str(loc)+'.pdf',dpi=600)
# -

# both the variance (first panel) and the return level value (second panel) are linear functions of scale ($\sigma$), with the return level value approaching loc value ($\mu$) as scale goes to zero.
#
# Hence the relative error (ratio of error to the return value) is given by 
# $\frac{a + b \sigma}{c + d\sigma}$
#
# now $a \approx 0$ and $c \approx \mu$
#
# which simplifies to
#
# $\frac{b \sigma}{\mu + d \sigma} $
#
# for $ d \sigma $ >> $\mu$ this becomes
# $\frac{b}{d}$, which is constant and independent of $\sigma$
#

# + [markdown] tags=[]
# # Develop function that return relative uncertainty for a given sample size
#
# for a given scale, shape and sigma value I calculate the relative error of the a specified ARI as a function of number of samples
#
# these calculations fit a single curve to a given set of parameters, which is the simplies way to tackle the problem

# +
# rtmp is the relative error as a function of 
# (scale,shape,Sample size, return period)
# for a varying scale, shape, return period the relative error for the return period

nret=1; nsh=6; nsc=0
for j in range(11):
#    plt.plot(xscale("log"))
    rtmp[nsc,j,:,1].plot.line(xscale="log",label=str(ashp[j]))

plt.legend()


# -

str(ashp[3])

# +
# curve fit
# define 3 different functions

from scipy.optimize import curve_fit
def func(x, a1, a2, a3):
    nln=np.log(x)
    y1 = a1 + a2*nln + a3*np.square(nln)
    return y1

def func1(x, a1, a2, a3):
    nln=x
    y1 = a1 + a2*nln + a3*np.square(nln)
    return y1

def func2(x, a1, a2, a3, a4):
    nln=np.log(x)
    y1 = a1 + a2*nln + a3*np.square(nln) +a4 * np.square(nln) * nln
    return y1


# +
par=np.zeros([10,11,3,5])
par1=np.zeros([10,11,4,5])


plt.figure(figsize=(8,8))

plt.subplot(2,1,1)
xdata=rtmp.coords['size'].values
for j in range(0,11,2):
    ydata=rtmp[nsc,j,:,4].values
    popt, pcov = curve_fit(func, xdata, ydata)
    par[nsc,j,:,1]=popt[:]
    plt.plot(xdata,ydata-func(xdata,*popt),label=str(ashp[j]))
plt.legend()
    
plt.subplot(2,1,2)
xdata=rtmp.coords['size'].values
for j in range(0,11,2):
    ydata=rtmp[nsc,j,:,4].values
    popt, pcov = curve_fit(func2, xdata, ydata)
    par1[nsc,j,:,1]=popt[:]
    plt.plot(xdata,ydata-func2(xdata,*popt),label=str(ashp[j]))
plt.legend()

# fit the relative error with the log of the sample size


# -


np.diag(pcov)

# +
# example fit of the last point
print(popt,np.diag(pcov))

plt.plot(xdata,ydata,'r')
plt.plot(xdata,func2(xdata,*popt))

# +
# what do the parameters look like

plt.figure(figsize=(8, 11))
plt.subplot(2,2,1)
plt.plot(ashp,par1[nsc,:,0,1])
plt.subplot(2,2,2)
plt.plot(ashp,par1[nsc,:,1,1])
plt.subplot(2,2,3)
plt.plot(ashp,par1[nsc,:,2,1])
plt.subplot(2,2,4)
plt.plot(ashp,par1[nsc,:,3,1])

# +
# what do the parameters look like

plt.figure(figsize=(11, 8))
plt.subplot(1,3,1)
plt.plot(ashp,par[nsc,:,0,1])
plt.subplot(1,3,2)
plt.plot(ashp,par[nsc,:,1,1])
plt.subplot(1,3,3)
plt.plot(ashp,par[nsc,:,2,1])
# -

# For the variable shape a good fit using $a+ b *log(N) + c (log(N))^2$
#
# The parameters of the fit look exponential so I will try to do $log(\xi)$ 

# +
# fit parameter output
xdata1=ashp+1.5
p2=np.zeros([3,4])
for j in range(3):
    ydata1=par[nsc,:,j,1]
    popt1, pcov = curve_fit(func2, xdata1, ydata1)
#    par[nsc,j,:,1]=popt[:]
    p2[j,:]=popt1
    plt.plot(xdata1,ydata1,'r')
    plt.plot(xdata1,func2(xdata1,*popt1),'b')
    
    
#for m,l in itertools.product(range(ascl.size),range(ashp.size)):
# -


# # Solving with SVD for multiple variables

# ## consider a function of scale, shape and size of ensemble and fit the results to return the relative error of a given ARI value 

# +
# recreate the 3 d array
def unroll(yroll,rtmp):
    nsize=rtmp.shape
    atmp=rtmp.copy()
    l=-1
#    atmp=np.zeros([nsize[0],nsize[1],nsize[2]])
    for i,j,k in itertools.product(range(1),range(nsize[1]),range(nsize[2])):
        l=l+1 #print(i,j,k)
        atmp[i,j,k,0]=yroll[l]
        
    return atmp

# create the data
def xroll(rtmp,nari):
    nsize=rtmp.shape
    xdata=np.zeros([nsize[0]*nsize[1]*nsize[2],14])
    ydata=np.zeros([nsize[0]*nsize[1]*nsize[2]])
    x1=rtmp.coords['scale'].values
    x2=rtmp.coords['shape'].values *(-1) +2  # min value is -.5 
    x3=rtmp.coords['size'].values
    l=-1
#    for i,j,k in itertools.product(range(1),range(1),range(nsize[2])):
    for i,j,k in itertools.product(range(1),range(nsize[1]),range(nsize[2])):
#    for i,j,k in itertools.product(range(nsize[0]),range(nsize[1]),range(nsize[2])):
        l=l+1 #print(i,j,k)
        xdata[l,0]=1
        xdata[l,1]=x1[i] #np.log(x1[i])
        xdata[l,2]=x2[j] # np.log(x2[j])
        xdata[l,3]=np.log(x3[k])
        xdata[l,4]=np.square(xdata[l,2])
        xdata[l,5]=np.square(xdata[l,2])*xdata[l,2]
        xdata[l,6]=xdata[l,2]*xdata[l,3]
        xdata[l,7]=np.square(xdata[l,2])*xdata[l,3]
        xdata[l,8]=np.square(xdata[l,2])*xdata[l,2]*xdata[l,3]
        xdata[l,9]=np.square(xdata[l,3])
        xdata[l,10]=xdata[l,2]*np.square(xdata[l,3])
        xdata[l,11]=np.square(xdata[l,2])*np.square(xdata[l,3])
        xdata[l,12]=np.square(xdata[l,2])*xdata[l,2] * np.square(xdata[l,3])
        xdata[l,13]=np.square(xdata[l,2])*np.square(xdata[l,2])*np.square(xdata[l,3])*xdata[l,3]
        ydata[l]=rtmp[i,j,k,nari].values
        
    return xdata,ydata


# -

nsc=0
XX,yy=xroll(rtmp,4)


# + tags=[]
# Solve using SVD
import numpy.linalg as la
import scipy.linalg as spla

U, sigma, VT = la.svd(XX)
narray=XX.shape
nmin=min(narray)
nmin=12

Sigma = np.zeros(XX.shape)
Sigma[:nmin,:nmin] = np.diag(sigma[:nmin])

(U.dot(Sigma).dot(VT) - XX).round(4) # check matrix calculation

Sigma_pinv = np.zeros(XX.shape).T
Sigma_pinv[:nmin,:nmin] = np.diag(1/sigma[:nmin])

x_svd = VT.T.dot(Sigma_pinv).dot(U.T).dot(yy)

yd=(U.dot(Sigma).dot(VT)).dot(x_svd)

plt.subplot(1,2,1)
plt.plot(yd)
plt.plot(yy)

plt.subplot(1,2,2)
plt.plot(yd-yy, label='Estimate Error in Fit')

print( (yd-yy).std())

# +
plt.figure(figsize=(8,11))
plt.subplot(2,2,1)
rtmp[:,5,:,1].plot()

plt.subplot(2,2,2)
aa=unroll(yd,rtmp)
aa[:,5,:,0].plot()

plt.subplot(2,2,3)
(aa[:,5,:,0]-rtmp[:,5,:,1]).plot()

# -

aa=unroll(yd,rtmp)
aa[:,5,:,0].plot()





