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
#import lmoments3
#import lmoments
#import lmoments_rm as lr
import scipy
import matplotlib.pyplot as plt
import itertools
import xarray as xr
#from numba import jit

# + [markdown] toc-hr-collapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[]
# # Cluster setup

# + tags=[]
#from dask_jobqueue import PBSCluster

#cluster = PBSCluster(cores=36,
#                    processes=18, memory="6GB",
#                     project='UCLB0022',
#                     queue='premium',
#                     resource_spec='select=1:ncpus=36:mem=109G',
#                     walltime='02:00:00')
#cluster.scale(18)

#from dask.distributed import Client
#client = Client()
#client
#client = Client(cluster)

# +
#print(client)

# + [markdown] tags=[]
# # Scipy version
# -

# import GEV class for demo
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
import numpy as np

# + tags=[]
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




# +
#@jit
def gevboot(shp,loc,scl,nsize,niter,ari):
    # create storage for GEV information
    pff_p = np.zeros([niter,ari.size])
    params=np.zeros([niter,3])
    for i in range(niter):
# generate random GEV data
        data = gev.rvs(shp, loc=loc, scale=scl,
               size=nsize)
# fit distribution to data
        fshp,floc,fscl = gev.fit(data, shp, loc=loc, scale=scl)
#                     floc=loc, fscale=scl)
# evaluate PDF on given support
        pdf = gev.pdf(np.linspace(0,200,100),fshp, 
              loc=floc, scale=fscl)
        cdf = gev.cdf(np.linspace(0,200,100),fshp, 
              loc=floc, scale=fscl)
        for j in range(5):
            pff_p[i,j]=gev.ppf(ari[j], fshp, loc=floc, scale=fscl)
        params[i,:]=fshp,floc,fscl
#        plt.plot(cdf)
#
#    print(pff_p.mean(axis=0))
#    print(pff_p.std(axis=0)*2)
#    plt.plot(cdf)
    return params,pff_p

#@jit(nopython=True)
def gevcase(shp,loc,scl,niter,ari,n):
# create storage for GEV information
    nari=ari.size
    pff_t = np.zeros([nari])
# compute true value
    for j in range(nari):
        pff_t[j] = gev.ppf(ari[j],shp,loc=loc,scale=scl)
#    print(pff_t)
    sol=np.zeros([n.size,niter,nari])
    ipar=np.zeros([n.size,niter,3])
    for l in range(n.size):
        params,pff_p = gevboot(shp,loc,scl,n[l],niter,ari)
#        print(n[l])
#        print(pff_p.mean(axis=0))
#        print(pff_p.std(axis=0))
        sol[l,:,:]=pff_p[:,:]
        ipar[l,:,:]=params[:,:]
# return true value and array of ARI
    return pff_t,sol,ipar

#@jit
# function to do the iterations not used!
def gev_it():
    for m,l in itertools.product(range(ascl.size),range(ashp.size)):
      shp=ashp[l]; scl=ascl[m]
      pff_t,sol,pp=gevcase(shp,loc,scl,niter,ari,n)
      print(pff_t)
      print(1/(1-ari))
      for i in range(n.size):
#        print(n[i],2*sol[i,:,:].std(axis=0)/pff_t[i])
         asol[m,l,i,:] = 2*sol[i,:,:].std(axis=0)/pff_t[i]
      plt.plot(n,asol[m,l,:,3])
      plt.xscale("log")
      plt.yscale("log")
    return asol
    


# +

# shape, location, scale
shp, loc, scl = -0.2, params[0], 2
print(shp,loc,scl)

niter=1000
n=np.array([30,50,100,500,1000, 2000, 5000, 10000])  # sampe size
ari=np.array([.90,.95,.98,.99,.995])  # 1 in 10,20,50,100,200
ashp = np.array([-.5,-.4,-.2,0,.2,.4])
ascl = np.array([1,2,3,4])
#
ashp=np.linspace(-.5,.5,11)
ascl=np.linspace(.5,5,10)
asol= np.zeros([ascl.size,ashp.size,n.size,ari.size])
# additional storage for parameters and return levels
tparams= np.zeros([ascl.size,ashp.size,3])
iparams= np.zeros([ascl.size,ashp.size,n.size,niter,3])
iret= np.zeros([ascl.size,ashp.size,n.size,niter,ari.size])
tret= np.zeros([ascl.size,ashp.size,ari.size])
#asol=gev_it()

for m,l in itertools.product(range(ascl.size),range(ashp.size)):
    shp=ashp[l]; scl=ascl[m]
    tparams[m,l,:]=[loc,scl,shp]
    pff_t,sol,pp=gevcase(shp,loc,scl,niter,ari,n)
# store output
    iparams[m,l,:,:,:]=pp[:,:,:]
    tret[m,l,:]=pff_t
    print(shp,scl,loc)
    print(pff_t)
    print(1/(1-ari))
    for i in range(n.size):
#        print(n[i],2*sol[i,:,:].std(axis=0)/pff_t[i])
        asol[m,l,i,:] = 2*sol[i,:,:].std(axis=0)/pff_t[:]
        iret[m,l,i,:,:]=sol[i,:,:]
    plt.plot(n,asol[m,l,:,3])
    plt.xscale("log")
    plt.yscale("log")



# -

# The shape parameter sets determine the family of curves and the scale parameter just capture the same relationship.  As shape parameter becomes more positive the tail of the distribution shrinks and the ARI collapses, which helps make the determination easier with less samples.

# +
def cplot1(l):
    plt.contourf(n,ashp,asol[0,:,:,l]*100,cmap='RdBu_r',levels=[.1,1,5,10,25,50,100,200],
             extend='max',)
    plt.xlabel('Sample Number')
    plt.ylabel('Shape Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(1/(1-ari[l]))+' year' )
    plt.colorbar(label='% Error',shrink=0.8)
    plt.contour(n,ashp,asol[0,:,:,l]*100,levels=([10]),colors='white',linewidths=3)
    return

def cplot2(l):
    plt.contourf(n,ascl,asol[:,0,:,l]*100,cmap='RdBu_r',levels=[.1,1,5,10,25,50,100,200],
             extend='max')
    plt.xlabel('Sample Number')
    plt.ylabel('Shape Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(1/(1-ari[l]))+' year' )
    plt.colorbar()
    plt.contour(n,ascl,asol[:,0,:,l]*100,levels=([10]),colors='white',linewidths=3)
    return

def cplot3(l,m):
    cc=asol[l,m,:,:]*100
    plt.contourf(1/(1-ari),n,cc,cmap='RdBu_r',levels=[.1,1,5,10,25,50,100,200],
             extend='max')
    plt.xlabel('Return Period')
    plt.ylabel('Number of Samples')
    plt.title('Loc='+str(loc)+' Scale='+str(ascl[l])+' Shape='+str(ashp[m]))
   # plt.xscale("log")
    plt.colorbar()
    plt.contour(1/(1-ari),n,cc,levels=[10],colors='white',linewidths=3)


# -

cplot3(2,4)

aep=1/(1-ari)
print(asol.shape,ashp.shape,ascl.shape,n.shape,ari.shape)

# +
# save output as a xarray dataset
npars=np.array([0,1,2])
ds = xr.Dataset(
     data_vars=dict( 
         asol=(["scale","shape","size","ari"], asol*100),
         t_params=(["scale","shape","pars"], tparams ),
         i_params=(["scale","shape","size","iter","pars"], iparams ),
         t_ret=(["scale","shape","ari"], tret),
         i_ret=(["scale","shape","size","iter","ari"], iret ),
     ),
     coords=dict(
         shape=ashp, scale=ascl, size=n, ari=aep,
         pars=npars, irer=np.arange(niter),
     ),
     attrs=dict(description="GEV Analysis and Output"),
     )

ds.to_netcdf("loc"+str(loc)+".nc")

plt.hist(ds.t_ret[4,:,4])
print(loc)

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
# In the two figures above the white line denotes 10% error in ARI. For the 10% error, one needs 100s of samples and it approaches 1000s for a ARI of 1 in 100 year event.  


for i in range(5):
    plt.hist(iret[5,6,i,:,4])
#    print(sol[i,:,:].std(axis=0))
#    print(sol[i,:,:].mean(axis=0))


sol.shape
for j in range(5):
    mean=iret[5,6,:,:,j].mean(axis=1)
    sd= iret[5,6,:,:,j].std(axis=1)
    plt.plot(n,mean,'k')
    plt.fill_between(n,mean-sd,mean+sd,alpha=0.5)


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# ## look at the shape of a GEV as a function of the 3 parameters

# + tags=[]
loc=10; scl=1
fshp=np.array([-.4,-.2,0,.2,.4])
for i in range(fshp.size):
    cdf = gev.cdf(np.linspace(loc,100,100),fshp[i], 
              loc=loc, scale=scl)
    plt.plot(np.linspace(loc,100,100),1/(1-cdf+1e-10))
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(1,1000)

loc=10; scl=3
for i in range(fshp.size):
    cdf = gev.cdf(np.linspace(loc,100,100),fshp[i], 
              loc=loc, scale=scl)
    plt.plot(np.linspace(loc,100,100),1/(1-cdf+1e-10))
    plt.xscale("log")
    plt.yscale("log")  
    plt.xlabel('Box value')
    plt.ylabel('Return Interval')
    plt.ylim(1,1000)


# +
loc=10; scl=1
for i in range(fshp.size):
    cdf = gev.pdf(np.linspace(0,100,100),fshp[i], 
              loc=loc, scale=scl)
    plt.plot(np.linspace(0,100,100),cdf)
 #   plt.xscale("log")
 #   plt.yscale("log")

loc=10; scl=3
for i in range(fshp.size):
    cdf = gev.pdf(np.linspace(0,100,100),fshp[i], 
              loc=loc, scale=scl)
    plt.plot(np.linspace(0,100,100),cdf)
 #   plt.xscale("log")
#    plt.yscale("log")  
    plt.xlabel('Box value')
    plt.xlim(0,40)
    
# -


