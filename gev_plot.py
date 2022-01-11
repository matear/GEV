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
loc=0; scl=1.5
fshp=np.array([.5,.4,.2,0,-.2,-.4])
r=np.linspace(0,5,100)

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
    plt.xlim(0,5)
    plt.legend(loc='upper right')
    

plt.savefig('fig1.pdf',dpi=600)

# + [markdown] tags=[]
# # GEV analysis for sample size
# -
loc=10.0
file1='loc'+str(loc)+'.nc'
da=xr.open_dataset(file1)
file1

# + tags=[]
da

# +
asol=da.asol    # % return value error
t_ret=da.t_ret  # true solution
i_ret=da.i_ret  # solution for each iteration
aerr=i_ret.std(axis=3)/i_ret.mean(axis=3) # std/mean from iterations

n=asol.coords['size'].values
ashp=asol.coords['shape'].values*(-1)  # GEV code uses the opposite for shape parameter
ascl=asol.coords['scale'].values
ari=asol.coords['ari'].values


# + tags=[]
def cplot0(l):
# plot holding scale constant
    cmap='Pastel1'
    x= np.linspace(0.0, 1.0, 100)
    rgb = plt.cm.get_cmap(cmap)
    for m in range(4):
        col=rgb(m)
        cc=asol[l,:,:,m]
        CS=plt.contour(n,ashp,cc,levels=([10]),linewidths=3,colors=[col])
        plt.xlabel('Sample Number')
        plt.ylabel('Shape Parameter')
        plt.clabel(CS,fmt='%3.0f', colors='k', fontsize=10)
        plt.xscale("log")
        plt.title(' Scale={:.1f}'.format(ascl[l]))
    
    return

def cplot1(l,m):
    cc=asol[l,:,:,m]
    plt.contourf(n,ashp,cc,cmap='RdBu_r',colours='white',levels=[.1,1,5,10,25,50,75,100,200],
             extend='max',)
    plt.xlabel('Sample Number')
    plt.ylabel('Shape Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(ari[m])+' year' + 
              '  Scale={:.1f}'.format(ascl[l]))
    plt.colorbar(label='% Error',shrink=0.8)
    plt.contour(n,ashp,cc,levels=([10]),colors='white',linewidths=3)
    return

def cplot2(l,m):
    cc=asol[:,l,:,m]
    plt.contourf(n,ascl,cc,cmap='RdBu_r',levels=[.1,1,5,10,25,50,75,100,200],
             extend='max')
    plt.xlabel('Sample Number')
    plt.ylabel('Scale Parameter')
    plt.xscale("log")
    plt.title('ARI = 1 in  {:.1f}'.format(ari[m])+' year' +
             '  Shape={:.1f}'.format(ashp[l]))
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


# +

def cplot0(l):
# plot of return period holding scale constant
    cmap='Pastel1'
    x= np.linspace(0.0, 1.0, 100)
    rgb = plt.cm.get_cmap(cmap)
    for m in range(1,5):
        col=rgb(m)
        cc=rtmp[l,:,:,m]
        CS=plt.contour(n,ashp,cc,levels=([10]),linewidths=3,colors=[col])
        plt.xlabel('Sample Number')
        plt.ylabel('Shape Parameter')
        plt.clabel(CS,fmt='{:.0f}'.format(ari[m]), colors='k', fontsize=10)
        plt.xscale("log")
        plt.title(' Scale={:.1f}'.format(ascl[l]))
    
    return
def cplot4(l):
# plot of return period holding shape constant
    cmap='Pastel1'
    rgb = plt.cm.get_cmap(cmap)
    for m in range(1,5):
        col=rgb(m)
        cc=rtmp[:,l,:,m]
        CS=plt.contour(n,ascl,cc,levels=[10],linewidths=3,colors=[col])
        plt.xlabel('Sample Number')
        plt.ylabel('Scale Parameter')
        plt.clabel(CS,fmt='{:.0f}'.format(ari[m]), colors='k', fontsize=10)
        plt.xscale("log")
        plt.title(' Shape={:.1f}'.format(ashp[l]))
        
    return

def cplot5(l):
# plot of return period error with maximum samples
    cmap='Pastel1'
    rgb = plt.cm.get_cmap(cmap)
    for m in range(1,5):
        col=rgb(m)
        cc=rtmp[:,:,l,m]
        CS=plt.contour(ashp,ascl,cc,levels=[5],linewidths=3,colors=[col])
        plt.xlabel('Shape Parameter')
        plt.ylabel('Scale Parameter')
        plt.clabel(CS,fmt='{:.0f}'.format(ari[m]), colors='k', fontsize=10)
        plt.title(' Sample Size={:.1f}'.format(n[l]))
        
    return


# + tags=[]
# Example figure
plt.figure(figsize=(8, 11))
# they are the same
rtmp=aerr*100*2
#rtmp=asol
plt.subplot(2,2,1)
cplot0(1)

plt.subplot(2,2,3)
cplot4(5)


plt.subplot(2,2,2)
cplot0(3)

plt.subplot(2,2,4)
cplot4(3)


plt.savefig('fig2_'+str(loc)+'.pdf',dpi=600)

# +
l=1;m=10
def plot_sol(l,m):
    for j in range(5):
        mean=i_ret[l,m,:,:,j].mean(axis=1)
        sd= i_ret[l,m,:,:,j].std(axis=1)*2
        plt.plot(n,mean,'k')
        plt.fill_between(n,mean-sd,mean+sd,alpha=0.5)
        plt.title('Scale={:.1f}'.format(ascl[l])+
                    ' Shape={:.1f}'.format(ashp[m]))
    return
 
plt.figure(figsize=(8, 11))
plt.subplot(2,2,1)
plot_sol(1,5)

plt.subplot(2,2,2)
plot_sol(1,10)

plt.subplot(2,2,3)
plot_sol(3,5)

plt.subplot(2,2,4)
plot_sol(1,0)


plt.savefig('fig3_'+str(loc)+'.pdf',dpi=600)

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

# # Extra plotting

# +
def plothist(i,j,n):
    plt.plot([t_ret[i,j,n],t_ret[i,j,n]],[0,350],color='k')
    plt.title('scale='+str(ascl[i]))
    for m in range(6):
        plt.hist(i_ret[i,j,m,:,n])
#        print(i_ret[i,j,m,:,n].mean(axis=0).values,
#           i_ret[i,j,m,:,n].std(axis=0).values)
    return

plt.figure(figsize=(8, 11))

plt.subplot(2,2,1)
i=1;j=5;n=1
plothist(1,1,1)

plt.subplot(2,2,2)
plothist(1,1,2)

plt.subplot(2,2,3)
plothist(1,1,3)
    
plt.subplot(2,2,4)
plothist(1,1,4)
    
plt.savefig('fig5_'+str(loc)+'.pdf',dpi=600)

asol[:,1,:,2].plot()
asol[:,1,:,2].plot.contour(levels=[10])
rtmp[:,1,:,2].plot.contour(levels=[10],colors='white')


# + tags=[]
n=2;i=2
a1=i_ret[i,1,n,:,2]*1.
a2=a1.sortby(a1)

print((a2[950]-a2[50]).values, (a1.std()*2).values)
a2.plot()

# +
plt.figure(figsize=(8, 11))
plt.subplot(2,2,1)
(i_ret[:,7,0,:,3].std(axis=1)).plot()
(i_ret[:,7,1,:,3].std(axis=1)).plot()
(i_ret[:,7,2,:,3].std(axis=1)).plot()
(i_ret[:,7,3,:,3].std(axis=1)).plot()

plt.subplot(2,2,2)
(t_ret[:,7,1]).plot()
(t_ret[:,7,2]).plot()
(t_ret[:,7,3]).plot()
(t_ret[:,7,4]).plot()

plt.subplot(2,2,3)
(rtmp[:,7,0,3]).plot()
(rtmp[:,7,1,3]).plot()
(rtmp[:,7,2,3]).plot()
(rtmp[:,7,3,3]).plot()


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

# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] jp-MarkdownHeadingCollapsed=true
# # Old figures that are not used

# +
#cplot3(1,6)

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
