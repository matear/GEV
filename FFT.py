#!/usr/bin/env python
# coding: utf-8
# # Play with FFT to understand the filtering better

# +
#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf
import scipy.stats as stats
import scipy.fftpack as fft
import scipy.signal as signal
import sys
import xarray as xr
import netCDF4 as nc

#from eofs.xarray import Eof


# +
# Useful procedures

def FFT(Adat):
    Fdat=fft.fft2(Adat)
    fnum=fft.fftshift(Fdat)
    fmag=abs(fnum)
    return fmag,Fdat
    
def plot_tf(Adat,Fdat):
    plt.figure(figsize=[15,12])
    plt.subplot(2,2,1)
    plt.contourf(Adat)
    plt.title('Random time doman')
    plt.colorbar(label='Magnitude')

    fnum=fft.fftshift(Fdat)
    fmag=abs(fnum)
    plt.subplot(2,2,2)
    plt.contourf(np.log(fmag))
    plt.title('->. Frequency Domain')
    plt.colorbar(label='Abs')

    return

def filter1(Fdat):
    Ffil= np.copy(Fdat)*0+1e-32
    Ffil[:,0]=Fdat[:,0]*1
    Ffil[0,:]=Fdat[0,:]*1
    Afil=fft.ifft2(Ffil)
    return Ffil,Afil

def filter2(Fdat):
    n1=100 ;n2=512-n1
    Ffil= np.copy(Fdat)*0+1e-32
    Ffil[n1:n2,0]=Fdat[n1:n2,0]*1
    Ffil[0,n1:n2]=Fdat[0,n1:n2]*1
    Afil=fft.ifft2(Ffil)
    return Ffil,Afil

def filter3(Fdat):
    n1=100 ;n2=512-n1
    Ffil= np.copy(Fdat)
    Ffil[n1:n2,0]= Ffil[n1:n2,0]*0+1e-99
    Ffil[0,n1:n2]=Ffil[0,n1:n2]*0+1e-99
    Afil=fft.ifft2(Ffil)
    return Ffil,Afil
    
    
    
    
  
def plot_tf2(Adat,fmag):
    plt.figure(figsize=[15,12])
    plt.subplot(2,2,1)
    plt.contourf(Adat)
    plt.title('Random time doman')
    plt.colorbar(label='Magnitude')

    plt.subplot(2,2,2)
    plt.contourf(fmag)
    plt.title('->. Frequency Domain')
    plt.colorbar(label='Abs')

    plt.subplot(2,2,3)
    ffil=fft.fftshift(Ffil)
    plt.contourf((abs(ffil)))
    plt.title('Filtered Frequency doman')
    plt.colorbar(label='Abs')

    plt.subplot(2,2,4)
    plt.contourf(Afil)
    plt.title('-> time doman')
    plt.colorbar(label='Magnitude')
    return
    


# -

# #  Simple example looking at random numbers 

# +
#create 2d array to filter

Adat=np.random.rand(512,512)-.5
mm=np.mean(Adat)
Fdat=fft.fft2(Adat)
# -

print(mm)

# ## get frequency
# Nx=512
# xfw=fft.fftfreq(512,1)*Nx
# xfx=fft.fftfreq(Nx,1) *Nx  # puts it into wave number
# #
# xfw=fft.fftshift(xfw)
# xfx=fft.fftshift(xfx)
# fnum=fft.fftshift(Fdat)
# fmag=abs(fnum)

# +
Ffil= np.copy(Fdat)*0
Ffil[:,0]=Fdat[:,0]*1
Ffil[0,:]=Fdat[0,:]*1

Afil=fft.ifft2(Ffil)

# +
plt.figure(figsize=[15,12])
plt.subplot(2,2,1)
plt.contourf(Adat)
plt.title('Random time doman')
plt.colorbar(label='Magnitude')

plt.subplot(2,2,2)
plt.contourf(fmag)
plt.title('->. Frequency Domain')
plt.colorbar(label='Abs')

plt.subplot(2,2,3)
ffil=fft.fftshift(Ffil)
plt.contourf((abs(ffil)))
plt.title('Filtered Frequency doman')
plt.colorbar(label='Abs')

plt.subplot(2,2,4)
plt.contourf(Afil)
plt.title('-> time doman')
plt.colorbar(label='Magnitude')

# -

plt.plot(abs(Ffil[1,:]))
#plt.plot(abs(Ffil[0,:]))

# #  Now look at the mean rainfall fields  

file1='clim_2014_2020.zarr/MT_clim_2014_2020.zarr'  # original data
file2='clim_2014_2020.zarr/MI_clim_2014_2020.zarr'  # bilinear
file3='clim_2014_2020.zarr/exp3_oro2_clim_2014_2020.zarr'
file4='clim_2014_2020.zarr/exp3_oro2_1000_clim_2014_2020.zarr'
file5='clim_2014_2020.zarr/exp3_clim_2014_2020.zarr'
da1=xr.open_zarr(file1)
da2=xr.open_zarr(file2)
da3=xr.open_zarr(file3)
da4=xr.open_zarr(file4)
da5=xr.open_zarr(file5)

plt.figure(figsize=[17,5])
plt.subplot(1,3,1)
da1.pr.plot(levels=20)
plt.subplot(1,3,2)
da2.pr.plot(levels=20)
plt.subplot(1,3,3)
da3.pr.plot(levels=20)

da1.pr.mean().values

# +
Adat1=(np.copy(da1.pr) - da1.pr.mean().values )*3600
Adat2=(np.copy(da2.pr) - da2.pr.mean().values )*3600
Adat3=np.copy(da3.pr) - da3.pr.mean().values
Adat4=np.copy(da4.pr) - da4.pr.mean().values
Adat5=np.copy(da5.pr) - da5.pr.mean().values
print(np.mean(Adat2))

fmag1,Fdat1=FFT(Adat1)
fmag2,Fdat2=FFT(Adat2)

plot_tf(Adat1,Fdat1)
plot_tf(Adat2,Fdat2)
print(Fdat2[0,0:2],np.mean(Adat2))


# +
# filter the data
Ffil1,Afil1=filter1(Fdat1)
plot_tf(Afil1,Ffil1)

Ffil1,Afil1=filter2(Fdat1)
plot_tf(Afil1,Ffil1)

Ffil1,Afil1=filter3(Fdat1)
plot_tf(Afil1,Ffil1)
plot_tf(Adat1,Fdat1)

Ffil1, Afil1=filter3(Fdat2)
plot_tf(Afil1,Ffil1)
plot_tf(Adat2,Fdat2)

# +
h = signal.windows.hamming(512)
#h = signal.windows.hann(512)
window = np.sqrt(np.outer(h,h))
#h = signal.windows.blackman(512)
#window = np.nan_to_num( np.square(np.outer(h,h)))

plt.contourf(window)

Asmooth1 = Adat1*window
fsmooth1,Fsmooth1=FFT(Asmooth1)
Asmooth2 = Adat2*window
fsmooth2,Fsmooth2=FFT(Asmooth2)
Asmooth3 = Adat3*window
fsmooth3,Fsmooth3=FFT(Asmooth3)
Asmooth4 = Adat4*window
fsmooth4,Fsmooth4=FFT(Asmooth4)
Asmooth5 = Adat5*window
fsmooth5,Fsmooth5=FFT(Asmooth5)
ff,Fdat4=FFT(Adat4)
ff,Fdat3=FFT(Adat3)
ff,Fdat5=FFT(Adat5)

plot_tf(Asmooth1,Fsmooth1)
plt.savefig('fig1_1.png')
plot_tf(Asmooth2,Fsmooth2)
plt.savefig('fig1_2.png')
plot_tf(Asmooth3,Fsmooth3)
plt.savefig('fig1_3.png')
plot_tf(Asmooth4,Fsmooth4)
plt.savefig('fig1_4.png')
plot_tf(Asmooth5,Fsmooth5)
plt.savefig('fig1_5.png')

# +
n=np.arange(0,256)

plt.figure(figsize=[10,5])
plt.subplot(1,2,1)
#plt.plot(np.log(np.abs(Fsmooth1[n,n])))
plt.plot(np.log(np.abs(Fdat1[n,n])))
#plt.plot(np.log(np.abs(Fsmooth2[n,n])))
#plt.plot(np.log(np.abs(Fsmooth3[n,n])))
#plt.plot(np.log(np.abs(Fsmooth4[n,n])))
plt.plot(np.log(np.abs(Fdat2[n,n])))
plt.plot(np.log(np.abs(Fdat3[n,n])))
plt.plot(np.log(np.abs(Fdat4[n,n])))
plt.plot(np.log(np.abs(Fdat5[n,n])))

plt.subplot(1,2,2)
plt.plot(np.log(np.abs(Fsmooth1[n,n])))
plt.plot(np.log(np.abs(Fsmooth2[n,n])))
plt.plot(np.log(np.abs(Fsmooth3[n,n])))
plt.plot(np.log(np.abs(Fsmooth4[n,n])))
plt.plot(np.log(np.abs(Fsmooth5[n,n])))

plt.savefig('fig2.png')
# -






