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

# + [markdown] tags=[]
# ## Iterate through a suite of DL models for representing the GEV data.
# Output stored in his.txt

# +
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
from numpy.random import seed
import sys
import os

sys.stdout.flush()
#tf.__version__
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.list_physical_devices()

# + tags=[]

for i,l,j in itertools.product(range(9),range(9),range(2)):
    reg=j*0.001
    layer1=4*(i+1)
    layer2=4*(l+1)
    astr=str(reg)+' 0.004 400 ' + str(layer1) + ' ' + str(layer2) + ' 64'
    command=' python fit2.py  ' + astr
    print(command)
    exitCode = os.system(str(command))
#    %run fit2.py << print(astr) 
#0.002 0.004 400 64 64 64
# -


