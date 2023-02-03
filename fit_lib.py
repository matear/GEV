# ## Module for DL model used to fit the GEV output
#
# modifying the file to turn it into a module.
# -

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import clone_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import sample
import xarray as xr
import itertools
from numpy.random import seed
dev = '/gpu:0'

# build and compile a generic model with multiple layers set by layers
def bc_model(norm,loss,reg,learn,layers):
    model = tf.keras.models.Sequential([norm])
    for n in layers :
        model.add(tf.keras.layers.Dense(n, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(reg)) )
        
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(loss=loss,
      optimizer=tf.keras.optimizers.Adam(learn))
    return model

# fit the DL model to the data    
def dnn(loss,reg,learn,epochs,layers,xt,yt,n_save_after,lr_scheduler):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(xt))
    print(normalizer.mean.numpy())
#    with tf.device(dev):
    dnn_model = bc_model(normalizer,loss,reg,learn,layers)
    history = dnn_model.fit(xt,yt,validation_split=0.2, verbose=0, epochs=epochs,callbacks=[lr_scheduler])
# save the model
    for i in range(n_save_after):
        htmp = dnn_model.fit(xt,yt,validation_split=0.2, verbose=0, epochs=1)
        dnn_model.save('model_' + str(i) )
        print(htmp.history['loss'])
    
    return dnn_model,history

# plots of the error in the fit    
def plot_scatter(dnn_model,xt,yt):    
    yp = dnn_model.predict(xt).flatten()
    error= yp -yt
    plt.figure(figsize=(15,5))
#    
    plt.subplot(1,2,1)
    plt.scatter(yp, yt, )
    plt.plot(yt,yt,'k')
    plt.xlabel('Predicted')
    plt.ylabel('Data')
    plt.legend()
    print('Predicted max=',np.max(yp))
    print('Data max =', np.max(yt))
#
    plt.subplot(1,2,2)
    plt.hist(error, bins=20)
    plt.xlabel('Prediction Error ')
    plt.ylabel('Count')
    return error 

    
# Plot the loss as a function of epoch
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss ')
  plt.legend()
  plt.grid(True)
    
### Multiple Models
# load models from file
def load_all_models(basename,n_start, n_end):
	all_models = list()
	for epoch in range(n_start, n_end):
		# define filename for this ensemble
		filename = basename+'_' + str(epoch)
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# load 1 model from file
def load_1_model(filename):
	model=load_model(filename)
	return model
 
# create a model from the weights of multiple models
def model_weight_ensemble(loss,learn,members, weights):
	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = np.array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
#	model = clone_model(members[0])
	model = members[0]
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(loss=loss,  optimizer=tf.keras.optimizers.Adam(learn) )
	return model

# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
	# select a subset of members
	subset = members[:n_members]
	# prepare an array of equal weights
	weights = [1.0/n_members for i in range(1, n_members+1)]
	# create a new model with the weighted average of all model weights
	model = model_weight_ensemble(subset, weights)
	# make predictions and evaluate accuracy
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	return test_acc


    
#########################################
## Manipulate the GEV data
# recreate the 3 d array
def unroll(yroll,rtmp):
    nsize=rtmp.shape
    atmp=rtmp.copy()
    l=-1
#    atmp=np.zeros([nsize[0],nsize[1],nsize[2]])
    for i,j,k,m in itertools.product(range(1),range(nsize[1]),range(nsize[2]),range(nsize[3])):
        l=l+1 #print(i,j,k)
        atmp[i,j,k,m]=yroll[l]
    return atmp

# create the Deep Learning data
def xroll(rtmp):
    nsize=rtmp.shape
    xdata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3],4])
    ydata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3]])
    x1=rtmp.coords['scale'].values
    x2=rtmp.coords['shape'].values *(-1) # convert to standard GEV convention 
    x3=rtmp.coords['size'].values
    x4=rtmp.coords['ari'].values
    l=-1
    print(x1,x2,x3,x4)
    for i,j,k,m in itertools.product(range(nsize[0]),range(nsize[1]),range(nsize[2]),range(nsize[3])): #range(nsize[3])):
        l=l+1 #print(i,j,k)
        xdata[l,0]=x1[i] #np.log(x1[i])
        xdata[l,1]=x2[j] # np.log(x2[j])
        xdata[l,2]=x3[k]
        xdata[l,3]=x4[m]
        ydata[l]=rtmp[i,j,k,m].values
# predict number of samples for given relative error
    tmpx=np.copy(ydata)
    tmpy=np.copy(xdata[:,2])
    xdata[:,2]=np.copy(tmpx)
    ydata=tmpy[:]/xdata[:,3]  # number of samples / ARI
#    ydata=np.log(ydata)
#    xdata[:,3]=np.log(xdata[:,3])
    return xdata,ydata

# create the Deep Learning data
# predict relative error from GEV and number of samples
def xroll1(rtmp):
    nsize=rtmp.shape
    xdata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3],4])
    ydata=np.zeros([nsize[0]*nsize[1]*nsize[2]*nsize[3]])
    x1=rtmp.coords['scale'].values
    x2=rtmp.coords['shape'].values *(-1) # convert to standard GEV convention 
    x3=rtmp.coords['size'].values
    x4=rtmp.coords['ari'].values
    l=-1
    print(x1,x2,x3,x4)
    for i,j,k,m in itertools.product(range(nsize[0]),range(nsize[1]),range(nsize[2]),range(nsize[3])): #range(nsize[3])):
        l=l+1 #print(i,j,k)
        xdata[l,0]=x1[i] #np.log(x1[i])
        xdata[l,1]=x2[j] # np.log(x2[j])
        xdata[l,2]=x3[k]
        xdata[l,3]=x4[m]
        ydata[l]=rtmp[i,j,k,m].values
    return xdata,ydata

def rroll(ascl,ashp,erel,eari):
    n1=ascl.size
    n2=ashp.size
    n3=erel.size
    n4=eari.size
    xdata=np.zeros([n1*n2*n3*n4,4])
    x1=ascl
    x2=ashp
    x3=erel
    x4=eari
    l=-1
    for i,j,k,m in itertools.product(range(n1),range(n2),range(n3),range(n4)):
        l=l+1 
#        print(i,j,k,m)
        xdata[l,0]=ascl[i] 
        xdata[l,1]=ashp[j]
        xdata[l,2]=erel[k]
        xdata[l,3]=eari[m]
    return xdata

def runroll(ascl,ashp,erel,eari,yp):
    n1=ascl.size
    n2=ashp.size
    n3=erel.size
    n4=eari.size
    xdata=np.zeros([n1,n2,n3,n4])
    l=-1
    for i,j,k,m in itertools.product(range(n1),range(n2),range(n3),range(n4)):
        l=l+1 
#        print(i,j,k,m)
        xdata[i,j,k,m]=yp[l] 
    return xdata
