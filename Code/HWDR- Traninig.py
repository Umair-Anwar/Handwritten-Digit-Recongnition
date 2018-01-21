import cv2
import numpy as np
from PIL import Image
import theano.tensor as T
import theano as t
from theano.tensor.nnet import conv
import matplotlib as plt
import lasagne as l
import lasagne.layers as L
import gzip
import pickle
import array
from skimage.io import imread, imsave

def load_dataset():  #function to load data
  train,val,test=pickle.load(gzip.open('c:/users/Microsoft/desktop/mnist.pkl.gz'))
  #x_train=imread('c:/users/Microsoft/desktop/train.jpg')     #load training data
  #y_train=np.loadtxt('c:/users/Microsoft/desktop/train.txt') #load training labels
  #return x_train,y_train
  x_train,y_train=train
  return x_train,y_train

x_train,y_train=load_dataset()      #function is being called here
def batch_gen(X,y,N):
   while True:
      idx=np.random.choice(len(y),N)
      yield X[idx].astype('float32'),y[idx].astype('int32')

def build_mlp(input_var=None):     #this is the simplest network according to tutorial,u can add layers after observing accuracy of ur data
    l_in=L.InputLayer((None,784)) #multiply dimensions of ur image n replace 22500 with ur result
    l_shape=L.ReshapeLayer(l_in,(-1,1,28,28))
    l_conv=L.Conv2DLayer(l_shape,num_filters=6,filter_size=(2,2),stride=(1,1),pad=0,nonlinearity=l.nonlinearities.rectify)
    pool=L.Pool2DLayer(l_conv,2)
    l_output = L.DenseLayer(pool, num_units=10,nonlinearity=l.nonlinearities.softmax)
    return l_output

#symbolic variables for our input features and targets
x_sym=T.matrix()
y_sym=T.ivector()

l_output=build_mlp(x_sym)
output=L.get_output(l_output,x_sym)
pred=output.argmax(-1)
loss=T.mean(l.objectives.categorical_crossentropy(output,y_sym))
acc=T.mean(T.eq(pred,y_sym))
params=L.get_all_params(l_output)
grad=T.grad(loss,params)
updates=l.updates.sgd(grad,params,learning_rate=0.05)  #u can change learning rate, 0.05 was being used in tutorial
f_train=t.function([x_sym,y_sym],[loss,acc],updates=updates)
f_predict=t.function([x_sym],pred)
batch_size=100 #u can change batch size too according to ur number of images
max_epoch=5   #u can change this number too,it is number of cycles
n_batches=len(x_train)//batch_size  #try to make this value integer otherwise the remaining images won't be trained
train_batches=batch_gen(x_train,y_train,batch_size)
for epoch in range(max_epoch):
   train_loss=0
   train_acc=0
   for _ in range(n_batches):
      x,y=next(train_batches)
      loss,acc=f_train(x,y)
      train_loss+=loss
      train_acc+=acc
   train_loss/=n_batches
   train_acc/=n_batches

   print(epoch,train_loss,train_acc)
   np.savez('c:/users/Microsoft/desktop/trained_parameters'+str(epoch)+'.npz', *L.get_all_param_values(l_output))


#np.savez('c:/users/Microsoft/desktop/trained_parameters.npz', *L.get_all_param_values(l_output)) #this will save trained parameters in .npz file to use for testing
#f = open('c:/users/microsoft/desktop/trained_parameters1.pkl', 'wb')
#pickle.dump(L.get_all_param_values(l_output), f, protocol=pickle.HIGHEST_PROTOCOL)#this will save trained parameters in .pkl file.u can use either one of the files
#f.close()


