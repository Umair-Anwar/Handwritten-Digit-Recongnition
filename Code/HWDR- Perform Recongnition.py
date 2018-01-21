import cv2
import numpy as np
from PIL import Image
import theano.tensor as T
import theano as t
import lasagne as l
import lasagne.layers as L
from skimage.io import imread, imsave
import pickle
import gzip

def build_mlp(input_var=None):
    l_in=L.InputLayer((None,784))
    l_shape=L.ReshapeLayer(l_in,(-1,1,28,28))
    l_conv=L.Conv2DLayer(l_shape,num_filters=6,filter_size=(2,2),stride=(1,1),pad=0,nonlinearity=l.nonlinearities.rectify)
    pool=L.Pool2DLayer(l_conv,2)
    l_output = L.DenseLayer(pool, num_units=10,nonlinearity=l.nonlinearities.softmax)
    return l_output

x_sym=T.matrix()
l_output=build_mlp(x_sym)
f=np.load('c:/users/microsoft/desktop/trained_parameters2.npz') #this is the file in which trained parameters are saved
param_values = [f['arr_%d'%i] for i in range(len(f.files))]
L.set_all_param_values(l_output,param_values)

w_sym=T.matrix()
l_out=build_mlp(w_sym)
out=L.get_output(l_out,w_sym)
pred=out.argmax(-1)
f_predict=t.function([w_sym],pred)

test=imread('c:/users/Microsoft/desktop/tst.jpg')
#train,val,test=pickle.load(gzip.open('c:/users/Microsoft/desktop/mnist.pkl.gz'))
f=f_predict(test) #this will return predicted label of ur image
print(f)