"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.
"""



"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.
This implementation simplifies the model in the following ways:
 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer
References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""
import os, sys, timeit
import numpy as np
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
import BatchReader2
import scipy.misc
import pickle
import CNNL
import CNN

theano.config.floatX = 'float32'

if __name__ == '__main__':
    
    rng = numpy.random.RandomState(42)
    br = BatchReader2.inputs()
    br2 = BatchReader2.inputs(testingData = True)
    X, Y = br.getNPArray(2)
    testX = br2.getNPArray(2)
    print testX.shape
    num_epochs = 200
    n = X.shape[0]
    sizeTrain = 0.8
    trainX = X[:n*sizeTrain].reshape(-1,1,48,48)
    trainY = Y[:n*sizeTrain].astype('uint8')
    validationX = X[n*sizeTrain:].reshape(-1,1,48,48)
    validationY = Y[n*sizeTrain:].astype('uint8')
    testX = testX.reshape(-1,1,48,48)
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = CNNL.convnetL(input_var)
    network.setParams('p3.npz')    
    #network.trainNetwork(trainX, trainY, validationX, validationY ,0.08 ,num_epochs,target_var,0.0001,'p3.npz')
    #network.getParams('p3.npz')
    y = np.array([])
    for i in range(0,testX.shape[0]/500):
        y = np.append(y,network.makePred(testX[i*500:(i+1)*500],input_var))

    f = file('CNNL_Prediction.csv','w')
    f.write("Id,Prediction\n")        
    for i,p in enumerate(y):
        f.write("%d,%d\n"%(i+1,p))
    f.close()