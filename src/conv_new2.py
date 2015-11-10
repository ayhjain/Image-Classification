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
K_FOLD = 2


if __name__ == '__main__':
    
    rng = numpy.random.RandomState(42)
    br = BatchReader2.inputs()
    br2 = BatchReader2.inputs(testingData = True)

    X, Y = br.getNPArray(2)
    testX = br2.getNPArray(2)
    nkerns = [35,70,35]
    num_epochs = 2
    n_examples = X.shape[0]
    fs = foldsize = n_examples/K_FOLD
    
    # k folds
    for k in range(K_FOLD):
        trainX = np.vstack([X[:k*fs], X[(k+1)*fs:]]).reshape(-1,1,48,48)
        trainY = np.append(Y[:k*fs],Y[(k+1)*fs:]).astype('uint8')
        validationX = X[k*fs:(k+1)*fs].reshape(-1,1,48,48)
        validationY = Y[k*fs:(k+1)*fs].astype('uint8')
        
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        network = CNNL.convnetL(input_var)
        #network.setParams('model5.npz')
        print "Training for %dth KFold" %(k+1) 
        network.trainNetwork(trainX, trainY, validationX, validationY ,0.05 ,num_epochs,target_var,0.0001)
    
        network.getParams('model5.npz')
        '''
        try:
            dict_params = pickle.load(file('Conv_Params.pkl','r'))
            conv_net.setParams(dict_params)
            print "Imported weights"
        except IOError:
            pass
        '''

        # To be removed
        if k == 2:
           break
        
    '''    
    # Now dump file has the parameters for lowest error on Validation set
    dict_params = pickle.load(file('Conv_Params.pkl','r'))
    conv_net.setParams(dict_params)
    y = conv_net.evaluate(test_X)
    f = file('Test_Prediction.csv','w')
    f.write("Id,Prediction\n")        
    for i,p in enumerate(y):
        f.write("%d,%d\n"%(i+1,np.argmax(p)))
    f.close()
    '''
    print "Done!!!"