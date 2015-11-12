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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
theano.config.floatX = 'float32'

def loadData():
    sizeTrain = 0.7
    sizeVal = 0.2
    br = BatchReader2.inputs()
    br2 = BatchReader2.inputs(testingData = True)
    X, Y = br.getNPArray(2)
    n = X.shape[0]
    testX = br2.getNPArray(2)
    trainX = X[:n*sizeTrain].reshape(-1,1,48,48)
    trainY = Y[:n*sizeTrain].astype('uint8')
    validationX = X[n*sizeTrain:n*(sizeTrain+sizeVal)+1].reshape(-1,1,48,48)
    validationY = Y[n*sizeTrain:n*(sizeTrain+sizeVal)+1].astype('uint8')
    print (trainX.shape)
    print (validationX.shape)
    print (validationY.shape)
    tX = X[n*(sizeTrain+sizeVal)+1:].reshape(-1,1,48,48)
    tY = Y[n*(sizeTrain+sizeVal)+1:].astype('uint8')
    print (tX.shape)
    testX = testX.reshape(-1,1,48,48)
    return trainX,trainY,validationX,validationY,tX,tY,testX

def getPredictions(x,input_var):
    y = np.array([])
    for i in range(0,x.shape[0]/500):
        pred = network.makePred(x[i*500:(i+1)*500],input_var)
        y = np.append(y,pred)
    return y.astype('uint8')
	
def analysis(predictions,tY):
	precision = np.array([])
	recall = np.array([])
	f1 = np.array([])
	acc = np.array([])
	acc = np.append(acc,accuracy_score(tY,predictions))
	precision = np.append(precision,precision_score(tY,predictions,average='macro'))
	recall = np.append(recall,recall_score(tY,predictions,average='macro'))
	f1 = np.append(f1,f1_score(tY,predictions,average='macro'))
	return [acc, precision, recall, f1]
	
def getAcc(predictions,valY):
	return [accuracy_score(valY,predictions)]
	
def writeToFile(path,array,header):
    f = file(path,'a')
    f.write(header+"\n")	
    for i,p in enumerate(array):
        f.write(str(i+1)+","+str(p))
    f.close()
	
if __name__ == '__main__':
    
    rng = numpy.random.RandomState(42)
    num_epochs = 200
    lRate = 0.1
    model_type = 3
    training = False
    loadmodel = True
    predict = False
	
    trainX,trainY,validationX,validationY,tX,tY,testX = loadData()
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    enable = False
       
    if model_type == 1:
		path = 'p1.npz'
		filter = [16,32,256]
    elif model_type == 2:
		path = 'p2.npz'
		filter = [8,16,128]
    else:
		path = 'p3.npz'
		filter = [16,32,128]
		enable = True
		
    network = CNNL.convnetL(input_var,enable,filter)
	
    if loadmodel:
		print ("setting params")
		network.setParams(path)
    if training:		
        network.trainNetwork(trainX, trainY, validationX, validationY ,lRate ,num_epochs ,target_var, path)
	   
    
    pred = getPredictions(validationX,input_var)
    a = getAcc(pred,validationY)
    writeToFile('accuracy.csv',a,"id,accuracy")
	
    if model_type == 3:	
		pred = getPredictions(tX,input_var)
		a = analysis(pred,tY)
		writeToFile('scores.csv',a,"id,precision,recall,f1")
		labels = [0,1,2,3,4,5,6,7,8,9]
		cm = confusion_matrix(tY, pred, labels)
		print(cm)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		cax = ax.matshow(cm)
		pl.title('Confusion matrix for convolutional neural network')
		fig.colorbar(cax)
		ax.set_xticklabels([''] + labels)
		ax.set_yticklabels([''] + labels)
		pl.xlabel('Predicted Class')
		pl.ylabel('True Class')
		pl.show()
		