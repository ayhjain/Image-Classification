################################################################################################

# Artificial Neural Network
# Created on 31st October

################################################################################################

import numpy as np
import os, csv
import math
import random
import itertools
import collections
import pickle
from sklearn import datasets
import BatchReader

################################################################################################

# Gobal Variables

LOW = -0.01
HIGH = 0.01
ERROR_LIMIT=0.004
BATCH_SIZE = 7
ITER_LIMIT = 500000

# Setting seed for random for consistent results
#random.seed(12345)

################################################################################################

# Classes

'''
    Neural Network class
    Parameters - 
    n_inputs = Number of network input signals
    n_outputs = Number of desired outputs from the network
    n_hidden_nodes = Number of nodes in each hidden layer
    n_hidden_layers = Number of hidden layers in the network
'''

class ArtificialNeuralNetwork:

    def __init__(self, n_inputs, n_outputs, n_hidden_nodes, n_hidden_layers):
        self.nIn = n_inputs                
        self.nOut = n_outputs              
        self.nHiddenNodes = n_hidden_nodes              
        self.nHiddenLayers = n_hidden_layers
        print self.nIn, self.nOut, self.nHiddenNodes, self.nHiddenLayers

        # Creates a list of weight arrays for each layer. Setting each weight to random values. 
        # Using this kind of structure of optimized forward and back pass implementation
        self.weight = []
        firstLayer = [random.uniform(LOW, HIGH) for i in xrange( (self.nIn+1)*self.nHiddenNodes ) ]
        self.weight.append( np.array(firstLayer).reshape((self.nIn+1), self.nHiddenNodes) )

        for i in range(1, self.nHiddenLayers):
            hiddenLayer = [random.uniform(LOW, HIGH) for i in xrange( (self.nHiddenNodes+1)*self.nHiddenNodes ) ]
            self.weight.append( np.array(hiddenLayer).reshape( (self.nHiddenNodes+1), self.nHiddenNodes) )
            
        outputLayer = [random.uniform(LOW, HIGH) for i in xrange( (self.nHiddenNodes+1)*self.nOut ) ]
        self.weight.append( np.array(outputLayer).reshape((self.nHiddenNodes+1), self.nOut) )

        for layer, w in enumerate(self.weight):
            print 'layer: ', layer, 'w.shape: ', w.shape, 'np.max(w.T):', np.max(w.T)



    def biasTerm(self, input):
            # Add bias term of 1 at start of each data row
            n,m = input.shape
            return np.hstack(( np.ones(shape=[n,1]), input ))

    

    def sigmoid_function(self, x, derivative=False):
        # the activation function
        try:
            signal = 1/(1+math.e**(-x))
            
        except OverflowError:
            signal = float("inf")

        if derivative:
            # Return the partial derivation of the activation function
            return np.multiply(signal, 1-signal)
        else:
            # Return the activation signal
            return signal
     


    def backPropagation(self, trainX, trainY, alpha=0.3, momentum_factor=0.9  ):
                
        mse = float("inf")
        momentum = collections.defaultdict( int )
        
        i = 0

        # Gradient descent till we get error below ERROR_LIMIT or loop iterates for more than ITER_LIMIT
        while mse > ERROR_LIMIT and i<ITER_LIMIT:
            i += 1
            #print 'i:', i
            
            index = np.random.randint(0, len(trainX), BATCH_SIZE)
            #index = np.array([0,1,2,3,4,5,6,7])
            training_data = np.array([trainX[k] for k in index]).astype(np.float32)
            training_targets = np.array([trainY[k] for k in index]).astype(np.float32)
            
            '''
            training_targets = np.zeros(shape=(BATCH_SIZE, self.nOut)).astype(np.float32)
            for l, k in enumerate(index): training_targets[l,trainY[k]] = 1
            #print training_targets
            '''

            out = self.forwardPass(training_data)
                        
            #print "Back in backPropagation method"
            print 'out[-1][:,1:].shape:', out[-1][:,1:].shape, 'training_targets.shape:', training_targets.shape       
                              
            error = training_targets - out[-1][:,1:]
            delta = error
            mse = np.mean( np.power(error,2) )

            
            for layer in xrange((self.nHiddenLayers), -1, -1):
                # Loop over the weight layers in reversed order to calculate the deltas
                #print "Layer: ", layer
                w = self.weight[layer]
                layerSignal = out[layer]
                
                # Calculate weight change 
                #print 'layerSignal.T.shape:', layerSignal.T.shape, 'delta.shape:', delta.shape
                dw = alpha * np.dot( layerSignal.T, delta ) + momentum_factor * momentum[layer]
                # Update the weights
                #print "self.weight[layer].shape: ",self.weight[layer].shape, 'dW.shape: ', dW.shape
                self.weight[layer] += dw

                # Calculate the delta for the subsequent layer
                if layer!= 0:
                    #print "delta.shape:", delta.shape, "w[1:,:].T.shape: ", w[1:,:].T.shape
                    delta = np.multiply(  np.dot( delta, w[1:,:].T ), self.sigmoid_function( layerSignal[:,1:], derivative=True) )
                
                # Store the momentum
                momentum[layer] = dw
                

            if i%1000==0:
                # Show the current training status
                print "* current network error (mse):", mse
        
        print "* Converged to error bound (%.4g) with mse = %.4g." % ( ERROR_LIMIT, mse )
        print "* Trained for %d iterations." % i



    
    def forwardPass(self, inp_vec):
        # each entry of output list has 1 appended as bias term at 0th index
        #print "In Forward Pass"
        output = [ self.biasTerm(inp_vec) ]
        for w in self.weight:
            # Looping over network layers to calculate output
            #print "output[-1].shape: ",output[-1].shape, 'w.T.shape: ',w.T.shape 
            # print w[:,1:].T.shape
            o = np.dot( output[-1], w ) # adding bias term for each layer
            o = self.sigmoid_function(o)
            output.append( self.biasTerm(o) )
        #print len(output), output[0].shape
        #print 
        #for term in output: print term.shape
        return output

    

    def dumpToFile(self):
        print "Dumping parameters to ../data/ann_parameters.pkl"
        dir = os.getcwd()
        path = os.path.join(dir,"..//data")
        os.chdir(path)

        with open( "ann_parameters.pkl" , 'wb') as file:
            dict = {
                "nIn" : self.nIn,
                "nOut" : self.nOut,
                "nHiddenNodes" : self.nHiddenNodes,
                "nHiddenLayers" : self.nHiddenLayers,
                "weight" : self.weight
            }
            pickle.dump(dict, file, 2 )
        os.chdir(dir)



    def loadFromMemory(self):
        dir = os.getcwd()
        path = os.path.join(dir,"..//data")
        os.chdir(path)

        if os.path.isfile("ann_parameters.pkl") :
            print "Reading parameters from ../data/ann_parameters.pkl"
            with open(ann_parameters.pkl , 'rb') as file:
                d = pickle.load(file)
                self.nIn = d["nIn"]            
                self.nOut = d["nOut"]           
                self.nHiddenNodes = d["nHiddenNodes"]           
                self.nHiddenLayers = d["nHiddenLayers"]     
                self.weight = d["weight"]
        os.chdir(dir) 

    
#end class



if __name__=='__main__':

    #####################
    # Autoencoder
    #####################
    X = np.array([ [1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1] ]).astype(np.float32)
    Y=X
    trainX = X[:-1,:]
    trainY = Y[:-1,:]

    testX = X[-2:,:]
    testY = Y[-2:,:]

    
    '''
    ######################
    # Boolean OR function
    ######################
    X = [[0,0, 0],[0,0, 1],[0,1, 0],[0,1, 1],[1,0, 0],[1,0, 1],[1,1, 0]]
    Y = [[0],[1],[1],[1],[1],[1],[1]]
    
    ######################
    # IRIS datasets
    ######################
    iris = datasets.load_digits()
    X = iris.data
    Y = iris.target
    n_inputs = X.shape[1]
    n_outputs = len(Y)
    print X
    print Y
    
    #####################
    # Actual Image data 
    #####################
    br = BatchReader.inputs()
    
    X, Y = br.getNPArray(2)
    trainX = X[1000:,:]
    trainY = Y[1000:,:]

    testX = X[1000:1008,:]
    testY = Y[1000:1008,:]
    '''
    
    print X.shape, Y.shape
    n_inputs = X.shape[1]
    
    n_outputs = Y.shape[1]
    n_hiddens = 3
    n_hidden_layers = 1

    # initialize the neural network
    network = ArtificialNeuralNetwork(n_inputs, n_outputs, n_hiddens, n_hidden_layers)

    # start training on test set one
    network.backPropagation(X, Y, alpha=0.4, momentum_factor=0.9  )

    # save the trained network
    network.dumpToFile()
    
    # load a stored network configuration
    # network.loadFromMemory()
   # print [1,1,1], network.forwardPass(np.array([[1,1,1]]))[-1][:,1:]
    # print out the result

    

    predict = network.forwardPass(X)[-1][:,1:]
    for i in range(X.shape[0]):
        print np.argmax(predict[i,:]), Y[i,:], predict[i,:]
    print 

    '''
    print predict.shape, testY.shape
    count=0
    for i in range(testY.shape[0]):
        print np.argmax(predict[i]), predict[i], testY[i]
        if(np.argmax(testY[i]) != np.argmax(predict[i])): count += 1
    print "Testing error: ", count
    '''