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

################################################################################################

# Gobal Variables

LOW = -0.01
HIGH = 0.01
ERROR_LIMIT=1e-3

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

class ArtificialNeuralNetwork():


    def __init__(self, n_inputs, n_outputs, n_hidden_nodes, n_hidden_layers):
        self.nIn = n_inputs                
        self.nOut = n_outputs              
        self.nHiddenNodes = n_hidden_nodes              
        self.nHiddenLayers = n_hidden_layers

        # Setting weights of different nodes to random values
        # n is number of weights required (+1 for bais terms)
        n = ( (self.nIn+1)*self.nHiddenNodes )+ ( ((self.nHiddenNodes+1)*self.nHiddenNodes)*(self.nHiddenLayers-1) ) + (self.nHiddenNodes+1)*self.nOut
        self.weight = []
        self.weightUpdate( [random.uniform(LOW, HIGH) for i in xrange(n)] )


    
    def weightUpdate(self, weightList):
        '''
        weightUpdate : 
        Parameter: weightList takes list of weights to update model weights for all nodes
        Creates a list of weight arrays for each layer. 
        Using this kind of structure of optimized forward and back pass implementation
        '''
        self.weight = []
        firstLayer = weightList[:((self.nIn+1)*self.nHiddenNodes)]
        #print firstLayer
        weightList = weightList[((self.nIn+1)*self.nHiddenNodes): ]
        self.weight.append( np.array(firstLayer).reshape(self.nHiddenNodes, (self.nIn+1)) )

        for i in range(1, self.nHiddenLayers):
            hiddenLayer = weightList[:((self.nHiddenNodes+1)*self.nHiddenNodes)]
            self.weight.append( np.array(hiddenLayer).reshape(self.nHiddenNodes, (self.nHiddenNodes+1)) )
            weightList = weightList[((self.nHiddenNodes+1)*self.nHiddenNodes):]

        outputLayer = weightList
        self.weight.append( np.array(outputLayer).reshape(self.nOut, (self.nHiddenNodes+1)) )

        for layer, w in enumerate(self.weight):
            print 'layer: ', layer, 'w.shape: ', w.shape


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
            

    def backPropagation(self, trainX, trainY , alpha=0.3, momentum_factor=0.9  ):
        
        assert len(trainX[0]) == self.nIn, "ERROR: input size varies from the defined input setting"
        assert len(trainY[0]) == self.nOut, "ERROR: output size varies from the defined output setting"
        
        training_data = np.array( trainX )
        training_targets = np.array( trainY )
        
        mse = float("inf")
        momentum = collections.defaultdict( int )
        #print "Calculating weights inside backPropagation method"
        i = 0
        # Gradient descent till we get error below ERROR_LIMIT
        while mse > ERROR_LIMIT:
            out = self.forwardPass(training_data)
            #print "Back in backPropagation method"
            #print len(out), out[-1][:,:-1].shape, training_targets.shape       
            error = training_targets - out[-1][:,1:] # output from last layer - also excluding first column (the bias term)
            #print "i: ", i, "error shape = ", error.shape
            mse = np.mean( np.power(error,2) )
            delta = error

            # Iterate backwards on layer to update weights for each neuron
            for j in xrange((self.nHiddenLayers), -1, -1):
                #print 'j:', j
                w = self.weight[j]
                inp = out[j]
                #print 'j:', j, "inp.T shape = ", inp.T.shape, "delta.shape =", delta.shape
                #print inp.T

                dW = alpha * np.dot(inp.T, delta) + momentum_factor * momentum[j]
                '''
                if j != self.nHiddenLayers:
                    dW = alpha * np.dot(inp.T, delta[:,1:]) + momentum_factor * momentum[j]
                else:
                    dW = alpha * np.dot(inp.T, delta) + momentum_factor * momentum[j]
                '''
                #print 'dW.shape:', dW.shape
                
                # Update the weights
                '''
                if j != self.nHiddenLayers:
                    print "self.weight[j].shape: ",self.weight[j].shape, 'dW.T[1:,:].shape: ', dW.T[1:,:].shape
                    self.weight[j] += dW.T[1:,:]

                else:
                    print "self.weight[j].shape: ",self.weight[j].shape, 'dW.T.shape: ', dW.T.shape
                    self.weight[j] += dW.T
                '''
                #print "self.weight[j].shape: ",self.weight[j].shape, 'dW.T.shape: ', dW.T.shape
                self.weight[j] += dW.T
                #print dW.T

                # Calculate previous layer's delta for next calculation
                if j!= 0:
                 #   print "delta.shape:", delta.shape, "w.shape: ", w.shape
                    delta = np.multiply( np.dot( delta, w[:,1:] ) , self.sigmoid_function(inp[:,1:], derivative=True) )
                 #   print 'delta.shape: ', delta.shape
                
                # Store the momentum
                momentum[j] = dW
                 
            i += 1
            if i%1000==0:
                print "MSE after %g iterations: %d" % (mse, i)
        
        print "Converged to error bound (%.4g) with MSE = %.4g." % ( ERROR_LIMIT, mse )
        print "Training ran for %d iterations." % i



    def forwardPass(self, inp_vec):
        # each entry of output list has 1 appended as bias term at 0th index
        #print "In Forward Pass"
        output = [ self.biasTerm(inp_vec) ]
        for w in self.weight:
            # Looping over network layers to calculate output
            #print "output[-1].shape: ",output[-1].shape, 'w.T.shape: ',w.T.shape 
            # print w[:,1:].T.shape
            o = np.dot( output[-1], w.T ) # adding bias term for each layer
            o = self.sigmoid_function(o)
            output.append( self.biasTerm(o) )
        #print len(output), output[0].shape
        #print 
        #for term in output: print term
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






if __name__=='__main__':

    X =  [ [0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0] ]
    Y = X
    n_inputs = 8
    n_outputs = 8
    n_hiddens = 3
    n_hidden_layers = 2

     # initialize the neural network
    network = ArtificialNeuralNetwork(n_inputs, n_outputs, n_hiddens, n_hidden_layers)

    # start training on test set one
    network.backPropagation(X, Y, alpha=0.4, momentum_factor=0.9  )

    # save the trained network
    network.dumpToFile()

    # load a stored network configuration
    # network.loadFromMemory()

    # print out the result
    i= [0,0,0,0,0,0,0,1]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i

    i= [1,0,0,0,0,0,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i

    i= [0,1,0,0,0,0,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i

    i= [0,0,1,0,0,0,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i
    i= [0,0,0,1,0,0,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i
    i= [0,0,0,0,1,0,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i
    i= [0,0,0,0,0,1,0,0]
    print i, network.forwardPass( np.array(i) )[-1][:,1:], "\ttarget:", i
