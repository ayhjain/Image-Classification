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
import BatchReader
import scipy.misc
import pickle



theano.config.floatX = 'float32'
K_FOLD = 5



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            name = 'W',
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input





class Network(object):

    def __init__(self, batch_size = 200, 
                    nkerns = [40,40,200], 
                    dim_filter = [9, 5, 5], 
                    num_hidden = [100,10], 
                    gamma = 1e-6):

        self.batch_size = batch_size
        self.lowestError = 1.
        x = T.matrix()
        y = T.matrix()
        learning_rate = T.scalar()
        

        print '... building the model'

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((batch_size, 1, 48, 48))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer( 
            rng, 
            input=layer0_input,
            image_shape=(batch_size, 1, 48, 48),
            filter_shape=(nkerns[0], 1, 9, 9),
            poolsize=(2, 2)
        )
        
        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 20, 20),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer2 = LeNetConvPoolLayer(
            rng,
            input=layer1.output,
            image_shape=(batch_size, nkerns[1], 8, 8),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2)
        )

        #layer2_param, layer2_out = layer2.params, layer2.output

        #img_dim = ( img_dim - dim_filter[2] + 1 )/2

        layer3_input = layer2.output.flatten(2)
        layer3 = HiddenLayer(
            rng,
            input=layer3_input,
            n_in=nkerns[2] * 2 * 2,
            n_out=500,
            activation=T.tanh
        )

        '''
        layer4 = LogisticRegression(
            input=layer3.output,
            n_in = num_hidden[0],
            n_out = num_hidden[1]
        )

        params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params         

        # using L2 regularization
        #L2_reg = sum([T.sum(i**2) for i in params if 'W' in i.name])
        
        cost = layer4.negative_log_likelihood(np.argmax(y))
        #cost += gamma * L2_reg
        '''
        layer4_input = layer3.output.flatten(2)
        layer4 = HiddenLayer(
            rng,
            input=layer4_input,
            n_in=500,
            n_out=10,
            activation=T.tanh
        )
        params = layer0.params + layer1.params + layer2.params + layer3.params + layer4.params   

        # using L2 regularization
        L2_reg = sum([T.sum(i**2) for i in params if i.name == 'W'])
        
        cost = T.sum((y-layer4.output)**2)/y.shape[0]
        cost += gamma*L2_reg
        grads = T.grad(cost, params)
        
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]
        
        self.train_model =theano.function(
            [x,y,learning_rate],
            cost,
            updates=updates
        )
        
        self.eval_net = theano.function([x],layer4.output)

        self.params = params
        self.epoch = 0


    def rotate(self, X,r):
        return np.float32([1./255 *scipy.misc.imrotate(i.reshape((48,48)),r).flatten() for i in X])


    def getParams(self):
        return {'W':[i.get_value() for i in self.params],'epoch':self.epoch}
    
    
    def setParams(self, dict_params):
        for i,p in zip(dict_params['W'],self.params):
            p.set_value(i)
        self.epoch = dict_params['epoch']



    def trainNetwork(self, trainX, trainY,  validationX, validationY, learning_rate = 0.1, nepochs=800, tau = 250):
        
        n_batches = trainX.shape[0] / self.batch_size

        start = self.epoch
        print "Epoch:", self.epoch

        # train for a while
        for i in range(start, nepochs):
            self.epoch = i
            c = 0
            t0 = timeit.default_timer()
            for batch in range(n_batches):
                x = trainX[batch*self.batch_size:(batch+1)*self.batch_size]
                y = trainY[batch*self.batch_size:(batch+1)*self.batch_size]
                alpha = learning_rate * tau / (i * 1. + tau)
                x = self.rotate(x, np.random.uniform(0,360))
                
                c += self.train_model(x, y, alpha)
            
            print i,c,timeit.default_timer()-t0
            
            if i%50 == 0 or i==nepochs-1:
                err = self.calcError(trainX,trainY)
                print "Train Error:", err
                err = self.calcError(validationX, validationY)
                print "Validation Error:", err

                if err[1] < self.lowestError :
                    self.lowestError = err[1]
                    f = file('Conv_Params.pkl','w')
                    pickle.dump(self.getParams(), f)
                    f.close()



    def calcError(self, X, Y):
        predict = self.evaluate(X)
        err = 1-1.*np.equal(np.argmax(predict,axis=1),(np.argmax(Y,axis=1))).sum() / Y.shape[0]
        return ((Y-predict)**2).sum()/Y.shape[0], err

    
    def evaluate(self, x):
        y_i = []
        for k in range(x.shape[0] / self.batch_size):
            y_k = []
            for angle in range(0,360,60):
                y_k.append(self.eval_net(
                    self.rotate( x[k*self.batch_size:(k+1)*self.batch_size],angle) )
                )
            y_i.append(np.float32(y_k).mean(axis=0))
        y = np.vstack(y_i)
        return y




if __name__ == '__main__':
    
    rng = numpy.random.RandomState(23455)
    br = BatchReader.inputs()
    br2 = BatchReader.inputs(testingData = True)

    X, Y = br.getNPArray(2)
    testX = br2.getNPArray(2)
    nkerns = [35,70,35]

    print X.shape, Y.shape
    print testX.shape

    n_examples = X.shape[0]
    fs = foldsize = n_examples/K_FOLD
    
    # k folds
    for k in range(K_FOLD):
        trainX = np.vstack(
            [X[:k*fs], X[(k+1)*fs:]]
        )
        
        trainY = np.vstack(
            [Y[:k*fs],Y[(k+1)*fs:]]
        )
        
        validationX = X[k*fs:(k+1)*fs]
        validationY = Y[k*fs:(k+1)*fs]

        print trainX.shape,trainY.shape,validationX.shape,validationY.shape
        
        conv_net = Network(batch_size = 250)

        '''
        try:
            dict_params = pickle.load(file('Conv_Params.pkl','r'))
            conv_net.setParams(dict_params)
            print "Imported weights"
        except IOError:
            pass
        '''

        print "Training for %dth KFold" %(k+1)        
        conv_net.trainNetwork(trainX, trainY, validationX, validationY)
        
        
    # Now dump file has the parameters for lowest error on Validation set
    dict_params = pickle.load(file('Conv_Params.pkl','r'))
    conv_net.setParams(dict_params)

    y = conv_net.evaluate(test_X)
    f = file('Test_Prediction.csv','w')
    f.write("Id,Prediction\n")        
    for i,p in enumerate(y):
        f.write("%d,%d\n"%(i+1,np.argmax(p)))
    f.close()
    
    print "Done!!!"