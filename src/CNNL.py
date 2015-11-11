#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import BatchReader2




def load_dataset():
    br = BatchReader2.inputs()
    br2 = BatchReader2.inputs(testingData = True)
    X, Y = br.getNPArray(3)
    X_train = X[:50000*0.8].reshape(-1, 1, 48, 48)
    y_train = Y[:50000*0.8].astype('uint8')
    X_val = X[50000*0.8:].reshape(-1, 1, 48, 48)
    y_val = Y[50000*0.8:].astype('uint8')
    return X_train, y_train, X_val, y_val

class convnetL:

	def __init__(self,input_var):
		

		# Input layer, as usual:
		l_in = lasagne.layers.InputLayer(shape=(None, 1, 48, 48),
											input_var=input_var)
			
		# Convolutional layer with 32 kernels of size 5x5. Strided and padded
		# convolutions are supported as well; see the docstring.
		network = lasagne.layers.Conv2DLayer(
				l_in, num_filters=16, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify,
				W=lasagne.init.GlorotUniform())
				
		# Expert note: Lasagne provides alternative convolutional layers that
		# override Theano's choice of which implementation to use; for details
		# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.
		# Max-pooling layer of factor 2 in both dimensions:
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

		# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=32, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
				
		network = lasagne.layers.Conv2DLayer(
				network, num_filters=32, filter_size=(5, 5),
				nonlinearity=lasagne.nonlinearities.rectify)
				
		network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
		

		# A fully-connected layer of 256 units with 50% dropout on its inputs:
				
		network = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=128,
				nonlinearity=lasagne.nonlinearities.sigmoid)
		# And, finally, the 10-unit output layer with 50% dropout on its inputs:
		self.out = lasagne.layers.DenseLayer(
				lasagne.layers.dropout(network, p=.5),
				num_units=10,
				nonlinearity=lasagne.nonlinearities.softmax)
		
		self.input_var = input_var
	
	def getParams(self,path):
		np.savez(path,*lasagne.layers.get_all_param_values(self.out))
		
	def setParams(self,path):
		with np.load(path) as f:
			param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(self.out, param_values)
	
	def trainNetwork(self, X_train, y_train, X_val, y_val,alpha,num_epochs,target_var,coeff,path):
		# Prepare Theano variables for inputs and targets

		# Create neural network model (depending on first command line parameter)
		print("Building model and compiling functions...")
		
		# Create a loss expression for training, i.e., a scalar objective we want
		# to minimize (for our multi-class problem, it is the cross-entropy loss):
		prediction = lasagne.layers.get_output(self.out)
		loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
		loss = loss.mean()
		#reg = lasagne.regularization.regularize_layer_params(self.out,lasagne.regularization.l2)
		loss = loss#+reg*coeff
		# We could add some weight decay as well here, see lasagne.regularization.

		# Create update expressions for training, i.e., how to modify the
		# parameters at each training step. Here, we'll use Stochastic Gradient
		# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
		params = lasagne.layers.get_all_params(self.out, trainable=True)
		updates = lasagne.updates.nesterov_momentum(
				loss, params, learning_rate=alpha, momentum=0.90)

		# Create a loss expression for validation/testing. The crucial difference
		# here is that we do a deterministic forward pass through the network,
		# disabling dropout layers.
		test_prediction = lasagne.layers.get_output(self.out, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
																target_var)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
						  dtype=theano.config.floatX)

		# Compile a function performing a training step on a mini-batch (by giving
		# the updates dictionary) and returning the corresponding training loss:
		train_fn = theano.function([self.input_var, target_var], loss, updates=updates)

		# Compile a second function computing the validation loss and accuracy:
		val_fn = theano.function([self.input_var, target_var], [test_loss, test_acc])

		# Finally, launch the training loop.
		print("Starting training...")
		# We iterate over epochs:
		bestVal = 0.0
		for epoch in range(num_epochs):
			# In each epoch, we do a full pass over the training data:
			train_err = 0
			train_batches = 0
			start_time = time.time()
			for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
				inputs, targets = batch
				train_err += train_fn(inputs, targets)
				train_batches += 1

			# And a full pass over the validation data:
			val_err = 0
			val_acc = 0
			val_batches = 0
			for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
				inputs, targets = batch
				err, acc = val_fn(inputs, targets)
				val_err += err
				val_acc += acc
				val_batches += 1
			if (epoch+1)%50 == 0:
				alpha = alpha-0.02
				updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=alpha, momentum=0.90)
				train_fn = theano.function([self.input_var, target_var], loss, updates=updates)
				
			# Then we print the results for this epoch:
			print("Epoch {} of {} took {:.3f}s".format(
				epoch + 1, num_epochs, time.time() - start_time))
			print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
			print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
			print("  validation accuracy:\t\t{:.2f} %".format(
				val_acc / val_batches * 100))
			if (val_acc / val_batches * 100) > bestVal :
				bestVal = val_acc / val_batches * 100
				self.getParams(path)
				
	def makePred(self,testX,input_var):
		test_prediction = lasagne.layers.get_output(self.out, deterministic=True)
		predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
		return predict_fn(testX)
		
# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




    