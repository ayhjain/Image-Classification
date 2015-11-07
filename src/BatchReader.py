import numpy as np
import csv, os
import os.path

BATCH_SIZE = 50000

class inputs(object):

    def __init__(self, testingData=False):
        self.counter = 0
        self.testingData = testingData
        if not testingData:
            self.infile = open('..\data\train_inputs.csv', 'rb')
            self.inreader = csv.reader(self.infile, delimiter=',')
            next(self.inreader, None)  # skip the header

            self.outfile = open('..\data\train_outputs.csv', 'rb')
            self.outreader = csv.reader(self.outfile, delimiter=',')
            next(self.outreader, None)  # skip the header
        
        else :
            self.infile = open('..\data\test_inputs.csv', 'rb')
            self.inreader = csv.reader(self.infile, delimiter=',')
            next(self.inreader, None)  # skip the header


    def getCounter(self):
        return self.counter


    def getNPArray(self, d):
        # Returns array in (0-999)*k batches
        
        c = (d/BATCH_SIZE) + 1

        if not self.testingData:
            # Load 'BATCH_SIZE' training inputs to a python list
            dir = os.getcwd()
            path = os.path.join(dir,"..\data\TrainingData")
            os.chdir(path)

            if os.path.isfile("train_inputs"+str(c)+".npy") and os.path.isfile("train_outputs"+str(c)+".npy") : 
                print ("Data picked from memory.")
                train_inputs_np = np.load("train_inputs"+str(c)+".npy")
                train_outputs_np = np.load("train_outputs"+str(c)+".npy")

            else:
                while (self.counter < c):
                    self.counter += 1
            
                    i=0
                    train_inputs = []
                    for train_input in self.inreader: 
                        train_input_no_id = []
                        for dimension in train_input[1:]:
                            train_input_no_id.append(float(dimension))
                        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list
                        i+=1
                        if i>=BATCH_SIZE: break

                    # Load training ouputs to a python list
                    i=0
                    train_outputs = []
                    for train_output in self.outreader:  
                        train_output_no_id = int(train_output[1])
                        train_outputs.append(train_output_no_id)
                        i+=1
                        if i>=BATCH_SIZE: break

                    # Convert python lists to numpy arrays
                    train_inputs_np = np.asarray(train_inputs, dtype='float32')
                    train_outputs_np = np.asarray(train_outputs, dtype='float32')

                    # Save as numpy array files
                    np.save('train_inputs'+str(self.counter), train_inputs_np)
                    np.save('train_outputs'+str(self.counter), train_outputs_np)

            os.chdir(dir)
            return train_inputs_np, train_outputs_np
    

        else : # Reading Test data
            dir = os.getcwd()
            path = os.path.join(dir,"..\data\TestingData")
            os.chdir(path)

            if os.path.isfile("test_inputs"+str(c)+".npy") : 
                print ("Data picked from memory.")
                test_inputs_np = np.load("train_inputs"+str(c)+".npy")

            else:
                while (self.counter <= c):
                    self.counter += 1
                    
                    i=0
                    test_inputs = []
                    for test_input in self.inreader: 
                        test_input_no_id = []
                        for dimension in test_input[1:]:
                            test_input_no_id.append(float(dimension))
                        test_inputs.append(test_input_no_id) # Load each sample as a numpy array, which is appened to the python list
                        i+=1
                        if i>=BATCH_SIZE: break

                    # Convert python lists to numpy arrays
                    test_inputs_np = np.asarray(test_inputs, dtype='float32')

                    # Save as numpy array files
                    np.save('test_inputs'+str(self.counter), test_inputs_np)
                    os.chdir('..')

            os.chdir(dir)
            return test_inputs_np


