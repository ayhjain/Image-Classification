from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import LinearSVC
#from sklearn.ensemble import AdaBoostClassifier ##This doesnt work that well
from sklearn.metrics import accuracy_score

import pprint as pprint
import os
import numpy as np
import scipy.sparse as sps
import string
from collections import Counter
from datetime import datetime
from time import time
import cPickle
#import BatchReader

# Some global vars for cross validation and number of parallel jobs to run
kfolds = 5
num_jobs = -1

print("Loading images and associated labels...")


images = np.concatenate((np.load("..\\data\\TrainingData\\train_inputs1.npy"),np.load("..\\data\\TrainingData\\train_inputs2.npy"),np.load("..\\data\\TrainingData\\train_inputs3.npy"),np.load("..\\data\\TrainingData\\train_inputs4.npy"),np.load("..\\data\\TrainingData\\train_inputs5.npy"),np.load("..\\data\\TrainingData\\train_inputs6.npy"),np.load("..\\data\\TrainingData\\train_inputs7.npy"),np.load("..\\data\\TrainingData\\train_inputs8.npy"),np.load("..\\data\\TrainingData\\train_inputs9.npy"),np.load("..\\data\\TrainingData\\train_inputs10.npy"),np.load("..\\data\\TrainingData\\train_inputs11.npy"),np.load("..\\data\\TrainingData\\train_inputs12.npy"),np.load("..\\data\\TrainingData\\train_inputs13.npy"),np.load("..\\data\\TrainingData\\train_inputs14.npy"),np.load("..\\data\\TrainingData\\train_inputs15.npy"),np.load("..\\data\\TrainingData\\train_inputs16.npy"),np.load("..\\data\\TrainingData\\train_inputs17.npy"),np.load("..\\data\\TrainingData\\train_inputs18.npy"),np.load("..\\data\\TrainingData\\train_inputs19.npy"),np.load("..\\data\\TrainingData\\train_inputs20.npy")))
labels = np.concatenate((np.load("..\\data\\TrainingData\\train_outputs1.npy"),np.load("..\\data\\TrainingData\\train_outputs2.npy"),np.load("..\\data\\TrainingData\\train_outputs3.npy"),np.load("..\\data\\TrainingData\\train_outputs4.npy"),np.load("..\\data\\TrainingData\\train_outputs5.npy"),np.load("..\\data\\TrainingData\\train_outputs6.npy"),np.load("..\\data\\TrainingData\\train_outputs7.npy"),np.load("..\\data\\TrainingData\\train_outputs8.npy"),np.load("..\\data\\TrainingData\\train_outputs9.npy"),np.load("..\\data\\TrainingData\\train_outputs10.npy"),np.load("..\\data\\TrainingData\\train_outputs11.npy"),np.load("..\\data\\TrainingData\\train_outputs12.npy"),np.load("..\\data\\TrainingData\\train_outputs13.npy"),np.load("..\\data\\TrainingData\\train_outputs14.npy"),np.load("..\\data\\TrainingData\\train_outputs15.npy"),np.load("..\\data\\TrainingData\\train_outputs16.npy"),np.load("..\\data\\TrainingData\\train_outputs17.npy"),np.load("..\\data\\TrainingData\\train_outputs18.npy"),np.load("..\\data\\TrainingData\\train_outputs19.npy"),np.load("..\\data\\TrainingData\\train_outputs20.npy")))

n_samples = len(images)

print("Loaded "+str(n_samples)+" images.")

### Classifiers and Transformers ###
logit = LogisticRegression(solver='newton-cg',multi_class='multinomial')
lsvc = LinearSVC(tol=1e-3, multi_class='crammer_singer')
pca = PCA()
#ipca = IncrementalPCA()

### Pipeline
pipeline=Pipeline(steps=[
    # add transformer dictionary too
    ('pca', pca), 
    #('ipca', ipca),
    ('logit', logit)
    #('lsvc', lsvc)
])

### Parameters
parameters={
    # parameters to perform grid-search over
    'pca__n_components': [16, 64, 256, 576],
    #'ipca__n_components': [16, 64, 144],
    #'ipca__batch_size': [64, 144, 256, 576],
    #'logit__solver' : ('liblinear','newton-cg'), 
    'logit__C': np.logspace(-6,6,4)
    #'lsvc__C': (0.01, 0.5, 0.001)   # penalty for error term (C=0.01) etc.
}

if __name__ == '__main__':
    grid_search = GridSearchCV(
        pipeline, parameters, verbose = 1, cv = kfolds, n_jobs = num_jobs)

    print("Starting grid search with the following pipeline and parameters")
    print("Pipeline:", [name for name, _ in pipeline.steps])
    print("Parameters:")
    date=str(datetime.now()).split(" ")[0]
    fulltime=str(datetime.now()).split(" ")[1]
    realtime=fulltime.split(".")[0]
    #pickle_name=string.join((date, realtime), "--")
    pprint.pprint(parameters)
    t0=time()
    grid_search.fit(images, labels)
    print("Done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Estimator: ")
    pprint.pprint(grid_search.best_estimator_)
    print("Best results obtained with the following parameters:")
    best_params=grid_search.best_estimator_.get_params()
    for param_name in (sorted(parameters.keys())):
        print("\t%s: %r" % (param_name, best_params[param_name]))

    
    with open("..\\pickle\\logreg_ncg_pca2.pkl", "w") as fp:
        cPickle.dump(grid_search, fp)
    