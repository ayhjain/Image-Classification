from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import LinearSVC
#from sklearn.ensemble import AdaBoostClassifier ##This doesnt work that well
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import validation_curve

import pprint as pprint
import os
import cPickle
import numpy as np
import scipy.sparse as sps
import string
from collections import Counter
from datetime import datetime
from time import time
import BatchReader

# Some global vars for cross validation and number of parallel jobs to run
kfolds = 5
num_jobs = -1

print("Loading images and associated labels...")
images = np.concatenate((np.load("..\\data\\TrainingData\\train_inputs1.npy"),np.load("..\\data\\TrainingData\\train_inputs2.npy"),np.load("..\\data\\TrainingData\\train_inputs3.npy"),np.load("..\\data\\TrainingData\\train_inputs4.npy"),np.load("..\\data\\TrainingData\\train_inputs5.npy"),np.load("..\\data\\TrainingData\\train_inputs6.npy"),np.load("..\\data\\TrainingData\\train_inputs7.npy"),np.load("..\\data\\TrainingData\\train_inputs8.npy"),np.load("..\\data\\TrainingData\\train_inputs9.npy"),np.load("..\\data\\TrainingData\\train_inputs10.npy")))
labels = np.concatenate((np.load("..\\data\\TrainingData\\train_outputs1.npy"),np.load("..\\data\\TrainingData\\train_outputs2.npy"),np.load("..\\data\\TrainingData\\train_outputs3.npy"),np.load("..\\data\\TrainingData\\train_outputs4.npy"),np.load("..\\data\\TrainingData\\train_outputs5.npy"),np.load("..\\data\\TrainingData\\train_outputs6.npy"),np.load("..\\data\\TrainingData\\train_outputs7.npy"),np.load("..\\data\\TrainingData\\train_outputs8.npy"),np.load("..\\data\\TrainingData\\train_outputs9.npy"),np.load("..\\data\\TrainingData\\train_outputs10.npy")))

n_samples = len(images)

print("%d images" % n_samples)
print()

X_train = images[:.9*n_samples]
y_train = labels[:.9*n_samples]

### Classifiers and Transformers ###
logit = LogisticRegression()
lsvc = LinearSVC(tol=1e-3, multi_class='crammer_singer')
#pca = PCA()
ipca = IncrementalPCA()

pipeline=Pipeline(steps=[
    # add transformer dictionary too
    ('ipca', ipca), 
    #('pca', pca), 
    ('logit', logit)
    #('lsvc', lsvc)
])

parameters={
    # parameters to perform grid-search over
    #'vect__min_df': (0.01, 0.02, 0.05)
    #'mit__percentage': (0.5, 0.67)
    #'pca__n_components': [64, 144, 256, 576],
    #'logit__solver' : ('liblinear','newton-cg'), 
    'ipca__batch_size': [144,256,576],
    'ipca__n_components': [64, 144, 256, 576],
    'logit__C': np.logspace(-4, 4, 3)
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
    grid_search.fit(X_train, y_train)
    print("Done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Estimator: ")
    pprint.pprint(grid_search.best_estimator_)
    print("Best results obtained with the following parameters:")
    best_params=grid_search.best_estimator_.get_params()
    for param_name in (sorted(parameters.keys())):
        print("\t%s: %r" % (param_name, best_params[param_name]))

    
    with open("..\\pickle\\logisticreg_ipca.pkl", "w") as fp:
        cPickle.dump(grid_search, fp)
    print ("Pickled as..\\pickle\\logisticreg_ipca.pkl")