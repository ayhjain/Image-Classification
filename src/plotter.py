# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:20:48 2015

@author: Suzin
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit
#from String import find


# Loading the first 20000 as the training set and third 10000 as the test set.

X_train = np.concatenate((np.load("..\\data\\TrainingData\\train_inputs1.npy"),np.load("..\\data\\TrainingData\\train_inputs2.npy"),np.load("..\\data\\TrainingData\\train_inputs3.npy"),np.load("..\\data\\TrainingData\\train_inputs4.npy"),np.load("..\\data\\TrainingData\\train_inputs5.npy"),np.load("..\\data\\TrainingData\\train_inputs6.npy"),np.load("..\\data\\TrainingData\\train_inputs7.npy"),np.load("..\\data\\TrainingData\\train_inputs8.npy"),np.load("..\\data\\TrainingData\\train_inputs9.npy"),np.load("..\\data\\TrainingData\\train_inputs10.npy"),np.load("..\\data\\TrainingData\\train_inputs11.npy"),np.load("..\\data\\TrainingData\\train_inputs12.npy"),np.load("..\\data\\TrainingData\\train_inputs13.npy"),np.load("..\\data\\TrainingData\\train_inputs14.npy"),np.load("..\\data\\TrainingData\\train_inputs15.npy"),np.load("..\\data\\TrainingData\\train_inputs16.npy"),np.load("..\\data\\TrainingData\\train_inputs17.npy"),np.load("..\\data\\TrainingData\\train_inputs18.npy"),np.load("..\\data\\TrainingData\\train_inputs19.npy"),np.load("..\\data\\TrainingData\\train_inputs20.npy")))
y_train = np.concatenate((np.load("..\\data\\TrainingData\\train_outputs1.npy"),np.load("..\\data\\TrainingData\\train_outputs2.npy"),np.load("..\\data\\TrainingData\\train_outputs3.npy"),np.load("..\\data\\TrainingData\\train_outputs4.npy"),np.load("..\\data\\TrainingData\\train_outputs5.npy"),np.load("..\\data\\TrainingData\\train_outputs6.npy"),np.load("..\\data\\TrainingData\\train_outputs7.npy"),np.load("..\\data\\TrainingData\\train_outputs8.npy"),np.load("..\\data\\TrainingData\\train_outputs9.npy"),np.load("..\\data\\TrainingData\\train_outputs10.npy"),np.load("..\\data\\TrainingData\\train_outputs11.npy"),np.load("..\\data\\TrainingData\\train_outputs12.npy"),np.load("..\\data\\TrainingData\\train_outputs13.npy"),np.load("..\\data\\TrainingData\\train_outputs14.npy"),np.load("..\\data\\TrainingData\\train_outputs15.npy"),np.load("..\\data\\TrainingData\\train_outputs16.npy"),np.load("..\\data\\TrainingData\\train_outputs17.npy"),np.load("..\\data\\TrainingData\\train_outputs18.npy"),np.load("..\\data\\TrainingData\\train_outputs19.npy"),np.load("..\\data\\TrainingData\\train_outputs20.npy")))
X_test = np.concatenate((np.load("..\\data\\TrainingData\\train_inputs21.npy"),np.load("..\\data\\TrainingData\\train_inputs22.npy"),np.load("..\\data\\TrainingData\\train_inputs23.npy"),np.load("..\\data\\TrainingData\\train_inputs24.npy"),np.load("..\\data\\TrainingData\\train_inputs25.npy"),np.load("..\\data\\TrainingData\\train_inputs26.npy"),np.load("..\\data\\TrainingData\\train_inputs27.npy"),np.load("..\\data\\TrainingData\\train_inputs28.npy"),np.load("..\\data\\TrainingData\\train_inputs29.npy"),np.load("..\\data\\TrainingData\\train_inputs30.npy")))
y_test = np.concatenate((np.load("..\\data\\TrainingData\\train_outputs21.npy"),np.load("..\\data\\TrainingData\\train_outputs22.npy"),np.load("..\\data\\TrainingData\\train_outputs23.npy"),np.load("..\\data\\TrainingData\\train_outputs24.npy"),np.load("..\\data\\TrainingData\\train_outputs25.npy"),np.load("..\\data\\TrainingData\\train_outputs26.npy"),np.load("..\\data\\TrainingData\\train_outputs27.npy"),np.load("..\\data\\TrainingData\\train_outputs28.npy"),np.load("..\\data\\TrainingData\\train_outputs29.npy"),np.load("..\\data\\TrainingData\\train_outputs30.npy")))


def plot_confusion_matrix(cm): 
    # ...(cm, cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation='nearest')
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, np.arange(10))
    plt.yticks(tick_marks, np.arange(10))
    plt.ylabel('True digit')
    plt.xlabel('Predicted digit')
    plt.tight_layout()

# plotting confusion matrix: only call on test set
def c_matrix(clf, X, y):
    dill = cPickle.load(open("..\\pickle\\"+clf+".pkl"))

    # plt.rcdefaults() 
    # do not call plt.xkcd()...
    plt.figure(1)
    y_pred = dill.predict(X)
    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm)
    plt.savefig("..\\figures\\cm_"+clf+".png", dpi=300)


# plotting ROC curve --How do you draw ROC curves for multiclass cases?
'''
plt.figure(2)
y_pred = dill.predict_proba(X_test)
y_score = np.max(y_pred, axis=1)
fpr, tpr, th = roc_curve(y_test, y_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr)
plt.savefig("..\\figures\\roc_logreg_pca.png", dpi=300)
'''

# Plotting n_components v's explained_variance_ for PCA
def pca_var(clf, X, y):
    dill = cPickle.load(open("..\\pickle\\"+clf+".pkl"))
    params = dill.best_estimator_.get_params()
    #pca_params = dict((k, params[k]) for k in ('pca__copy', 'pca__n_components', 'pca__whiten'))
    pca = PCA(copy=params['pca__copy'], n_components=params['pca__n_components'], whiten=params['pca__whiten'])
    pca.fit(X)
    plt.figure()
    plt.clf()
    plt.axes([0.2, .2, 0.7, 0.7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.xlabel('n_components')    
    plt.axis('tight')
    plt.ylabel('explained_variance_')
    
    ###############################################################################
    # Prediction
    #plt.axvline(dill.best_estimator_.named_steps['pca'].n_components,
#                linestyle=':', label='n_components chosen')
    #plt.legend(prop=dict(size=12))
    plt.savefig("..\\figures\\pcavar_"+clf+".png", dpi=300)


# plotting learning curves: only call on training sets
def l_curve(clf, X, y):
    dill = cPickle.load(open("..\\pickle\\"+clf+".pkl"))
    
    cv = ShuffleSplit(len(y), n_iter=5, test_size=0.2, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(dill.best_estimator_, X, y, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig("..\\figures\\lc_"+clf+".png", dpi=300)