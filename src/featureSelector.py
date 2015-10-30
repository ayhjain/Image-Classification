import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
from nltk.stem import WordNetLemmatizer
import string
import pickle

import sys, os, codecs
reload(sys)
sys.setdefaultencoding('utf-8')


stoplist = stopwords.words('english')
stoplist.append('__eos__',)
stoplist.append('__EOS__')
stoplist.append('eos')
stoplist.append('EOS')


_use_TFIDF_ = True
no_of_features = 20000


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        doc = str(doc)
        s = "".join(doc.split("__EOS__"))
        doc = s.translate(None, string.punctuation)
        tokens = doc.word_tokenize(doc)
        bi = list(p1+" "+p2 for p1,p2 in nltk.bigrams(tokens))
        tokens.extend(bi)
        return [self.wnl.lemmatize(t) for t in tokens]            


if _use_TFIDF_ :
    #vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, stop_words=stoplist, max_features=no_of_features, tokenizer=LemmaTokenizer())
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=1, stop_words=stoplist, max_features=no_of_features)
    func_tokenizer =vectorizer.build_tokenizer()

'''
I was using two functions earlier for tokenization and data preprocessing.
Later implemented the LemmaTokenizer class for this. 
'''

def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)
    

def get_tokens(s):
#   Tokenize into words in sentences. Returns list of strs
    retval = []
    sents = sent_tokenize(s)
    for sent in sents:
        tokens = word_tokenize(sent)
        
        for word in tokens:
            word = word.lower()
            if (ispunct(word)): 
                continue
            if (word in stoplist): 
                continue
            elif (ispunct(word[-1])): 
                word = word[:-1] # removing puntuation from last place
            retval.append(word)
    
    return retval

def extract_featureMatrix(strings, classes, no_of_features,replace, lemmatize, lowercase, entriesToProcess):
    '''
    Extract features from text file f into a feature vector.
    '''
    
    if _use_TFIDF_ : 
        featurelist = get_tfidf_features(strings, classes, no_of_features, replace, lemmatize, lowercase)
    else :
        featurelist = get_bagofwords(strings, classes, m, l, lemmatize, lowercase)
    
    
    noOfFeatures = len(featurelist) # Total no. of features
    featureMatrix = np.zeros(shape=[entriesToProcess, noOfFeatures], dtype=float16)
    
    i = 0
    for str in strings : 
        
        '''
        Using TFIDF approach for initial feature selection. Hardcoding this check just for performance optimizations.

        if _use_TFIDF_ : 
            tokens = list(s.lower() for s in func_tokenizer(str) if s.lower() not in stoplist)
        
        else : 
            bigram=[]
            tokens = get_tokens(str)
            pair = list(p1+" "+p2 for p1,p2 in nltk.bigrams(tokens))
            tokens.extend(pair)
        '''
        
        dict_token={}
        for s in func_tokenizer(str):
            if (s in featurelist):
                if (s in dict_token):
                    dict_token[s] += 1
                else:
                    dict_token[s] = 1
        
        j=0
        if i%100 == 0:
            print ("Processing ",(i+1),"th entry.")
        
        for inst in featurelist : 
            if inst in dict_token:
                featureMatrix[i,j] = dict_token[inst]
            j+=1
        i+=1
    
    return featureMatrix



def extract_feature_Test(strings, entriesToProcess):
    '''
    Extract features from text file f into a feature vector.
    '''
    # This file will always be there for Test feature extraction!!!

    os.chdir("Data")
    featurelist = np.load("features.npy")
    
    
    noOfFeatures = len(featurelist) # Total no. of features
    featureMatrix = np.zeros(shape=[entriesToProcess, noOfFeatures], dtype=float16)
    
    i = 0
    for str in strings : 
        
        '''
        Using TFIDF approach for initial feature selection. Hardcoding this check just for performance optimizations.

        if _use_TFIDF_ : 
            tokens = list(s.lower() for s in func_tokenizer(str) if s.lower() not in stoplist)
        
        else : 
            bigram=[]
            tokens = get_tokens(str)
            pair = list(p1+" "+p2 for p1,p2 in nltk.bigrams(tokens))
            tokens.extend(pair)
        '''
        
        dict_token={}
        for s in func_tokenizer(str):
            if (s in featurelist):
                if (s in dict_token):
                    dict_token[s] += 1
                else:
                    dict_token[s] = 1
        j=0
        if i%100 == 0:
            print "Processing ",(i+1),"th test entry."
        
        for inst in featurelist : 
            if inst in dict_token:
                featureMatrix[i,j] = dict_token[inst]
            j+=1
        i+=1

    os.chdir("..")
    return featureMatrix


def get_tfidf_features(strings, classes, no_of_features, replace, lemmatize, lowercase):
#   Returns the bag of most common n words for each category
    directory =os.getcwd()
    os.chdir("Data")

    if (not replace) and os.path.isfile("features.npy") : 
        features = np.load("features.npy")
        print ("List of features picked from memory.")
        #for i in range(len(features)):
        #    print features[i]
        
    else : 
        i=0
        bow=["","","",""]
        for y in classes:
            bow[int(y)] += strings[i]
            i+=1
        
        vectorizer.fit_transform(bow)
        featureList = list(vectorizer.get_feature_names())
        idf = vectorizer.idf_

        #print ("Feature list with its TfIdf scores- ")
        #for i in zip(vectorizer.get_feature_names(), idf): print i

        dictionary = dict(zip(vectorizer.get_feature_names(), idf))
        pickle.dump(dictionary, open('feature_idf_score.txt', 'wb'))

        print ("New list of features created.")
        features = np.array(featureList)
        np.save("features", features)
    
    os.chdir("..")
    return features
    
''' 
def get_bagofwords(strings, classes, m, l, lemmatize, lowercase):
#   Returns the bag of most common n words for eah category
    

    
    featurelist=[]
    
    bow = []
    bow.append([])
    bow.append([])
    bow.append([])
    bow.append([])
    i = 0
    
    for str in strings : 
        tokens = get_tokens(str)
        
        unigram = tokens
        bigram = nltk.bigrams(tokens)
        category = int(classes[i])
        
        # Adding Unigrams to bag of words
        for inst in unigram : bow[category].append(inst)
        # Adding Bigrams to bag of words
        for inst in bigram : bow[category].append(inst)
        i += 1
    
    
    # Printing list of all n-grams for each prediction class
    #for j in range(3):
    #   print "Prediction Class : ",j, '\n', bow[j], '\n'
        
    # Picking best n-grams for each category and converting them to features
    dict_ngram = [{},{},{},{}]
    for j in range(4):
        dict_ngram[j] = Counter(bow[j])
    
    tf_idf = [{},{},{},{}]
    
    for j in range(4):
        sum=0
        for key, val in dict_ngram[j] : 
            s = sum(dict_ngram[i][key] for i in range(4) if i!=j)
            tf_idf[j][key] = val / dict_ngram[i][key] for i 
            
        for key,val in dict_ngram.most_common()[-l-10-1:-10:1] : 
            featurelist.append(key)
    
    return featurelist 
    '''
