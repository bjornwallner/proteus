#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:08:46 2015

@author: freso388
"""

from sys import argv, exit
from sklearn.externals import joblib
from os.path import exists, split, abspath, basename,dirname
#import pickle
import cPickle as pickle
import gzip

install_path=dirname(dirname(abspath(argv[0]))) + "/"
print install_path
clf = joblib.load(install_path + "/proteus/classifier/classifier.pkl")

outfile=install_path + "/proteus/classifier/classifier.cpickled.pkl.gz"
pickle.dump(clf,gzip.open(outfile,'w'), pickle.HIGHEST_PROTOCOL)
#with open(install_path + "/proteus/classifier/classifier.updated.cpickle", 'wb') as handle:
#        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
