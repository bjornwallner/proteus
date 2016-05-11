#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:08:46 2015

@author: freso388
"""

from sys import argv, exit
from sklearn.externals import joblib
#from Bio import SeqIO
#from Bio.Seq import Seq
#from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import generic_protein
from os import makedirs, chdir
from os.path import exists, split, abspath, basename,dirname
from subprocess import call
from hashlib import md5

from feature_functions import make_features
from method_functions import vis_prediction
from method_functions import make_csv
from method_functions import send_mail
import commands


#Modify
install_path=dirname(dirname(abspath(argv[0]))) + "/"
print install_path
# Read sequence and email from files
if len(argv) != 2:
    print "Please specify sequence" #and email files."
    exit()
fasta_file = abspath(argv[1])
job_folder = dirname(fasta_file)
name = basename(fasta_file)

sequence=""
for line in file(fasta_file):
    if ">" not in line :
        sequence=sequence+line.strip()

if job_folder != "":
    chdir(job_folder)

#    f.write(email)
#    f.write(sequence)

# Generate files (from Disopred and such)
#disopred_file = "/local/www/services/proteus/programs/DISOPRED/run_disopred.pl"
disopred_file = install_path + "/DISOPRED/run_disopred.pl"
cmd = disopred_file + " " + fasta_file
print cmd
#exit(1)
commands.getstatusoutput(cmd)

#call([disopred_file, fasta_file])

# Generate features
window_size = 15
X = make_features([name], window_size)

# Run through predictor
# THIS PATH MIGHT BE WRONG!
#clf = joblib.load("/local/www/services/proteus/programs/classifier/classifier.pkl")
# COMPARE TO THE ONE BELOW!
clf = joblib.load(install_path + "/proteus/classifier/classifier.pkl")
prediction = clf.predict(X)
probabilities = clf.predict_proba(X)
probabilities = [prob[1] for prob in probabilities]

# Save the result (somehow)
vis_prediction(name, prediction, probabilities)
make_csv(name, sequence, prediction, probabilities)
#send_mail([email], files=[name + '.csv', 'vis_' + name + '.png'],
#          smtpserver="smtp.gmail.com:587")
