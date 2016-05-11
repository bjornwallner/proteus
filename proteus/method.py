# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:33:13 2015

@author: fredrik
"""


def proteus(sequence, send_to):

    from sklearn.externals import joblib
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Alphabet import generic_protein
    from os import makedirs, chdir
    from os.path import exists
    from subprocess import call
    from hashlib import md5

    from feature_functions import make_features
    from method_functions import vis_prediction
    from method_functions import make_csv
    from method_functions import send_mail

    # Navigate to Proteus program folder
    #chdir("/local/www/services/proteus/") #/home/freso388/nsc/programs/proteus")
    disopred_file = "/local/www/services/proteus/programs/DISOPRED/run_disopred.pl"
    window_size = 15

    # Make name from sequence and create job folder
    m = md5()
    m.update(sequence)
    name = m.hexdigest()
    directory = '/local/www/services/proteus/jobs/' + name + '/'
    if not exists(directory):
        makedirs(directory)
    chdir(directory)
    fasta_file = name + ".fasta"
    input_file = name + ".inp"
    with open(input_file, "w") as f:
        f.write(send_to)
        f.write(sequence)

    # Make record from sequence
    record = SeqRecord(Seq(sequence, generic_protein), id=name)

    # Generate fasta file from sequence
    SeqIO.write(record, fasta_file, "fasta")

    # Generate files (from Disopred and such)
    call([disopred_file, fasta_file])

    # Generate features
    X = make_features([name], window_size)

    # Run through predictor
    clf = joblib.load("/local/www/services/proteus/programs/classifier/classifier.pkl")
    prediction = clf.predict(X)
    probabilities = clf.predict_proba(X)
    probabilities = [prob[1] for prob in probabilities]

    # Save the result (somehow)
    vis_prediction(name, prediction, probabilities)
    make_csv(name, sequence, prediction, probabilities)
    send_mail([send_to], files=[name + '.csv', 'vis_' + name + '.png'],
              smtpserver="smtp.gmail.com:587")

    return "A-OK!"
