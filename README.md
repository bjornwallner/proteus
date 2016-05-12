# Proteus

- Requires python and the machine learning package 'scikit-learn'
- A slightly modified version of DISOPRED3.0 is distributed with this package
- 'scikit-learn' may require additional packages to be installed 
- Also requires the uniref90.fasta database and its associated files in the folder DB
- download the updated-most version of 'uniref90.fasta' file (sequence database) from the web (http://www.ebi.ac.uk/uniprot/database/download.html)
- do a format database on it (formatdb) to generate the associated files 
- And empty DB directory is provided with the installation 

## Installation Notes for scikit-learn
This tutorial requires the following packages:

- Python version 2.6-2.7 or 3.3-3.4
- `numpy` version 1.5 or later: http://www.numpy.org/
- `scipy` version 0.10 or later: http://www.scipy.org/
- `matplotlib` version 1.3 or later: http://matplotlib.org/
- `scikit-learn` version 0.14 or later: http://scikit-learn.org
- `ipython` version 2.0 or later, with notebook support: http://ipython.org
- `seaborn` version 0.5 or later

### Installation of scikit-learn in Ubuntu 14.04

sudo apt-get install python-sklearn  
sudo apt-get update
sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base
pip install --user --install-option="--prefix=" -U scikit-learn

### Installing Proteus

```sh
$ git clone https://github.com/bjornwallner/proteus
$ cd Proteus
$ chmod +x proteus/run_proteus.py
```

## The program has just one inputs

        1. A fasta file containing a single amino acid sequence in fasta format

##### Run Step: 
```sh
$ ./proteus/run_proteus.py <basename.fasta>
```

> EXAMPLE OUTPUT: 
```sh 
$ cat basename.seq.csv
```
> 
          G H M E G K P K M E P A A S S Q A A V E E L R T Q V
          0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 0 0
          0.415 0.458 0.480 0.487 0.510 0.530 0.560 0.555 0.566 0.565 0.552 0.503 0.522 0.549 0.491 0.581 0.616 0.634 
          0.645 0.604 0.559 0.520 0.527 0.459 0.435 0.379
> 

and a graphical representation (.png) of the same (Protean segment prediction score vs. Residue)

![Example output graph] (https://github.com/bjornwallner/proteus/test/H-bond-triple.png)



