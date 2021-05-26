# -*- coding: utf-8 -*-

import string
import re
#from pickle import dump
from unicodedata import normalize
from numpy import array
import numpy as np
from numpy.random import rand
from numpy.random import shuffle
from trainer.util import *

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# split a loaded document into sentences
def to_pairs(doc):
	lines = doc.strip().split('\n')
	pairs = [line.split('\t') for line in  lines]
	return pairs


def save_clean_data(sentences, filename):
	#dump(sentences, open(filename, 'wb'))
    np.savetxt(filename, sentences, delimiter="\t", fmt='%s', encoding='utf-8')
    print('Saved: %s' % filename)
    
# load dataset
filename = 'deu-3253.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# save clean pairs to file
save_clean_data(pairs, 'full-data-1700.csv')
# spot check
for i in range(10):
	print('[%s] => [%s]' % (pairs[i][0], pairs[i][1]))

# load dataset
#raw_dataset = load_clean_sentences('english-german-full.csv')
raw_dataset = pairs
n_sentences = len(raw_dataset)
# reduce dataset size for testing purpose
#n_sentences = 1000
dataset = raw_dataset[:][ :]
#dataset = raw_dataset[:n_sentences, :]
# random shuffle (optional)
shuffle(dataset)
# split into train/test
ratio = 0.95
train_size = int(ratio*n_sentences)
train, test = dataset[:train_size], dataset[train_size:]
# save
#save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'train-1700.csv')
save_clean_data(test, 'test-1700.csv')