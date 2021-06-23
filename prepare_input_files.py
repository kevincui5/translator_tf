# -*- coding: utf-8 -*-

import string
import re
from unicodedata import normalize
from numpy import array
import numpy as np
from numpy.random import rand
from numpy.random import shuffle

RAW_FILE = 'deu.txt'
example_limit = 180000
FULL_DATA_FILE = 'english-german-{}.csv'.format(example_limit)
TRAIN_FILE = 'english-german-train-{}.csv'.format(example_limit)
TEST_FILE = 'english-german-test-{}.csv'.format(example_limit)
VALID_FILE = 'english-german-valid-{}.csv'.format(example_limit)
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
filename = RAW_FILE
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
pairs = pairs[:example_limit]
# save clean pairs to file
save_clean_data(pairs, FULL_DATA_FILE)
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
train_ratio = 0.90
val_test_ratio = 0.5
train_size = int(train_ratio*n_sentences)
valid_size = int((1 - train_ratio) * val_test_ratio * n_sentences)
test_start = train_size + valid_size
train, valid, test = dataset[:train_size], dataset[train_size:test_start], dataset[test_start:]
# save
#save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, TRAIN_FILE)
save_clean_data(test, TEST_FILE)
save_clean_data(valid, VALID_FILE)