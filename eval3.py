# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer3.util import get_dataset_params, Decoder, load_dataset, bleu_score,Encoder, predict_step
import os.path
from os import path

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

example_limit = 10000
full_data_path = 'english-german-{}.csv'.format(example_limit)
train_data_path = 'english-german-train-{}.csv'.format(example_limit)
test_data_path = 'english-german-test-{}.csv'.format(example_limit)
saved_model_path = './trained_model3_{}'.format(example_limit)
test_limit = BATCH_SIZE*3
    

params = get_dataset_params(full_data_path)  
inp_tokenizer = params['inp_tokenizer']
targ_tokenizer = params['targ_tokenizer']
max_length_inp = params['max_length_inp']
max_length_targ = params['max_length_targ']

train_inputs_indexed, train_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, train_data_path) 
BUFFER_SIZE = len(train_inputs_indexed)

encoder = Encoder(params['vocab_inp_size'], embedding_dim, hidden_units_num, 
                    max_length_inp, BATCH_SIZE)
decoder = Decoder(params['vocab_tar_size'], embedding_dim, hidden_units_num, 
                  BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  
  #restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(saved_model_path)) 
  #check point couldn't restore optimezer, loss, etc, so manually set them
#decoder.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
#                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'])

dataset_train = tf.data.Dataset.from_tensor_slices(train_inputs_indexed[:test_limit]).shuffle(BUFFER_SIZE)
dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
y_preds_indexed = []
for step, x_batch_train in enumerate(dataset_train):

    y_pred_batch_indexed = predict_step(x_batch_train, max_length_targ, encoder, 
                      decoder, BATCH_SIZE) #y one hot encoded, shape (m,Ty,vacab_targ_size), type is numpy array
    y_preds_indexed.append(y_pred_batch_indexed)

y_preds_indexed = tf.concat(y_preds_indexed, 0)
y_preds_indexed = y_preds_indexed.numpy()

bleu_score(train_inputs_indexed[:test_limit], train_targets_indexed[:test_limit], 
           y_preds_indexed, params)
test_inputs_indexed, test_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)  
dataset_test = tf.data.Dataset.from_tensor_slices(test_inputs_indexed[:test_limit]).shuffle(BUFFER_SIZE)
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)
 
#y_preds_indexed = predict_step(dataset_test, max_length_targ, encoder, 
#                      decoder, BATCH_SIZE) 
#bleu_score(test_inputs_indexed[:test_limit], test_targets_indexed[:test_limit], 
#           y_preds_indexed, params)

 