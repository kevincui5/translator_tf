# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer2.util2 import convert_to_sentence, get_dataset_params, Decoder, load_dataset, bleu_score,Encoder
from nltk.translate.bleu_score import corpus_bleu
import os.path
from os import path

hidden_units_num = 1024
BATCH_SIZE = 64
embedding_dim = 256

example_limit = 1700
full_data_path = 'english-german-{}.csv'.format(example_limit)
train_data_path = 'english-german-train-{}.csv'.format(example_limit)
test_data_path = 'english-german-test-{}.csv'.format(example_limit)
saved_model_path = './trained_model2_{}'.format(example_limit)
test_limit = BATCH_SIZE



  
def evaluate():
  params = get_dataset_params(full_data_path)  
  inp_tokenizer = params['inp_tokenizer']
  targ_tokenizer = params['targ_tokenizer']
  train_inputs_indexed, train_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, train_data_path)
 
  BUFFER_SIZE = len(train_inputs_indexed)
  #attention = BahdanauAttention(hidden_units_num)
  encoder = Encoder(params['vocab_inp_size'], embedding_dim, hidden_units_num)
  #encoder.load_weights(saved_model_path)
  decoder = Decoder(encoder, targ_tokenizer, params['vocab_inp_size'], params['vocab_tar_size'], embedding_dim, hidden_units_num, BATCH_SIZE)
  #decoder.load_weights(saved_model_path)
  optimizer = tf.keras.optimizers.Adam()
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  
  #restoring the latest checkpoint in checkpoint_dir
  checkpoint.restore(tf.train.latest_checkpoint(saved_model_path)) 
  #check point couldn't restore optimezer, loss, etc, so manually set them
  decoder.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'])

  dataset_train = tf.data.Dataset.from_tensor_slices((train_inputs_indexed[:test_limit], 
                                                  train_targets_indexed[:test_limit])).shuffle(BUFFER_SIZE)
  dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True) #((batch_size,Tx),(batch_size,Ty))
  #dataset_train = dataset_train.batch(BATCH_SIZE)

  #test=decoder(train_inputs_indexed)
  print(dataset_train)
  y_preds_oh = decoder.predict(dataset_train) #y one hot encoded, shape (m,Ty,vacab_targ_size), type is numpy array
  #y_preds_indexed = tf.math.argmax(y_preds_oh, axis=-1)

  y_preds_indexed = np.argmax(y_preds_oh, axis=-1)
  bleu_score(train_inputs_indexed[:test_limit], train_targets_indexed[:test_limit], y_preds_indexed, params)
  test_inputs_indexed, test_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)  
  dataset_test = tf.data.Dataset.from_tensor_slices((test_inputs_indexed[:test_limit], 
                                                  test_targets_indexed[:test_limit])).shuffle(BUFFER_SIZE)
  dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)
 
  y_preds_oh = decoder.predict(dataset_test)
  y_preds_indexed = np.argmax(y_preds_oh, axis=-1)
  bleu_score(test_inputs_indexed[:test_limit], test_targets_indexed[:test_limit], y_preds_indexed, params)
evaluate()
 