# -*- coding: utf-8 -*-
#import numpy as np
import tensorflow as tf
from trainer2.util2 import convert_to_sentence, get_dataset_params, Decoder,
load_dataset
from nltk.translate.bleu_score import corpus_bleu
import os.path
from os import path

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

example_limit = 1700
full_data_path = 'english-german-{}.csv'.format(example_limit)
train_data_path = 'english-german-train-{}.csv'.format(example_limit)
test_data_path = 'english-german-test-{}.csv'.format(example_limit)
saved_model_path = './trained_model2_{}'.format(example_limit)
test_limit = 100

'''
#each input is a single example
def translate(input_indexed, inp_tokenizer, targ_tokenizer, max_length_inp, 
              max_length_targ, encoder, decoder):

  attention_plot = np.zeros((max_length_targ, max_length_inp))
  input_tensor = tf.convert_to_tensor(input_indexed)
  #cant use encoder.initialize_hidden_state() because for single example need manually initialize shape[0] to 1
  init_enc_hidden_states = [tf.zeros((1, hidden_units_num))]

  enc_out, enc_hidden_states = encoder(input_tensor, init_enc_hidden_states)

  init_dec_hidden_states = enc_hidden_states
  dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)  #no need to multiply batch size because it's single example
  predicted_sentence = '<start> '
  dec_hidden = init_dec_hidden_states
  for t in range(max_length_targ):
    prediction, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    #attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.math.argmax(prediction[0]).numpy()
    predicted_word = targ_tokenizer.index_word[predicted_id]
    predicted_sentence += predicted_word + ' '

    if predicted_word == '<end>':
      return predicted_sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return predicted_sentence, attention_plot
'''
def bleu_score(inputs_indexed, targets_indexed, y_preds_indexed, params):
  targ_sentences, predicted_sentences = list(), list()
  for i, input_indexed in enumerate(inputs_indexed): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(params['targ_tokenizer'], targets_indexed[i])
    predicted_sentence = convert_to_sentence(params['targ_tokenizer'], y_preds_indexed[i])
    inp_sentence = (convert_to_sentence(params['inp_tokenizer'], input_indexed))
    targ_sentences.append([targ_sentence.split()[1:-1]])
    predicted_sentences.append(predicted_sentence.split()[1:-1])
    if i > test_limit:
      return
    if i > 5: #show a preview
      continue;
    print('src=[%s], target=[%s], predicted=[%s]' % (inp_sentence, 
          targ_sentence, predicted_sentence))
    #attention_plot = attention_plot[:len(predicted_sentence.split(' ')),
    #                              :len(targ_sentence.split(' '))]
    #plot_attention(attention_plot, targ_sentence.split(' '), predicted_sentence.split(' ')) 

  print('BLEU-1: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25)))

  
def evaluate():
  params = get_dataset_params(full_data_path)  
  inp_tokenizer = params['inp_tokenizer']
  targ_tokenizer = params['targ_tokenizer']
  train_inputs_indexed, train_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, train_data_path)
  #optimizer = tf.keras.optimizers.Adam()
  #checkpoint = tf.train.Checkpoint(optimizer=optimizer,
  #                               encoder=encoder,
  #                               decoder=decoder)  
  # restoring the latest checkpoint in checkpoint_dir
  #checkpoint.restore(tf.train.latest_checkpoint(saved_model_path))  
  BUFFER_SIZE = len(train_inputs_indexed)
  decoder = Decoder(targ_tokenizer, params['vocab_inp_size'], params['vocab_tar_size'], embedding_dim, hidden_units_num, BATCH_SIZE)
  dataset_train = tf.data.Dataset.from_tensor_slices((train_inputs_indexed, 
                                                  train_targets_indexed)).shuffle(BUFFER_SIZE)
  dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
  #test=decoder(train_inputs_indexed)
  y_preds_indexed = decoder.predict(dataset_train)

  bleu_score(train_inputs_indexed, train_targets_indexed, y_preds_indexed, params)
  test_inputs_indexed, test_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)  
  dataset_test = tf.data.Dataset.from_tensor_slices((test_inputs_indexed, 
                                                  test_targets_indexed)).shuffle(BUFFER_SIZE)
  dataset_test = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
 
  y_preds_indexed = decoder.predict(dataset_test)
  bleu_score(test_inputs_indexed, test_targets_indexed, y_preds_indexed, params)
evaluate()
 