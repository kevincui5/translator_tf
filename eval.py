# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer.util import *
from nltk.translate.bleu_score import corpus_bleu

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

test_data_path = 'test-3253.csv'
full_data_path = 'full-data-3253.csv'
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

  
def evaluate():
  params = get_dataset_params(full_data_path)  
  inp_tokenizer = params['inp_tokenizer']
  targ_tokenizer = params['targ_tokenizer']
  inputs_indexed, targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)
  max_length_targ = inputs_indexed.shape[1]
  max_length_inp = targets_indexed.shape[1]  
  encoder = Encoder(params['vocab_inp_size'], embedding_dim, hidden_units_num, BATCH_SIZE)
  decoder = Decoder(params['vocab_tar_size'] , embedding_dim, hidden_units_num, BATCH_SIZE)  
  optimizer = tf.keras.optimizers.Adam()
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  
  # restoring the latest checkpoint in checkpoint_dir
  checkpoint.restore(tf.train.latest_checkpoint("training_checkpoints"))  
  targ_sentences, predicted_sentences = list(), list()
  for i, in_targ_indexed in enumerate(zip(inputs_indexed, targets_indexed)): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(targ_tokenizer, in_targ_indexed[1])
    inp_sentence = (convert_to_sentence(inp_tokenizer, in_targ_indexed[0]))
    targ_sentences.append([targ_sentence.split()[1:-1]])
    predicted_sentence, attention_plot = translate([in_targ_indexed[0]], inp_tokenizer, targ_tokenizer, max_length_inp, max_length_targ, encoder, decoder)
    predicted_sentences.append(predicted_sentence.split()[1:-1])
    if i > 5: #show a preview
      continue;
    print('src=[%s], target=[%s], predicted=[%s]' % (inp_sentence, 
          targ_sentence, predicted_sentence))
    attention_plot = attention_plot[:len(predicted_sentence.split(' ')),
                                  :len(targ_sentence.split(' '))]
    plot_attention(attention_plot, targ_sentence.split(' '), predicted_sentence.split(' ')) 
  print('BLEU-1: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25)))

  #translate(sentence, params)
  #result, sentence, attention_plot = translate(sentence, params)
evaluate()
 