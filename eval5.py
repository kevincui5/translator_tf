# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer5.util import get_dataset_params, load_dataset, Decoder, Encoder, convert_to_sentence
from nltk.translate.bleu_score import corpus_bleu
from os import path

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

example_limit = 1700
full_data_path = 'english-german-{}.csv'.format(example_limit)
train_data_path = 'english-german-train-{}.csv'.format(example_limit)
test_data_path = 'english-german-test-{}.csv'.format(example_limit)
saved_model_path = './trained_model5_{}'.format(example_limit)
test_limit = 500
#each input is a single example
def translate(input_indexed, inp_tokenizer, targ_tokenizer, max_length_inp, 
              max_length_targ, encoder, decoder):

  attention_plot = np.zeros((max_length_targ, max_length_inp))
  input_tensor = tf.convert_to_tensor(input_indexed)
  #cant use encoder.initialize_hidden_state() because for single example need manually initialize shape[0] to 1
  #init_enc_hidden_states = [tf.zeros((1, hidden_units_num))]

  enc_out, enc_hidden_states, enc_cell = encoder(input_tensor)

  #init_dec_hidden_states = enc_hidden_states
  #init_dec_cell = enc_cell
  dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)  #no need to multiply batch size because it's single example
  predicted_sentence = '<start> '
  #dec_hidden = init_dec_hidden_states
  #dec_cell = init_dec_cell
  for t in range(max_length_targ):
    prediction, _ = decoder(dec_input)
    # storing the attention weights to plot later on
    #attention_weights = tf.reshape(attention_weights, (-1, ))
    #attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.math.argmax(prediction[0]).numpy()
    predicted_word = targ_tokenizer.index_word[predicted_id]
    predicted_sentence += predicted_word + ' '

    if predicted_word == '<end>':
      return predicted_sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return predicted_sentence, attention_plot

def predict(inputs_indexed, targets_indexed, params, encoder, decoder):
  targ_sentences, predicted_sentences = list(), list()
  for i, in_targ_indexed in enumerate(zip(inputs_indexed, targets_indexed)): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(params['targ_tokenizer'], in_targ_indexed[1])
    inp_sentence = (convert_to_sentence(params['inp_tokenizer'], in_targ_indexed[0]))
    targ_sentences.append([targ_sentence.split()[1:-1]])
    predicted_sentence, attention_plot = translate([in_targ_indexed[0]], 
                                                   params['inp_tokenizer'], params['targ_tokenizer'], 
                                                   params['max_length_inp'], params['max_length_targ'], encoder, decoder)
    predicted_sentences.append(predicted_sentence.split()[1:-1])
    if i > test_limit:
      break
    if i > 5: #show a preview
      continue
    print('src=[%s], target=[%s], predicted=[%s]' % (inp_sentence, 
          targ_sentence, predicted_sentence))
    attention_plot = attention_plot[:len(predicted_sentence.split(' ')),
                                  :len(targ_sentence.split(' '))]
    #plot_attention(attention_plot, targ_sentence.split(' '), predicted_sentence.split(' ')) 

  print('BLEU-1: %f' % corpus_bleu(targ_sentences, predicted_sentences, 
                                   weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(targ_sentences, predicted_sentences, 
                                   weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(targ_sentences, predicted_sentences, 
                                   weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(targ_sentences, predicted_sentences, 
                                   weights=(0.25, 0.25, 0.25, 0.25)))

  
def evaluate():
  params = get_dataset_params(full_data_path)  
  inp_tokenizer = params['inp_tokenizer']
  targ_tokenizer = params['targ_tokenizer']
  max_length_inp = params['max_length_inp']
  train_inputs_indexed, train_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, train_data_path)
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    encoder = Encoder(params['vocab_inp_size'], embedding_dim, hidden_units_num, 
                      max_length_inp, BATCH_SIZE)
    decoder = Decoder(encoder, params['vocab_tar_size'] , embedding_dim, hidden_units_num, 
                      BATCH_SIZE)  
    optimizer = tf.keras.optimizers.Adam()
  checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)  
  # restoring the latest checkpoint in checkpoint_dir
  #checkpoint.restore(tf.train.latest_checkpoint(saved_model_path)).expect_partial()  
  checkpoint.restore(tf.train.latest_checkpoint(saved_model_path))  
  predict(train_inputs_indexed, train_targets_indexed, params, encoder, decoder)
  test_inputs_indexed, test_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)  
  predict(test_inputs_indexed, test_targets_indexed, params, encoder, decoder)
evaluate()
 