# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer.util import *
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

#each input is a single example
def translate(input_indexed, inp_tokenizer, targ_tokenizer, max_length_inp, max_length_targ, encoder, decoder):

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

  
def evaluate(data_full):
  examples_limit = 100000 #need to match that of training or the tokenizers will be diff
  test_limit = 100
  inputs_indexed, targets_indexed, inp_tokenizer, targ_tokenizer = load_dataset(data_full, examples_limit)
  max_length_targ = inputs_indexed.shape[1]
  max_length_inp = targets_indexed.shape[1]  
  vocab_inp_size = len(inp_tokenizer.word_index)+1
  vocab_tar_size = len(targ_tokenizer.word_index)+1  
  encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num, BATCH_SIZE)
  decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num, BATCH_SIZE)  
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
    if i > test_limit:
      break;
    if i > 10: #show a preview
      continue;
    #print(in_targ_indexed[0])  
    print('src=[%s], target=[%s], predicted=[%s]' % (inp_sentence, 
          targ_sentence, predicted_sentence))

  print('BLEU-1: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25)))
    #print(predicted_sentence) 
    #print(input_processed_sentence)
  #input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(params['input_tensor'], params['target_tensor'], test_size=0.2)    

  #translate(sentence, params)
  #result, sentence, attention_plot = translate(sentence, params)

  #print('Input:', input_tensor_train)
  #print('Predicted translation:', result)

  #attention_plot = attention_plot[:len(result.split(' ')),
   #                               :len(sentence.split(' '))]
  #plot_attention(attention_plot, sentence.split(' '), result.split(' ')) 

evaluate('deu.txt')
 