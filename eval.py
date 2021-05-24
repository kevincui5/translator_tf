# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from trainer.util import *
from sklearn.model_selection import train_test_split

hidden_units_num = 1024
BATCH_SIZE = 64
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
  predicted_sentence = ''
  dec_hidden = init_dec_hidden_states
  for t in range(max_length_targ):
    prediction, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    #attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.math.argmax(prediction[0]).numpy()
    print(predicted_id)
    predicted_word = targ_tokenizer.index_word[predicted_id]
    predicted_sentence += predicted_word + ' '

    if predicted_word == '<end>':
      return predicted_sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return predicted_sentence, attention_plot

  
def evaluate(data_full):
  examples_limit = 30
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

  for i, input_indexed in enumerate(inputs_indexed): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    print(convert(inp_tokenizer, input_indexed))
    predicted_sentence, attention_plot = translate([input_indexed], inp_tokenizer, targ_tokenizer, max_length_inp, max_length_targ, encoder, decoder)
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

evaluate('spa-6624.txt')
 