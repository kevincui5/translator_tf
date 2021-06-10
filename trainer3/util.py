# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, Input, LSTM, Dense
import unicodedata
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import corpus_bleu

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def split_input_output(path):
  #lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
  lines = pd.read_csv(path, sep='\t', encoding='utf_8')
  sentences_pair = lines.applymap(preprocess_sentence)
  return sentences_pair.iloc[:,0].tolist(), sentences_pair.iloc[:,1].tolist()

def tokenize(text):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  #Updates internal vocabulary based on a list of texts.
  #text can be a list of strings, a generator of strings (for memory-efficiency), or a list of list of strings.
  lang_tokenizer.fit_on_texts(text)
  tokenized_text = lang_tokenizer.texts_to_sequences(text)
  tokenized_text = tf.keras.preprocessing.sequence.pad_sequences(tokenized_text,
                                                         padding='post')
  return tokenized_text, lang_tokenizer

def load_full_dataset(path):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = split_input_output(path)

  input_ndarray, inp_lang_tokenizer = tokenize(inp_lang)
  target_ndarray, targ_tokenizer = tokenize(targ_lang)

  return input_ndarray, target_ndarray, inp_lang_tokenizer, targ_tokenizer

def load_dataset(inp_tokenizer, targ_tokenizer, path):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = split_input_output(path)
  inp_tokenized_text = inp_tokenizer.texts_to_sequences(inp_lang)
  inp_tokenized_text = tf.keras.preprocessing.sequence.pad_sequences(inp_tokenized_text,
                                                         padding='post')
  targ_tokenized_text = targ_tokenizer.texts_to_sequences(targ_lang)
  targ_tokenized_text = tf.keras.preprocessing.sequence.pad_sequences(targ_tokenized_text,
                                                         padding='post')
  return inp_tokenized_text, targ_tokenized_text

def get_dataset_params(data_full_file):
    params = {}
    input_ndarray, target_ndarray, inp_tokenizer, targ_tokenizer = load_full_dataset(data_full_file)   
    # Calculate max_length of the target tensors
    max_length_targ, max_length_inp = target_ndarray.shape[1], input_ndarray.shape[1]  
    vocab_inp_size = len(inp_tokenizer.word_index)+1
    vocab_tar_size = len(targ_tokenizer.word_index)+1
    params['vocab_inp_size'] = vocab_inp_size
    params['vocab_tar_size'] = vocab_tar_size
    params['inp_tokenizer'] = inp_tokenizer
    params['targ_tokenizer'] = targ_tokenizer
    params['max_length_targ'] = max_length_targ
    params['max_length_inp'] = max_length_inp
    return params

def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print(f'{t} ----> {lang.index_word[t]}')
      
def convert_to_sentence(tokenizer, tensor):  
  sentence = ''
  for t in tensor:
    if t != 0:
      sentence += tokenizer.index_word[t] + ' '     
  return sentence

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units_num, Tx, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units_num = enc_units_num
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                               input_length=Tx)
    self.bi_LSTM_last = Bidirectional(LSTM(enc_units_num, return_sequences=True, 
                                      return_state = True))
    self.bi_LSTM_middle = Bidirectional(LSTM(enc_units_num, return_sequences=True))
    self.bi_LSTM_fist = Bidirectional(LSTM(enc_units_num, return_sequences=True,
                                           input_shape = (batch_sz, Tx, embedding_dim)))

  def call(self, x):
    x = self.embedding(x)
    #total 4 layers ob bi-lstm
    #x = self.bi_LSTM_fist(x)
    #enc_layers = 2
    #for _ in range(enc_layers):
    #    x = self.bi_LSTM_middle(x)
    encoder_output, hidden_fwd, cell_fwd, hidden_bwd, cell_bwd = self.bi_LSTM_last(x)
    hidden = Concatenate(axis=-1)([hidden_fwd, hidden_bwd])
    cell = Concatenate(axis=-1)([cell_fwd, cell_bwd])
    output = x
    return output, hidden, cell

  #def initialize_hidden_state(self):
   # return tf.zeros((self.batch_sz, self.enc_units_num))

class Attention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = LSTM(self.dec_units*2, return_state=True)
    #self.fc = Dense(vocab_size, activation='softmax')
    self.fc = Dense(vocab_size)
    # used for attention
    self.attention = Attention(self.dec_units)

  def call(self, x, hidden, cell, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, hidden, cell = self.lstm(x, initial_state = [hidden, cell])

    # output shape == (batch_size * 1, hidden_size)
    #output = tf.reshape(output, (-1, output.shape[2]))

    # x shape == (batch_size, vocab_tar_size)
    x = self.fc(output)

    return x, hidden, cell, attention_weights


@tf.function
def train_step(inp, targ, encoder, decoder, BATCH_SIZE, 
               targ_tokenizer, optimizer):
  loss = 0
  with tf.GradientTape() as tape:
    loss = forward_pass(inp, targ, encoder, decoder, targ.shape[1],
                           BATCH_SIZE, True)
  batch_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def predict_step(inp, max_length_targ, encoder, decoder, batch_sz):
  #cant use encoder.initialize_hidden_state() because for single example need manually initialize shape[0] to 1
  #init_enc_hidden_states = [tf.zeros((1, hidden_units_num))]
  y_pred = forward_pass(inp, None, encoder, decoder, max_length_targ, #prediction doesn't use teacher's forcing, so no need to pass y
                        batch_sz, False)
  return y_pred

def forward_pass(inp, targ, encoder, decoder, Ty, batch_sz, 
                 train_mode):
    '''
    args: train_mode, True means for training, False for prediction or evaluation
    '''
    enc_output, enc_hidden, enc_cell = encoder(inp) #inp shape (batch_size, Tx)
    dec_hidden = enc_hidden # (batch_size,dec_units)  #if encoder and decoder hidden units # diff, need to initialize dec_hidden accordingly
    dec_cell = enc_cell
    dummy = 1
    dec_input = tf.expand_dims([dummy] * batch_sz, 1) #use dummy value to initialize first value of y
    y_pred = []
    loss = 0
    for t in range(Ty):   #targ.shape[1] is Ty
      # passing enc_output to the decoder. prediction shape == (batch_size, vocab_tar_size)
      predictions, dec_hidden, dec_cell, _ = decoder(dec_input, dec_hidden, 
                                                     dec_cell, enc_output) #throws away attension weights
      if(train_mode):
      # Teacher forcing - feeding the target (ground truth) as the next decoder input
        if(t + 1 < Ty):
          loss += loss_function(targ[:, t + 1], predictions)
          dec_input = tf.expand_dims(targ[:, t + 1], 1)  # targ[:, t] is y
    
      else:
        predicted_id = tf.math.argmax(predictions, axis=-1) #predicted_id shape == (batch_size,)
        dec_input = tf.expand_dims(predicted_id, 1)
        y_pred.append(predicted_id) #(Ty, batch_size)
      # the predicted ID is fed back into the model
    y_pred = tf.transpose(y_pred)
   #y_pred = tf.concat(y_pred, axis=0)   #y_pred (list of Ty tensors) shape (Ty, batch_size, 1) -> (Ty*batch_size, 1)

   #y_pred = tf.reshape(y_pred, [batch_sz, max_length_targ, y_pred.shape[1]]) #(batch_size,Ty,vocab_tar_size)
    
    if train_mode:
      return loss
    else:
      return y_pred
  
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()

def bleu_score(inputs_indexed, targets_indexed, y_preds_indexed, params):
  targ_sentences, predicted_sentences = list(), list()
  assert(len(inputs_indexed) == len(targets_indexed) == len(y_preds_indexed))
  for i, input_indexed in enumerate(inputs_indexed): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(params['targ_tokenizer'], targets_indexed[i])
    predicted_sentence = convert_to_sentence(params['targ_tokenizer'], y_preds_indexed[i])
    inp_sentence = (convert_to_sentence(params['inp_tokenizer'], input_indexed))
    targ_sentences.append([targ_sentence.split()[1:-1]])
    predicted_sentences.append(predicted_sentence.split()[1:-1])
    #if i > test_limit:
    #  return
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