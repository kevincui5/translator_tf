# -*- coding: utf-8 -*-
import tensorflow as tf
import unicodedata
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
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
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, BATCH_SIZE, 
               targ_tokenizer, optimizer):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden
    #insert a length 1 axis at 2nd dimension, meaning use a start word id as each training example
    dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1) 

    # Teacher forcing - feeding the target (ground truth) as the next decoder input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output) #throws away attension weights

      loss += loss_function(targ[:, t], predictions) #take parameters of ground truth and prediction

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)  # targ[:, t] is y

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

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
