# -*- coding: utf-8 -*-
import tensorflow as tf
import unicodedata
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorflow.keras.layers import Bidirectional, LSTM, Concatenate, Dense
import os.path
from nltk.translate.bleu_score import corpus_bleu
import numpy as np

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

def convert_to_sentence_tagged(tokenizer, tensor):
  sentence = ''
  for t in tensor:
    if t != 0:
      if t==1:
        sentence += str(t)
      sentence += tokenizer.index_word[t] + ' '  
  return sentence

class Encoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, units, 
               name="Encoder", **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    #notice no input_shape parameter is neccesary to pass in the layer init fun
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.bi_LSTM_last = Bidirectional(LSTM(units, return_sequences=True, 
                                      return_state = True))
    self.bi_LSTM_middle = Bidirectional(LSTM(units, return_sequences=True))
    self.bi_LSTM_fist = Bidirectional(LSTM(units, return_sequences=True))
    
  def call(self, x):
    x = self.embedding(x)
    #total 4 layers ob bi-lstm
    x = self.bi_LSTM_fist(x)
    enc_layers = 2
    for _ in range(enc_layers):
        x = self.bi_LSTM_middle(x)
    encoder_output, hidden_fwd, cell_fwd, hidden_bwd, cell_bwd = self.bi_LSTM_last(x)
    hidden = Concatenate(axis=-1)([hidden_fwd, hidden_bwd])
    cell = Concatenate(axis=-1)([cell_fwd, cell_bwd])
    output = x
    return output, hidden, cell

  def get_config(self):
    config = super(Encoder, self).get_config()
    config.update({"units": self.units})
    return config
   
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units, name="BahdanauAttention", **kwargs):
    super(BahdanauAttention, self).__init__(**kwargs)
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

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

  def get_config(self):
    config = super(BahdanauAttention, self).get_config()
    config.update({"units": self.units})
    return config

class Decoder(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embedding_dim, dec_units, 
               name="Decoder", **kwargs):
    super(Decoder, self).__init__(**kwargs)
    self.units = dec_units * 2
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.lstm = LSTM(self.units, return_state=True)
    self.fc = Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(dec_units)

  def call(self, x, hidden, cell, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    #output and hidden are the same here because only return_state is set true
    output, hidden, cell = self.lstm(x, initial_state = [hidden, cell])
    # output shape == (batch_size * 1, hidden_size)
    #output = tf.reshape(output, (-1, output.shape[2]))

    # x shape == (batch_size, vocab_tar_size)
    x = self.fc(output)

    return x, hidden, cell, attention_weights

  def get_config(self):
    config = super(Decoder, self).get_config()
    config.update({"units": self.units})
    return config

class Translator(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, 
                 hidden_units_num, name="Translator", 
                 **kwargs):
      super(Translator, self).__init__(name=name, **kwargs)
      self.encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num)
      self.decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num)
      self.loss = 0
      
    def call(self, data):
      inp, targ = data
      return self.forward_pass(inp, targ)
    
    def train_step(self, data):
      inp, targ = data
      loss = 0
      with tf.GradientTape() as tape:
        enc_output, enc_hidden, enc_cell = self.encoder(inp)    
        dec_hidden = enc_hidden
        dec_cell = enc_cell
        dec_input = tf.expand_dims(targ[:,0], 1) 
        # Teacher forcing - feeding the target (ground truth) as the next decoder input
        for t in range(1, targ.shape[1]):
          # passing enc_output to the decoder. predictions shape == (batch_size, vocab_tar_size)
          predictions, dec_hidden, dec_cell, _ = self.decoder(dec_input, dec_hidden,
                                                         dec_cell, enc_output) #throws away attension weights
          #targ shape == (batch_size, ty)
          loss += loss_function(targ[:, t], predictions) #take parameters of ground truth and prediction
          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)  # targ[:, t] is y
          self.compiled_metrics.update_state(targ[:, t], predictions)

      trainable_vars = self.trainable_variables    
      gradients = tape.gradient(loss, trainable_vars)    
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      # Update metrics (includes the metric that tracks the loss)
      #self.compiled_metrics.update_state(targ, y_pred)
      # Return a dict mapping metric names to current value
      return {m.name: m.result() for m in self.metrics}  
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        self(data, training=False)
        # Updates the metrics tracking the loss
        #self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        #self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
  
    def predict_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(data, training=False)  
        return y_pred
      
    #change fun name
    def forward_pass(self, inp, targ):
      '''
      args: train_mode, True means for training, False for prediction or evaluation
      '''   
      enc_output, enc_hidden, enc_cell = self.encoder(inp) #inp shape (batch_size, Tx)
      dec_hidden = enc_hidden # (batch_size,dec_units)  #if encoder and decoder hidden units # diff, need to initialize dec_hidden accordingly
      dec_cell = enc_cell
      Ty = targ.shape[1]
      dec_input = tf.expand_dims(targ[:,0], 1) 
      y_pred = []
      y_pred = tf.TensorArray(tf.float32, size=Ty)
      for t in range(Ty):   #targ.shape[1] is Ty
        # passing enc_output to the decoder. prediction shape == (batch_size, vocab_tar_size)
        predictions, dec_hidden, dec_cell, _ = self.decoder(dec_input, dec_hidden, 
                                                       dec_cell, enc_output) #throws away attension weights
        predicted_id = tf.math.argmax(predictions, axis=-1) #predicted_id shape == (batch_size,)
        #y_pred.append(predictions) #(Ty, batch_size)
        y_pred = y_pred.write(t, predictions)
        dec_input = tf.expand_dims(predicted_id, 1)
      #vocab_tar_size = y_pred[0].shape[1] 
      #batch_size = y_pred[0].shape[0] 
      #y_pred = tf.concat(y_pred, axis=-1)
      #y_pred = tf.reshape(y_pred, [batch_size, Ty, vocab_tar_size])
      y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
      return y_pred
    
    def custom_eval(self, input_indexed, inp_tokenizer, targ_tokenizer, max_length_inp, 
              max_length_targ, hidden_units_num):
      attention_plot = np.zeros((max_length_targ, max_length_inp))
    
      attention_plot = np.zeros((max_length_targ, max_length_inp))
      input_tensor = tf.convert_to_tensor(input_indexed)
    
      enc_out, enc_hidden_states, enc_cell = self.encoder(input_tensor)
    
      init_dec_hidden_states = enc_hidden_states
      dec_cell = enc_cell
      dec_input = tf.expand_dims([targ_tokenizer.word_index['<start>']], 0)  #no need to multiply batch size because it's single example
      predicted_sentence = '<start> '
      dec_hidden = init_dec_hidden_states
      for t in range(max_length_targ):
        prediction, dec_hidden, dec_cell, attention_weights = self.decoder(dec_input,
                                                             dec_hidden, dec_cell,
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

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

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

  #a customized callback saving models
class SaveCheckpoint(tf.keras.callbacks.Callback):
  def __init__(self, model, optimizer, path):
    super(SaveCheckpoint, self).__init__()      
    self.checkpoint_prefix = os.path.join(path, "ckpt")
    self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
  def on_epoch_end(self, epoch, logs=None):
    if (epoch + 1) % 2 == 0:
      self.checkpoint.save(file_prefix=self.checkpoint_prefix)
      
def bleu_score(inputs_indexed, targets_indexed, y_preds_indexed, params):
  targ_sentences, predicted_sentences = list(), list()
  assert(len(inputs_indexed) == len(targets_indexed) == len(y_preds_indexed))
  for i, input_indexed in enumerate(inputs_indexed): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(params['targ_tokenizer'], targets_indexed[i])
    predicted_sentence = convert_to_sentence_tagged(params['targ_tokenizer'], y_preds_indexed[i])
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

