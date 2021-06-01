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

  def call(self, x):
    x = self.embedding(x)
    hidden = self.initialize_hidden_state()
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
  def __init__(self, targ_tokenizer, vocab_inp_size, vocab_tar_size, 
               embedding_dim, dec_units, batch_sz):
      super(Decoder, self).__init__()
      self.batch_sz = batch_sz
      self.dec_units = dec_units
      self.embedding = tf.keras.layers.Embedding(vocab_tar_size, embedding_dim)
      self.gru = tf.keras.layers.GRU(self.dec_units,
                                     return_sequences=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform')
      self.fc = tf.keras.layers.Dense(vocab_tar_size)        
      # used for attention
      self.attention = BahdanauAttention(self.dec_units)
      self.encoder = Encoder(vocab_inp_size, embedding_dim, dec_units, 
                             self.batch_sz) #encoder use same hidden units for now
      self.targ_tokenizer = targ_tokenizer
      self.enc_output = None  #(batch_size,Tx,dec_units)
      self.dec_hidden = None # (batch_size,dec_units)
      #self.optimizer = optimizer
        
  def call(self, x):
      # enc_output shape == (batch_size, max_length, hidden_size)
      context_vector, attention_weights = self.attention(self.dec_hidden, self.enc_output)
      
      # x shape after passing through embedding == (batch_size, 1, embedding_dim)
      x = self.embedding(x)
      
      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
      
      # passing the concatenated vector to the GRU
      output, state = self.gru(x)
      
      # output shape == (batch_size * 1, hidden_size)
      output = tf.reshape(output, (-1, output.shape[2]))
      
      # x shape == (batch_size, vocab_tar_size)
      x = self.fc(output)
      
      return x, state, attention_weights
      
  def train_step(self, data): #data shape (tensor shape(batch_size,Tx),tensor shape(batch_size,Ty))
    loss = 0
    inp, targ, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
    #inp shape (batch_size,Tx), targ shape (batch_size,Ty)

    with tf.GradientTape() as tape:
      self.enc_output, enc_hidden = self.encoder(inp)
  
      self.dec_hidden = enc_hidden
      #insert a length 1 axis at 2nd dimension, meaning use a start word id as each training example
      dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']] * self.batch_sz, 1) 
      y_pred = []
      # Teacher forcing - feeding the target (ground truth) as the next decoder input
      for t in range(1, targ.shape[1]):   #targ.shape[1] is Ty
        # passing enc_output to the decoder. prediction shape == (batch_size, vocab_tar_size)
        prediction, dec_hidden, _ = self(dec_input) #dec_input shape == (batch_size,1). throws away attension weights
        y_pred.append(prediction)
        #targ shape == (batch_size, ty)
        #loss += self.loss_function(targ[:, t], prediction) #take parameters of ground truth and prediction
        loss += self.compiled_loss(targ[:, t], y_pred, regularization_losses=self.losses)
         
        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)  # targ[:, t] is y
      y_pred = tf.concat(y_pred, axis=0)   #y_pred (list of Ty tensors) shape (Ty, batch_size, vocab_tar_size) -> (Ty*batch_size, vocab_tar_size)

      y_pred = tf.reshape(y_pred, [self.batch_sz, targ.shape[1]-1, y_pred.shape[1]]) #(batch_size,Ty,vocab_tar_size)
    #because of teacher's forcing, targ shape has 1 more y_pred along Ty axis 
    targ = targ[:,1:]
    batch_loss = (loss / int(targ.shape[1]))
  
    variables = self.encoder.trainable_variables + self.trainable_variables
  
    gradients = tape.gradient(loss, variables)
  
    self.optimizer.apply_gradients(zip(gradients, variables))
    self.compiled_metrics.update_state(targ, y_pred)
    return {m.name: m.result() for m in self.metrics} 
    #return batch_loss
  
  def predict_step(self, data):
    """The logic for one inference step.
    This method overridde that of Keras.Model to support one step translation,
    the forward pass.
    Args:
      data: A nested structure of `Tensor`s.
    Returns:
      The translated sentences.
    """  
    inp, targ, sample_weigh = tf.keras.utils.unpack_x_y_sample_weight(data)
    #if (not isinstance(t, tf.Tensor) or
    #  (not isinstance(t.shape, tf.TensorShape)) or t.shape.rank != 1):
    #  return
    #inp = tf.expand_dims(inp, 1)
    self.enc_output, enc_hidden = self.encoder(inp)
    self.dec_hidden = enc_hidden # (batch_size,dec_units)
    dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']] * self.batch_sz, 1) 
    y_pred = []
    for t in range(targ.shape[1]):
      #dec_input shape == (batch_size,1)
      prediction, dec_hidden, _ = self(dec_input) #throws away attension weights
      predicted_id = tf.math.argmax(prediction, 1)
      #predicted_word = self.targ_tokenizer.index_word[predicted_id]
      #predicted_sentence += predicted_word + ' '
      #if predicted_word == '<end>':
      #  return predicted_sentence
      dec_input = tf.expand_dims(predicted_id, 1)
      y_pred.append(dec_input)
    y_pred = tf.concat(y_pred, axis=0)   #y_pred (list of Ty tensors) shape (Ty, batch_size, 1) -> (Ty*batch_size, 1)
#    y_pred = tf.reshape(y_pred, [self.batch_sz, targ.shape[1], y_pred.shape[1]]) #(batch_size,Ty,vocab_tar_size)
    y_pred = tf.reshape(y_pred, [self.batch_sz, targ.shape[1]]) #(batch_size,Ty,vocab_tar_size)
    return tf.convert_to_tensor(y_pred)
  '''  
    def test_step(self, data):
      """The logic for one evaluation step.
      This method overridde that of Keras.Model to support one step evaluationlogic,
      that is, the forward pass, loss calculation, and metrics updates.
      Args:
        data: A nested structure of `Tensor`s.
      Returns:
        metrics.
      """   
      inp, targ, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

      self.enc_output, enc_hidden = self.encoder(inp)
      dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']] * self.batch_sz, 1) 
      for t in range(targ.shape[1]):   #targ.shape[1] is Ty
        # passing enc_output to the decoder. prediction shape == (batch_size, vocab_tar_size)
        prediction, dec_hidden, _ = self(dec_input) #throws away attension weights
        predicted_id = tf.math.argmax(prediction, axis=1).numpy() #predicted_id shape == (batch_size, 1)
        predicted_word = targ_tokenizer.index_word[predicted_id]
    predicted_sentence += predicted_word + ' '

    if predicted_word == '<end>':
      return predicted_sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
      self.compiled_loss(y, y_pred, regularization_losses=self.losses)
      self.compiled_metrics.update_state(y, y_pred)
      return {m.name: m.result() for m in self.metrics}
      #return batch_loss
  '''

  def predict_step_one_example(self, inp, max_len_out):
    '''
    Helper function to be called by test_step and predict_step
    It pass one test example through forward pass
    Args:
      inp with dimention of (Tx,).
    Returns:
      A translated sentence.
    '''
    inp = tf.expand_dims(inp, 0)
    self.enc_output, enc_hidden = self.encoder(inp)
    dec_input = tf.expand_dims([self.targ_tokenizer.word_index['<start>']], 0)  #no need to multiply batch size because it's single example
    predicted_sentence = '<start> '
    for t in range(max_len_out):
      prediction, dec_hidden, _ = self(dec_input) #throws away attension weights
      predicted_id = tf.math.argmax(prediction[0]).numpy()
      predicted_word = self.targ_tokenizer.index_word[predicted_id]
      predicted_sentence += predicted_word + ' '
      if predicted_word == '<end>':
        return predicted_sentence
      dec_input = tf.expand_dims([predicted_id], 0)
    return predicted_sentence
  
  def loss_function(self, real, pred):
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
