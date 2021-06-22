# -*- coding: utf-8 -*-
import tensorflow as tf
import unicodedata
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os.path
from nltk.translate.bleu_score import corpus_bleu
from trainer6.Translator import Translator

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
  vocab_inp_size = len(inp_tokenizer.word_index) + 1
  vocab_tar_size = len(targ_tokenizer.word_index) + 1
  params['vocab_inp_size'] = vocab_inp_size
  params['vocab_tar_size'] = vocab_tar_size
  params['inp_tokenizer'] = inp_tokenizer
  params['targ_tokenizer'] = targ_tokenizer
  params['max_length_targ'] = max_length_targ
  params['max_length_inp'] = max_length_inp
  return params

def build_model(vocab_inp_size, vocab_tar_size, inp_tokenizer, targ_tokenizer,
                embedding_dim, hidden_units_num):
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = Translator(vocab_inp_size, vocab_tar_size, embedding_dim, hidden_units_num)
    #work around from https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122    
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=tf.Variable(0.001),
      beta_1=tf.Variable(0.9),
      beta_2=tf.Variable(0.999),
      epsilon=tf.Variable(1e-7),
    )
    optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
    optimizer.decay = tf.Variable(0.0)  # Adam.__init__ assumes ``decay`` is a float object, so this needs to be converted to tf.Variable **after** __init__ method.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=['SparseCategoricalCrossentropy', 'SparseCategoricalAccuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, optimizer
  
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

# class BleuMetric(tf.keras.metrics.Metric): 
#   def __init__(self, inp_tokenizer, targ_tokenizer, name=None):
#     super(BleuMetric, self).__init__(name=name)
#     self.inp_tokenizer = inp_tokenizer
#     self.targ_tokenizer = targ_tokenizer
#     self.targ_sentences = list()
#     self.predicted_sentences = list()
       
#   def update_state(self, y, y_pred,sample_weight=None):
#     targ_sentence = convert_to_sentence(self.targ_tokenizer, y.eval())
#     predicted_sentence = convert_to_sentence_tagged(self.targ_tokenizer, y_pred.eval())
#     self.targ_sentences.append([targ_sentence.split()[1:-1]])
#     self.predicted_sentences.append(predicted_sentence.split()[1:-1])    
#     return self.total_cm
      
#   def result(self):
#     blue1 = corpus_bleu(self.targ_sentences, self.predicted_sentences, weights=(1.0, 0, 0, 0))
#     return blue1
