# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Bidirectional, LSTM, Concatenate, Layer, Embedding

class Encoder(Layer):
  def __init__(self, vocab_size, embedding_dim, units, 
               name="Encoder", **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.units = units
    self.embedding = Embedding(vocab_size, embedding_dim)
    self.bi_LSTM_last = Bidirectional(LSTM(units, return_sequences=True, 
                                      return_state = True))
    self.bi_LSTM = Bidirectional(LSTM(units, return_sequences=True))
    
  def call(self, x):
    x = self.embedding(x)
    #total 4 layers ob bi-lstm
    #enc_layers = 3
    #for _ in range(enc_layers):
    #    x = self.bi_LSTM(x)
    output, hidden_fwd, cell_fwd, hidden_bwd, cell_bwd = self.bi_LSTM_last(x)
    hidden = Concatenate(axis=-1)([hidden_fwd, hidden_bwd])
    cell = Concatenate(axis=-1)([cell_fwd, cell_bwd])
    return output, hidden, cell

  def get_config(self):
    config = super(Encoder, self).get_config()
    config.update({"units": self.units})
    return config
