# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense
from trainer6.BahdanauAttention import BahdanauAttention

class Decoder(Layer):
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

    #output and hidden are the same here because only return_state is set true
    output, hidden, cell = self.lstm(x, initial_state = [hidden, cell])
    # output shape == (batch_size * 1, hidden_size)

    # x shape == (batch_size, vocab_tar_size)
    x = self.fc(output)

    return x, hidden, cell, attention_weights

  def get_config(self):
    config = super(Decoder, self).get_config()
    config.update({"units": self.units})
    return config

