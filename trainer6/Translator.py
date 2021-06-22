# -*- coding: utf-8 -*-
import tensorflow as tf
from trainer6.Encoder import Encoder
from trainer6.Decoder import Decoder
import numpy as np

class Translator(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, 
                 hidden_units_num, name="Translator", 
                 **kwargs):
      super(Translator, self).__init__(name=name, **kwargs)
      self.encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num)
      self.decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num)
      #self.loss = 0
      
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
          loss += self.compiled_loss(targ[:, t], predictions) #take parameters of ground truth and prediction
          # using teacher forcing
          dec_input = tf.expand_dims(targ[:, t], 1)  # targ[:, t] is y
          self.compiled_metrics.update_state(targ[:, t], predictions)

      trainable_vars = self.trainable_variables    
      gradients = tape.gradient(loss, trainable_vars)    
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      # Return a dict mapping metric names to current value
      return {**{'loss': loss}, **{m.name: m.result() for m in self.metrics}}
    
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(data, training=False)
        
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
      enc_output, enc_hidden, enc_cell = self.encoder(inp) #inp shape (batch_size, Tx)
      dec_hidden = enc_hidden # (batch_size,dec_units)  #if encoder and decoder hidden units # diff, need to initialize dec_hidden accordingly
      dec_cell = enc_cell
      Ty = targ.shape[1]
      dec_input = tf.expand_dims(targ[:,0], 1) 
      #y_pred = []
      y_pred = tf.TensorArray(tf.float32, size=Ty)
      for t in range(Ty):   #targ.shape[1] is Ty
        # passing enc_output to the decoder. prediction shape == (batch_size, vocab_tar_size)
        predictions, dec_hidden, dec_cell, _ = self.decoder(dec_input, dec_hidden, 
                                                       dec_cell, enc_output) #throws away attension weights
        predicted_id = tf.math.argmax(predictions, axis=-1) #predicted_id shape == (batch_size,)
        #y_pred.append(predictions) #(Ty, batch_size)
        y_pred = y_pred.write(t, predictions)
        dec_input = tf.expand_dims(predicted_id, 1)
        self.compiled_loss(targ[:, t], predictions, regularization_losses=self.losses)
        self.compiled_metrics.update_state(targ[:, t], predictions)
      #vocab_tar_size = y_pred[0].shape[1] 
      #batch_size = y_pred[0].shape[0] 
      #y_pred = tf.concat(y_pred, axis=-1)
      #y_pred = tf.reshape(y_pred, [batch_size, Ty, vocab_tar_size])
      y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
      return y_pred
    
    def translate(self, input_indexed, inp_tokenizer, targ_tokenizer, max_length_inp, 
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
    
