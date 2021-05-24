# -*- coding: utf-8 -*-

import tensorflow as tf

from sklearn.model_selection import train_test_split

#import numpy as np
import os
import time
from trainer.util import *

#example_input_batch, example_target_batch = next(iter(dataset))
#example_input_batch.shape, example_target_batch.shape

def train(args):
    #params = create_Dataset(args['complete_data_path'])
    examples_limit = 400000
    input_ndarray, target_ndarray, inp_tokenizer, targ_tokenizer = load_dataset(args['complete_data_path'], examples_limit)
    BATCH_SIZE = 256
    #steps_per_epoch = 6624
    embedding_dim = 256
    hidden_units_num = 1024
    vocab_inp_size = len(inp_tokenizer.word_index)+1
    vocab_tar_size = len(targ_tokenizer.word_index)+1
    input_ndarray_train, input_ndarray_val, target_ndarray_train, target_ndarray_val = train_test_split(input_ndarray, target_ndarray, test_size=0.2)       
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE
 
    BUFFER_SIZE = len(input_ndarray_train)
    encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    #attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num, BATCH_SIZE)
    #sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)), sample_hidden, sample_output)
    
    optimizer = tf.keras.optimizers.Adam()        
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    EPOCHS = 2
    dataset = tf.data.Dataset.from_tensor_slices((input_ndarray_train, target_ndarray_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    for epoch in range(EPOCHS):
      start = time.time()
    
      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0
    
      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden, encoder, decoder, 
                                BATCH_SIZE, targ_tokenizer, optimizer)
        total_loss += batch_loss
    
        if batch % 100 == 0:
          print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    
      print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
      print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
#train('')
'''  
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')
translate(u'esta es mi vida.')
translate(u'¿todavia estan en casa?')
# wrong translation
translate(u'trata de averiguarlo.')
'''