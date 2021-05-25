# -*- coding: utf-8 -*-

import tensorflow as tf
#import numpy as np
import os
import time
from trainer.util import *

def train(args):
    params = get_dataset_params(args['full_data_path'])
    BATCH_SIZE = args['batch_size']
    embedding_dim = args['embedding_dim']
    hidden_units_num = args['hidden_units']
    inp_tokenizer = params['inp_tokenizer']
    targ_tokenizer = params['targ_tokenizer']
    vocab_inp_size = params['vocab_inp_size']
    vocab_tar_size = params['vocab_tar_size']
    input_ndarray_train, target_ndarray_train = load_dataset(inp_tokenizer, 
                                                             targ_tokenizer, 
                                                             args['train_data_path'])      
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE
 
    BUFFER_SIZE = len(input_ndarray_train)
    encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num, BATCH_SIZE)
    attention_layer = BahdanauAttention(10)
    #attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
    decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num, BATCH_SIZE)
    
    optimizer = tf.keras.optimizers.Adam()        
    checkpoint_dir = args['output_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    EPOCHS = args['num_epochs']
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
