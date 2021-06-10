# -*- coding: utf-8 -*-

import tensorflow as tf
#import numpy as np
import os
from trainer5.util import get_dataset_params, load_dataset, Decoder, Encoder, SaveCheckpoint, train_step
import os.path
from os import path
import time

def train(args):
    assert not path.exists(args['output_dir']), "Model directory {} exists.".format(args['output_dir'])
    #path.exists(args['full_data_path'])
    #path.exists(args['train_data_path'])
    params = get_dataset_params(args['full_data_path'])
    BATCH_SIZE = args['batch_size']
    embedding_dim = args['embedding_dim']
    hidden_units_num = args['hidden_units']
    inp_tokenizer = params['inp_tokenizer']
    targ_tokenizer = params['targ_tokenizer']
    vocab_inp_size = params['vocab_inp_size']
    vocab_tar_size = params['vocab_tar_size']
    max_length_inp = params['max_length_inp']
    input_ndarray_train, target_ndarray_train = load_dataset(inp_tokenizer, 
                                                             targ_tokenizer, 
                                                             args['train_data_path'])      
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE
 
    BUFFER_SIZE = len(input_ndarray_train)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num, 
                          max_length_inp, BATCH_SIZE)
        
        #attention_layer = BahdanauAttention(10)
        #attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
        decoder = Decoder(encoder, vocab_tar_size, embedding_dim, hidden_units_num, BATCH_SIZE)
        
        optimizer = tf.keras.optimizers.Adam()        
    checkpoint_dir = args['output_dir']
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    EPOCHS = args['num_epochs']
    dataset_train = tf.data.Dataset.from_tensor_slices((input_ndarray_train, target_ndarray_train)).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
    decoder.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'], run_eagerly = True)    

#    callbacks = [SaveCheckpoint(encoder, decoder, optimizer, args['output_dir'])]
#    decoder.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, 
#                verbose=2, callbacks=callbacks) 

    
    for epoch in range(EPOCHS):
      start = time.time()
    
      
      total_loss = 0
    
      for (batch, (inp, targ)) in enumerate(dataset_train.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, encoder, decoder, 
                                BATCH_SIZE, targ_tokenizer, optimizer)
        total_loss += batch_loss
    
        if batch % 100 == 0:
          print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    
      print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
      print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
      