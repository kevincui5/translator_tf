# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
#import os
#import time
from trainer2.util2 import get_dataset_params, load_dataset, Decoder, bleu_score, Encoder, SaveCheckpoint, convert_to_sentence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#import os.path
from os import path

def train(args):
    #assert not path.exists(args['output_dir']), "Model directory {} exists.".format(args['output_dir'])
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
    input_ndarray_valid, target_ndarray_valid = load_dataset(inp_tokenizer, 
                                                             targ_tokenizer, 
                                                             args['valid_data_path']) 
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE 
 
    BUFFER_SIZE = len(input_ndarray_train)

    
    optimizer = tf.keras.optimizers.Adam()        

    num_epochs = args['num_epochs']
    dataset_train = tf.data.Dataset.from_tensor_slices((input_ndarray_train, 
                                                  target_ndarray_train)).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num)
    decoder = Decoder(encoder, targ_tokenizer, vocab_inp_size, vocab_tar_size, embedding_dim, 
                      hidden_units_num, BATCH_SIZE)
    decoder.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'], run_eagerly = True)
#    checkpoint_dir = args['output_dir']
#    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#    checkpoint = tf.train.Checkpoint(optimizer='adam',
#                                     encoder=encoder,
#                                     decoder=decoder)
    early_stopping = EarlyStopping(monitor='loss', patience=2)
#    early_stopping = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=2)
    #decoder.fit(dataset_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, 
     #           callbacks=[early_stopping], validation_data=(dataset_valid), verbose=2)
    callbacks = [SaveCheckpoint(encoder, decoder, optimizer, args['output_dir'])]
    decoder.fit(dataset_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, 
                verbose=2, callbacks=callbacks) 
     #           callbacks=[early_stopping], validation_split = 0.8)
    #tf.keras.models.save_model(decoder, args["output_dir"])
    #decoder.save_weights(args['output_dir'])
    #decoder.get_encoder().save_weights(args['output_dir'])
    #decoder.get_attention().save_weights(args['output_dir'])


    dataset_valid = tf.data.Dataset.from_tensor_slices((input_ndarray_valid, 
                                                  target_ndarray_valid))
    if BATCH_SIZE > input_ndarray_valid.shape[0]:
      return
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=True)
    test = decoder.predict(dataset_valid)
    print(vocab_tar_size)
    print(len(test))
    print(type(test))
    print(test[0])
    print('***************')
    print(test[1])
    y_preds_indexed = np.argmax(test, axis=-1)
    bleu_score(input_ndarray_valid[:BATCH_SIZE], target_ndarray_valid[:BATCH_SIZE], y_preds_indexed,params)
    #print(test[:10])
    #print(y_preds_indexed[:10])
    '''
    for epoch in range(EPOCHS):
      start = time.time()
    
      total_loss = 0
    
      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = decoder.train_step(inp, targ)
        total_loss += batch_loss
    
        if batch % 100 == 0:
          print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
      # saving (checkpoint) the model every 2 epochs
      if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    
      print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
      print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
      '''