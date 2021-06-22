# -*- coding: utf-8 -*-

import tensorflow as tf
from trainer6.util import get_dataset_params, load_dataset, build_model, SaveCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import os.path

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
    input_ndarray_valid, target_ndarray_valid = load_dataset(inp_tokenizer, 
                                                             targ_tokenizer, 
                                                             args['valid_data_path'])
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE
 
    BUFFER_SIZE = len(input_ndarray_train)
    

    EPOCHS = args['num_epochs']
    dataset_train = tf.data.Dataset.from_tensor_slices((input_ndarray_train, 
                                                        target_ndarray_train)).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
    dataset_valid = tf.data.Dataset.from_tensor_slices((input_ndarray_valid, 
                                                  target_ndarray_valid))
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=True)

    model, optimizer = build_model(vocab_inp_size, vocab_tar_size, inp_tokenizer, 
                                   targ_tokenizer,embedding_dim, hidden_units_num)

    early_stopping = EarlyStopping(monitor='loss', patience=2)

    #callbacks = [SaveCheckpoint(optimizer, model, args['output_dir'])]
    model.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, 
              verbose=2, callbacks=[early_stopping], validation_data=dataset_valid) 

      
      
    checkpoint_prefix = os.path.join(args['output_dir'], "ckpt")
    model.save_weights(checkpoint_prefix, save_format='tf')
