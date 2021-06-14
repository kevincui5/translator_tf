# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from trainer6.util import get_dataset_params, load_dataset, Translator, SaveCheckpoint, bleu_score
from tensorflow.keras.callbacks import EarlyStopping
import os.path

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
    steps_per_epoch = len(input_ndarray_train)//BATCH_SIZE
 
    BUFFER_SIZE = len(input_ndarray_train)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = Translator(vocab_inp_size, vocab_tar_size, embedding_dim, hidden_units_num)
      
      optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.Variable(0.001),
        beta_1=tf.Variable(0.9),
        beta_2=tf.Variable(0.999),
        epsilon=tf.Variable(1e-7),
      )
      optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
      optimizer.decay = tf.Variable(0.0)  # Adam.__init__ assumes ``decay`` is a float object, so this needs to be converted to tf.Variable **after** __init__ method.

    EPOCHS = args['num_epochs']
    dataset_train = tf.data.Dataset.from_tensor_slices((input_ndarray_train, target_ndarray_train)).shuffle(BUFFER_SIZE)
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
    #work around from https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
      
#    model.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
#                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'], 
#                    run_eagerly = True)
    model.compile(optimizer=optimizer, loss='SparseCategoricalCrossentropy', 
                    metrics=['SparseCategoricalCrossentropy','SparseCategoricalAccuracy'])
    early_stopping = EarlyStopping(monitor='loss', patience=2)

    #if want to continue training after restore, include optimizer in SaveCheckpoint
    callbacks = [SaveCheckpoint(optimizer, model, args['output_dir'])]
    model.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, 
                verbose=2) 
    checkpoint_prefix = os.path.join(args['output_dir'], "ckpt")
    model.save_weights(checkpoint_prefix, save_format='tf')
    input_ndarray_valid, target_ndarray_valid = load_dataset(inp_tokenizer, 
                                                             targ_tokenizer, 
                                                             args['valid_data_path'])
    dataset_valid = tf.data.Dataset.from_tensor_slices((input_ndarray_valid, 
                                                  target_ndarray_valid))
    if BATCH_SIZE > input_ndarray_valid.shape[0]:
      return
    dataset_valid = dataset_valid.batch(BATCH_SIZE, drop_remainder=True)
    test = model.predict(dataset_valid)
    print('***************')
    for i in range(3):
      print(test[i])
    iterator = iter(dataset_valid)
    for _ in range(3):
      print(iterator.next())
    y_preds_indexed = np.argmax(test, axis=-1)
    bleu_score(input_ndarray_valid[:BATCH_SIZE], target_ndarray_valid[:BATCH_SIZE], y_preds_indexed,params)
    #print(test[:10])

#    for epoch in range(EPOCHS):
#      start = time.time()
#      total_loss = 0
#    
#      for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
#        batch_loss, _ = train_step(inp, targ, model, optimizer)
#        total_loss += batch_loss
#    
#        if batch % 100 == 0:
#          print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
#      # saving (checkpoint) the model every 2 epochs
#      if (epoch + 1) % 2 == 0:
#        checkpoint.save(file_prefix=checkpoint_prefix)
#    
#      print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
#      print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
    