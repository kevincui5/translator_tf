# -*- coding: utf-8 -*-
from trainer6.util import convert_to_sentence, get_dataset_params, load_dataset, build_model
import os.path
from tensorflow.data import Dataset
from nltk.translate.bleu_score import corpus_bleu

hidden_units_num = 1024
BATCH_SIZE = 256
embedding_dim = 256

example_limit = 180000
full_data_path = 'english-german-{}.csv'.format(example_limit)
train_data_path = 'english-german-train-{}.csv'.format(example_limit)
test_data_path = 'english-german-test-{}.csv'.format(example_limit)
saved_model_path = './trained_model6_{}'.format(example_limit)
test_limit = BATCH_SIZE

def predict(inputs_indexed, targets_indexed, params, model):
  targ_sentences, predicted_sentences = list(), list()
  for i, in_targ_indexed in enumerate(zip(inputs_indexed, targets_indexed)): #because we use special token to mark end of sentence, each indexed sentence has diff len, so can't predict as vector
    targ_sentence = convert_to_sentence(params['targ_tokenizer'], in_targ_indexed[1])
    inp_sentence = (convert_to_sentence(params['inp_tokenizer'], in_targ_indexed[0]))
    targ_sentences.append([targ_sentence.split()[1:-1]])
    predicted_sentence, attention_plot = model.translate([in_targ_indexed[0]], 
                                                   params['inp_tokenizer'], params['targ_tokenizer'], 
                                                   params['max_length_inp'], params['max_length_targ'],
                                                   hidden_units_num)
    predicted_sentences.append(predicted_sentence.split()[1:-1])
    if i > test_limit:
      break
    if i > 5: #show a preview
      continue
    print('src=[%s], target=[%s], predicted=[%s]' % (inp_sentence, 
          targ_sentence, predicted_sentence))

  print('BLEU-1: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(targ_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25)))


  
def evaluate():
  params = get_dataset_params(full_data_path)  
  inp_tokenizer = params['inp_tokenizer']
  targ_tokenizer = params['targ_tokenizer']
  vocab_inp_size = params['vocab_inp_size']
  vocab_tar_size = params['vocab_tar_size']
  train_inputs_indexed, train_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, train_data_path)
 
  BUFFER_SIZE = len(train_inputs_indexed)
  model, optimizer = build_model(vocab_inp_size, vocab_tar_size, inp_tokenizer, 
                                 targ_tokenizer,embedding_dim, hidden_units_num)

  dataset_train = Dataset.from_tensor_slices((train_inputs_indexed[:test_limit], 
                                                  train_targets_indexed[:test_limit])).shuffle(BUFFER_SIZE)
  dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True) #((batch_size,Tx),(batch_size,Ty))

  checkpoint_prefix = os.path.join(saved_model_path, "ckpt")
  model.load_weights(checkpoint_prefix)

  predict(train_inputs_indexed, train_targets_indexed, params, model)
  
  
  
  test_inputs_indexed, test_targets_indexed = load_dataset(inp_tokenizer, 
                                                 targ_tokenizer, test_data_path)  
  dataset_test = Dataset.from_tensor_slices((test_inputs_indexed[:test_limit], 
                                                  test_targets_indexed[:test_limit])).shuffle(BUFFER_SIZE)
  dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)
  predict(test_inputs_indexed, test_targets_indexed, params, model)

  model.evaluate(dataset_test) 

evaluate()
 