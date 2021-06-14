copied from trainer4, encoder, decoder, attentions are all layers, Translator a model
training can finish but nothing is trained if use model.compile_loss()

have to use customized training loop so the sampling(inference) can use algorithm such as 
beam search

trained on 10000 and 1700, prediction long sentence of non sense
cause is checkpoint restore issue.
use model.save_weights in tf format instead
to test on alienubuntu