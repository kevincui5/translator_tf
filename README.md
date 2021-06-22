
# Another NMT Keras Tutorial

![Compatibility](img/Python-3.7-blue.svg)![Compatibility](img/Tensorflow-2.4-blue.svg) [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/lvapeab/nmt-keras/blob/master/LICENSE)

Neural Machine Translation with attention mechanism implemented with tensorflow 2's newest feature such as overriding training step and test step function of Keras subclassed model.

Though [Keras NMT with attention tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention) already shows how to implement a basic NMT model with attention with tensorflow, it would be interesting to try out some other new functionality comes out since tensorflow 2.2. since these model needs custom training and evaluating loop and it would be a good test case to implement these features.
By subclassing keras model and overriding train_step and test_step for customized training and inference loop
but still able to use the many features come with keras model, such as fit, evaluate,
metric, etc...


using the best practice suggested by tensorflow guide
i picked NMT because it is relatively complex that using functional API is ok but 
hard to maintain the code.  using the subclassing (objected oriented) approach produce much better result
Note that the __init__() method of the base Layer class takes some keyword arguments, in particular a name and a dtype. It's good practice to pass these arguments to the parent class in __init__() and to include them in the layer config:

Encoder, Decoder, and BahdanauAttention are all layers, and Translator is a model.
`
class Encoder(Layer):
`
`
class Decoder(Layer):
`
`
class BahdanauAttention(Layer):
`

By assigning Keras Layers' instance as attributes of other Layers, the weights of inner layer become trackable:
```
class Translator(tf.keras.Model):
    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, 
                 hidden_units_num, name="Translator", 
                 **kwargs):
      super(Translator, self).__init__(name=name, **kwargs)
      self.encoder = Encoder(vocab_inp_size, embedding_dim, hidden_units_num)
      self.decoder = Decoder(vocab_tar_size, embedding_dim, hidden_units_num)
```      
So Translator model is composed of Decoder and Encoder Layers, and Decoder Layer is composed of Embedding, LSTM, Dense, and BahdanauAttention.  By doing so, all the weights from all these layers are trackable, making accessing trainable weights very easy:
```
trainable_vars = self.trainable_variables    
gradients = tape.gradient(loss, trainable_vars)
``` 
In the original tutorial, the Encoder is a Keras Model instead of Layer, so Enocder's trainable_variables have to be manually included. 

Note: we lazily create layers' weights during layers' instantiation like Kera's best practice guide suggests.  Notice there is no input_shape specified in lstm __init__() or input_length in embedding init():

```
class Encoder(Layer):
  def __init__(self, vocab_size, embedding_dim, units, 
               name="Encoder", **kwargs):
    super(Encoder, self).__init__(**kwargs)
    self.units = units
    self.embedding = Embedding(vocab_size, embedding_dim)
    self.bi_LSTM_last = Bidirectional(LSTM(units, return_sequences=True, 
                                      return_state = True))
    self.bi_LSTM = Bidirectional(LSTM(units, return_sequences=True))
```
Other things you can do with an individual Keras Layer are customizing loss and metric, making it trainable or not etc.

For the loss function, you can either implement one like that in the original tutorial, or use the Keras' build-in loss functions passed in through model.compile():

`
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
`

`
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
`

Also in the training loop, we can use the specified loss object and metrics objects from model.compile function inside train_step function because we are overriding train_step:
```
...
loss += self.compiled_loss(targ[:, t], predictions)
self.compiled_metrics.update_state(targ[:, t], predictions)
...
self.optimizer.apply_gradients(zip(gradients, trainable_vars))
...
```
Notice we pass from_logits=True to the loss function object for numerical stability.  We have to use linear activation function in the Dense layer in Decoder then.  The loss function object will take care of softmax function part.  See [this](https://stackoverflow.com/questions/52125924/why-does-sigmoid-crossentropy-of-keras-tensorflow-have-low-precision/52126567#52126567) for detail.

By subclassing Keras Model class and overriding train_step(), we can use model.fit() to train our model.  We now gain all the functionality that come with fit(), such as callbacks, batch and epoch handling, validation set metrics monitoring, and custom training loop logic.
`model.fit(dataset_train, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, verbose=2, callbacks=[early_stopping], validation_data=dataset_valid)`

We are also able to provide different logic in back prop, which is in train_step() and in forward pass, which is in model's call() function.
```
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
```
Here, teacher's force is used, just like the original Keras tutorial, where ground truth at time step t is extracted and feed in as decoder's input.
```
def forward_pass(self, inp, targ):   
      enc_output, enc_hidden, enc_cell = self.encoder(inp) #inp shape (batch_size, Tx)
      dec_hidden = enc_hidden # (batch_size,dec_units)  #if encoder and decoder hidden units # diff, need to initialize dec_hidden accordingly
      dec_cell = enc_cell
      Ty = targ.shape[1]
      dec_input = tf.expand_dims(targ[:,0], 1) 
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
      y_pred = tf.transpose(y_pred.stack(), [1, 0, 2])
      return y_pred
```
Here in the forward pass, at each time step, y_pred is appended to TensorArray.  Then at the end the prediction tensor is transposed so that first dimension is batch size, second is timestep.  greedy sampling method is used here, where max is applied on the probability distribution and converted to word index.  Beam search can be implemented here.
Again TensorArray is used here instead of python list to avoid potential python side effects as suggested by tensorflow guide as best practice.

We can also override test_step() to provide custom evaluation logic, so we can use model.evaluate() with all the functionality it brings to monitor the loss and metrics on validation set. 
```
def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(data, training=False)

        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
```
Inside the function it simply calls model's call() function which just invokes forward pass logic that records loss and metrics.



<!--<div align="left">-->
  <!--<br><br><img  width="100%" "height:100%" "object-fit: cover" "overflow: hidden" src=""><br><br>-->
<!--</div>-->

## Attentional recurrent neural network NMT model
![alt text](img/translator_model.png "RNN NMT")




## What each file does: 
 * trainer6/BahdanauAttention.py: Define attention layer.  Uses "add" attention mechanism. 
 * trainer6/Decoder.py: Define decoder layer, which contains a single LSTM layer.
 * trainer6/Encoder.py: Define encoder layer, which contains a single Bi-LSTM layer.
 * trainer6/model.py: Get input and target tokenizers, max length of input and target language sentences from the "full" dataset file, and read in training data from training file and validation data from validation file and does the training.
 * trainer6/task.py: Parse the commad arguments.
 * trainer6/Translator.py: Define the Translator model, which contains reference to encoder and decoder layers.  Also contains the overriden train_step and test_step functions and translate function, which is just prediction on a single example.
 * trainer6/util.py: All the utility functions and classes used by model.py and eval6.py.  Many from the Keras NMT tutorial.
 * config.yaml:
 * deu.txt:
 * english-german-x.csv: x represent the number of examples in "full" data file. Contains tab seperated language pairs examples.  Created by prepare_input_files.py

 * english-german-test-x.csv: Language pairs examples for testing.
 * english-german-train-x.csv: Language pairs examples for training.
 * english-german-valid-x.csv: Language pairs examples for validation.
 * eval6.py: Get input and target tokenizers, max length of input and target language sentences from the "full" dataset file, and read in training data from training file and validation data from validation file and does the evaluation in metrics from model.compile and in BLEU scores.
 * prepare_input_files.py: 
 * train-gcp.sh: A shell script submitting training job to AI platform service from Google Cloud.  Modify the config.yaml to configure the cloud server instance. 
 * train-local6.sh: 

## Usage

### Preparing input files
 1) Set a training configuration in the `config.py` script. Each parameter is commented. See the [documentation file](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/config.md) for further info about each specific hyperparameter.
 You can also specify the parameters when calling the `main.py` script following the syntax `Key=Value`

 2) Train!:

  ``
 python main.py
 ``

### Training
 1) Set a training configuration in the `config.py` script. Each parameter is commented. See the [documentation file](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/config.md) for further info about each specific hyperparameter.
 You can also specify the parameters when calling the `main.py` script following the syntax `Key=Value`

 2) Train!:

  ``
 python main.py
 ``


### Evaluating
 Once we have our model trained, we can translate new text using the [sample_ensemble.py](https://github.com/lvapeab/nmt-keras/blob/master/sample_ensemble.py) script. Please refer to the [ensembling_tutorial](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/ensembling_tutorial.md) for more details about this script. 
In short, if we want to use the models from the first three epochs to translate the `examples/EuTrans/test.en` file, just run:
 ```bash
  python sample_ensemble.py 
              --models trained_models/tutorial_model/epoch_1 \ 
                       trained_models/tutorial_model/epoch_2 \
              --dataset datasets/Dataset_tutorial_dataset.pkl \
              --text examples/EuTrans/test.en
  ```
 
 
### Translating
 
 The [score.py](https://github.com/lvapeab/nmt-keras/blob/master/score.py) script can be used to obtain the (-log)probabilities of a parallel corpus. Its syntax is the following:
```
python score.py --help
usage: Use several translation models for scoring source--target pairs
       [-h] -ds DATASET [-src SOURCE] [-trg TARGET] [-s SPLITS [SPLITS ...]]
       [-d DEST] [-v] [-c CONFIG] --models MODELS [MODELS ...]
optional arguments:
    -h, --help            show this help message and exit
    -ds DATASET, --dataset DATASET
                            Dataset instance with data
    -src SOURCE, --source SOURCE
                            Text file with source sentences
    -trg TARGET, --target TARGET
                            Text file with target sentences
    -s SPLITS [SPLITS ...], --splits SPLITS [SPLITS ...]
                            Splits to sample. Should be already includedinto the
                            dataset object.
    -d DEST, --dest DEST  File to save scores in
    -v, --verbose         Be verbose
    -c CONFIG, --config CONFIG
                            Config pkl for loading the model configuration. If not
                            specified, hyperparameters are read from config.py
    --models MODELS [MODELS ...]
                            path to the models
  ```

## Resources

 * [examples/documentation/nmt-keras_paper.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/nmt-keras_paper.pdf) contains a general overview of the NMT-Keras framework.
 
 * In [examples/documentation/neural_machine_translation.pdf](https://github.com/lvapeab/nmt-keras/blob/master/examples/documentation/neural_machine_translation.pdf) you'll find an overview of an attentional NMT system.

 * In the [examples](https://github.com/lvapeab/nmt-keras/blob/master/examples/) folder you'll find  2 colab notebooks, explaining the basic usage of this library:
 
 * An introduction to a complete NMT experiment: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lvapeab/nmt-keras/blob/master/examples/tutorial.ipynb) 
  * A dissected NMT model: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lvapeab/nmt-keras/blob/master/examples/modeling_tutorial.ipynb) 
 

 * In the [examples/configs](https://github.com/lvapeab/nmt-keras/blob/master/examples/configs) folder you'll find two examples of configs for larger models.

## Citation

If you use this toolkit in your research, please cite:

```
@article{nmt-keras:2018,
 journal = {The Prague Bulletin of Mathematical Linguistics},
 title = {{NMT-Keras: a Very Flexible Toolkit with a Focus on Interactive NMT and Online Learning}},
 author = {\'{A}lvaro Peris and Francisco Casacuberta},
 year = {2018},
 volume = {111},
 pages = {113--124},
 doi = {10.2478/pralin-2018-0010},
 issn = {0032-6585},
 url = {https://ufal.mff.cuni.cz/pbml/111/art-peris-casacuberta.pdf}
}
```


NMT-Keras was used in a number of papers:

* [Online Learning for Effort Reduction in Interactive Neural Machine Translation](https://arxiv.org/abs/1802.03594)
* [Adapting Neural Machine Translation with Parallel Synthetic Data](http://www.statmt.org/wmt17/pdf/WMT14.pdf)
* [Online Learning for Neural Machine Translation Post-editing](https://arxiv.org/pdf/1706.03196.pdf)


### Acknowledgement

Much of this library has been developed together with [Marc Bolaños](https://github.com/MarcBS) ([web page](http://www.ub.edu/cvub/marcbolanos/)) for other sequence-to-sequence problems. 

To see other projects following the same philosophy and style of NMT-Keras, take a look to:

[TMA: Egocentric captioning based on temporally-linked sequences](https://github.com/MarcBS/TMA).

[VIBIKNet: Visual question answering](https://github.com/MarcBS/VIBIKNet).

[ABiViRNet: Video description](https://github.com/lvapeab/ABiViRNet).

[Sentence SelectioNN: Sentence classification and selection](https://github.com/lvapeab/sentence-selectioNN).

[DeepQuest: State-of-the-art models for multi-level Quality Estimation](https://github.com/sheffieldnlp/deepQuest).





## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es 
