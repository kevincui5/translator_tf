
in trainer4 add pretrained embedding weights

in trainer4 add custom metric

use dataset in eval4.py, like trainer3's evaluation function

fix trainer2, prediction doens't work. not sure if issue in training or prediction

the tensorflow demo in the guide doesn't cover everything.  this prgram tries to
put as much as relevent new features in this project so you can better understand
how they work in an actual example.
multi gpu
nested layers
override train_step and test_step to for customized training and inference loop
but still able to use the many features come with keras model, such as fit, evaluate,
metric, etc...
using the best practice suggested by tensorflow guide
i picked NMT because it is relatively complex that using functional API is ok but 
hard to maintain the code.  using the subclassing (objected oriented) approach produce much better result
Note that the __init__() method of the base Layer class takes some keyword arguments, in particular a name and a dtype. It's good practice to pass these arguments to the parent class in __init__() and to include them in the layer config:

trainer 4v1
encoder bi-lstm single layer
decoder lstm single layer

trainer 4v2
encoder bi-lstm multi-layer
decoder lstm single layer

trainer 1
both encoder and decoder use GRU single layer

trial from trainer1
training examples 1700
BLEU-1: 0.337187
BLEU-2: 0.150184
BLEU-3: 0.079000
BLEU-4: 0.023919

BLEU-1: 0.314869
BLEU-2: 0.092428
BLEU-3: 0.000000
BLEU-4: 0.000000

trial from trainer1
training examples 80000
BLEU-1: 0.945607
BLEU-2: 0.920049
BLEU-3: 0.909508
BLEU-4: 0.880153

BLEU-1: 0.687310
BLEU-2: 0.585170
BLEU-3: 0.543949
BLEU-4: 0.446164

trail from trainer 4v1
training examples 1700
BLEU-1: 0.538687
BLEU-2: 0.291647
BLEU-3: 0.218193
BLEU-4: 0.119856

BLEU-1: 0.414557
BLEU-2: 0.169452
BLEU-3: 0.076513
BLEU-4: 0.000000

trail from trainer 4v1
training examples 80000
BLEU-1: 0.973352
BLEU-2: 0.958513
BLEU-3: 0.952494
BLEU-4: 0.935998

BLEU-1: 0.791289
BLEU-2: 0.688844
BLEU-3: 0.651557
BLEU-4: 0.572273

trail from trainer 4v2
training examples 80000
BLEU-1: 0.957678
BLEU-2: 0.939204
BLEU-3: 0.933369
BLEU-4: 0.913875

BLEU-1: 0.407349
BLEU-2: 0.210207
BLEU-3: 0.153813
BLEU-4: 0.074383