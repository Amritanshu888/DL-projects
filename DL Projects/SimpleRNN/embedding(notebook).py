## Tab 1
from tensorflow.keras.preprocessing.text import one_hot

## Tab 2
### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

## Tab 3
sent

## Tab 4
## Define the vocabulary size
voc_size=10000

## Tab 5
sent

## Tab 6
### One Hot Representation
one_hot_repr=[one_hot(words,voc_size)for words in sent]
one_hot_repr

## Tab 7
## word Embedding Representation

from tensorflow.keras.layers import Embedding
#from tensorflow.keras.processing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential

## Tab 8
import numpy as np

## Tab 9
sent_length=8
embedded_docs=pad_sequences(one_hot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

## Tab 10
## feature representation
dim=10

## Tab 11
model=Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_length))
model.compile('adam','mse')

## Tab 12
model.summary()

## Tab 13
model.predict(embedded_docs)

## Tab 14
embedded_docs[0]

## Tab 15
model.predict(embedded_docs[0])