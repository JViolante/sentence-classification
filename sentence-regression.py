import numpy as np
import theano
import math
import csv
import keras
import sklearn
import gensim
import random
import scipy
import keras
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn.base import BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from word_movers_knn import WordMoversKNN
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error , mean_absolute_error
from geopy import distance

# size of the word embeddings
embeddings_dim = 300

# maximum number of words to consider in the representations
max_features = 20000

# maximum length of a sentence
max_sent_len = 50

# percentage of the data used for model training
percent = 0.75

# special case for geocoding problems -- regression in which the metric to optimize is the geospatial distance instead of RMSE
is_geocoding = True

#number of dimensions in regression problem
reg_dimensions = 2

def geodistance( coords1 , coords2 ):
  lat1 , lon1 = coords1[ : 2]
  lat2 , lon2 = coords2[ : 2]
  try: return distance.vincenty( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0
  except: return distance.great_circle( ( lat1 , lon1 ) , ( lat2 , lon2 ) ).meters / 1000.0

def geoloss( a , b ): 
#  return keras.backend.mean(keras.backend.square(a - b), axis=-1)
  aa = theano.tensor.deg2rad( a )
  bb = theano.tensor.deg2rad( b )
  sin_lat1 = theano.tensor.sin( aa[:,0] )
  cos_lat1 = theano.tensor.cos( aa[:,0] )
  sin_lat2 = theano.tensor.sin( bb[:,0] )
  cos_lat2 = theano.tensor.cos( bb[:,0] )
  delta_lng = bb[:,1] - aa[:,1]
  cos_delta_lng = theano.tensor.cos(delta_lng)
  sin_delta_lng = theano.tensor.sin(delta_lng)
  d = theano.tensor.arctan2(theano.tensor.sqrt((cos_lat2 * sin_delta_lng) ** 2 + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2), sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng )
  return theano.tensor.mean( 6371.009 * d )

print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 

print ("Reading text data for classification and building representations...")
data = [ ( row["sentence"] , ( float( row["latitude"] ) , float( row["longitude"] ) ) ) for row in csv.DictReader(open("test-data-geo.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data) * percent)
train_texts = [ txt for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt for ( txt, label ) in data[train_size:-1] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
tokenizer = Tokenizer(nb_words=max_features, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tokenizer.fit_on_texts(train_texts)
train_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( train_texts ) , maxlen=max_sent_len )
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences( test_texts ) , maxlen=max_sent_len )
train_matrix = tokenizer.texts_to_matrix( train_texts )
test_matrix = tokenizer.texts_to_matrix( test_texts )
embedding_weights = np.zeros( ( max_features , embeddings_dim ) )
for word,index in tokenizer.word_index.items():
  if index < max_features:
    try: embedding_weights[index,:] = embeddings[word]
    except: embedding_weights[index,:] = np.random.rand( 1 , embeddings_dim )


print ("")
print ("Method = Linear ridge regression with bag-of-words features")
model = KernelRidge( kernel='linear' )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
if not(is_geocoding): 
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else: 
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )

#print ("")
#print ("Method = KNN with word mover's distance as described in 'From Word Embeddings To Document Distances'")
#model = WordMoversKNN(W_embed=embedding_weights, n_neighbors=3)
#model.fit( train_matrix , train_labels )
#results = model.predict( test_matrix )
#if not(is_geocoding):  
#  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
#  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
#else:
#  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
#  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = MLP with bag-of-words features")
np.random.seed(0)
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(reg_dimensions, activation='sigmoid'))
if not(is_geocoding): model.compile(loss='mean_absolute_error', optimizer='adam')
else: model.compile(loss=geoloss, optimizer='adam')
model.fit( train_matrix , train_labels , nb_epoch=10, batch_size=16)
results = model.predict( test_matrix )
if not(is_geocoding):
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = Stack of two LSTMs")
np.random.seed(0)
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(output_dim=embeddings_dim , activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(reg_dimensions))
model.add(Activation('sigmoid'))
if not(is_geocoding): model.compile(loss='mean_absolute_error', optimizer='adam')
else: model.compile(loss=geoloss, optimizer='adam')  
model.fit( train_sequences , train_labels , nb_epoch=10, batch_size=16)
results = model.predict( test_sequences )
if not(is_geocoding): 
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )  
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )  
else: 
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )

print ("")
print ("Method = CNN from the paper 'Convolutional Neural Networks for Sentence Classification'")
np.random.seed(0)
nb_filter = embeddings_dim
model = Graph()
model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
model.add_node(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=False, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(Dropout(0.25), name='dropout_embedding', input='embedding')
for n_gram in [3, 5, 7]:
    model.add_node(Convolution1D(nb_filter=nb_filter, filter_length=n_gram, border_mode='valid', activation='relu', subsample_length=1, input_dim=embeddings_dim, input_length=max_sent_len), name='conv_' + str(n_gram), input='dropout_embedding')
    model.add_node(MaxPooling1D(pool_length=max_sent_len - n_gram + 1), name='maxpool_' + str(n_gram), input='conv_' + str(n_gram))
    model.add_node(Flatten(), name='flat_' + str(n_gram), input='maxpool_' + str(n_gram))
model.add_node(Dropout(0.25), name='dropout', inputs=['flat_' + str(n) for n in [3, 5, 7]])
model.add_node(Dense(reg_dimensions, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
model.add_output(name='output', input='sigmoid')
if not(is_geocoding): model.compile(loss={'output': 'mean_absolute_error'}, optimizer='adam')
else: model.compile(loss={'output': geoloss}, optimizer='adam') 
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=16, nb_epoch=10)
results = np.array(model.predict({'input': test_sequences}, batch_size=16)['output'])
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = Bidirectional LSTM")
np.random.seed(0)
model = Graph()
model.add_input(name='input', input_shape=(max_sent_len,), dtype=int)
model.add_node(Embedding( max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ), name='embedding', input='input')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True), name='forward1', input='embedding')
model.add_node(Dropout(0.25), name="dropout1", input='forward1')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid'), name='forward2', input='forward1')
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True, return_sequences=True), name='backward1', input='embedding')
model.add_node(Dropout(0.25), name="dropout2", input='backward1') 
model.add_node(LSTM(embeddings_dim, activation='sigmoid', inner_activation='hard_sigmoid', go_backwards=True), name='backward2', input='backward1')
model.add_node(Dropout(0.25), name='dropout', inputs=['forward2', 'backward2'])
model.add_node(Dense(reg_dimensions, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')
if not(is_geocoding): model.compile(loss={'output': 'mean_absolute_error'}, optimizer='adam')
else: model.compile(loss={'output': geoloss}, optimizer='adam')
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=16, nb_epoch=10)
results = np.array(model.predict({'input': test_sequences}, batch_size=16)['output'])
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = CNN-LSTM")
np.random.seed(0)
filter_length = 3
nb_filter = embeddings_dim
pool_length = 2
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, weights=[embedding_weights]))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length, border_mode='valid', activation='relu', subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(embeddings_dim))
model.add(Dense(reg_dimensions))
model.add(Activation('sigmoid'))
if not(is_geocoding): model.compile(loss='mean_absolute_error', optimizer='adam')
else: model.compile(loss=geoloss, optimizer='adam')  
model.fit( train_sequences , train_labels , nb_epoch=10, batch_size=16)
results = model.predict( test_sequences )
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = Linear ridge regression with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = LabeledLineSentence( train_texts + test_texts )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try : model[w] = embeddings[w]
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = KernelRidge( kernel='linear' )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = Kernel ridge regression with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = LabeledLineSentence( train_texts + test_texts )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try : model[w] = embeddings[w]
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = KernelRidge( kernel='rbf' )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )  

print ("")
print ("Method = MLP with doc2vec features")
np.random.seed(0)
class LabeledLineSentence(object):
  def __init__(self, data ): self.data = data
  def __iter__(self):
    for uid, line in enumerate( self.data ): yield TaggedDocument( line.split(" ") , ["S_%s" % uid] )
model = Doc2Vec( alpha=0.025 , min_alpha=0.025 )
sentences = train_texts + test_texts
sentences = LabeledLineSentence( sentences )
model.build_vocab( sentences )
model.train( sentences )
for w in model.vocab.keys():
  try : model[w] = embeddings[w]
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_rep.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(reg_dimensions, activation='sigmoid'))
if not(is_geocoding): model.compile(loss='mean_absolute_error', optimizer='adam')
else: model.compile(loss=geoloss, optimizer='adam')
model.fit( train_rep , train_labels , nb_epoch=10, batch_size=16)
results = model.predict( test_rep )
if not(is_geocoding):  
  print ("RMSE = " + repr( np.sqrt(mean_squared_error( test_labels , results )) ) )
  print ("MAE = " + repr( mean_absolute_error( test_labels , results ) ) )
else:  
  print ("Mean error = " + repr( np.mean( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) )
  print ("Median error = " + repr( np.median( [ geodistance( results[i] , test_labels[i] ) for i in range(results.shape[0]) ] ) ) ) 
