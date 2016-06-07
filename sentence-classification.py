import numpy as np
import csv
import keras
import sklearn
import gensim
import random
import scipy
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Graph
from keras.layers.core import Dense , Dropout , Activation , Merge , Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding , LSTM
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC , SVC
from sklearn.naive_bayes import MultinomialNB
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec , TaggedDocument
from word_movers_knn import WordMoversKNN

# size of the word embeddings
embeddings_dim = 300

# maximum number of words to consider in the representations
max_features = 30000

# maximum length of a sentence
max_sent_len = 50

# percentage of the data used for model training
percent = 0.75

# number of classes
num_classes = 2

print ("")
print ("Reading pre-trained word embeddings...")
embeddings = dict( )
embeddings = Word2Vec.load_word2vec_format( "GoogleNews-vectors-negative300.bin.gz" , binary=True ) 

print ("Reading text data for classification and building representations...")
data = [ ( row["sentence"] , row["label"]  ) for row in csv.DictReader(open("test-data.txt"), delimiter='\t', quoting=csv.QUOTE_NONE) ]
random.shuffle( data )
train_size = int(len(data) * percent)
train_texts = [ txt.lower() for ( txt, label ) in data[0:train_size] ]
test_texts = [ txt.lower() for ( txt, label ) in data[train_size:-1] ]
train_labels = [ label for ( txt , label ) in data[0:train_size] ]
test_labels = [ label for ( txt , label ) in data[train_size:-1] ]
num_classes = len( set( train_labels + test_labels ) )
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
le = preprocessing.LabelEncoder( )
le.fit( train_labels + test_labels )
train_labels = le.transform( train_labels )
test_labels = le.transform( test_labels )
print "Classes that are considered in the problem : " + repr( le.classes_ )

print('Convert class vector to binary class matrix (for use with categorical_crossentropy)')
Y_train = np_utils.to_categorical(train_labels, num_classes)
Y_test = np_utils.to_categorical(test_labels, num_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print ("")
print ("Method = NB with bag-of-words features")
model = MultinomialNB( )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("")
print ("Method = Linear SVM with bag-of-words features")
model = LinearSVC( random_state=0 )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = NB-SVM with bag-of-words features")
model = MultinomialNB( fit_prior=False )
model.fit( train_matrix , train_labels )
train_matrix = np.hstack( (train_matrix, model.predict_proba( train_matrix ) ) )
test_matrix = np.hstack( (test_matrix, model.predict_proba( test_matrix ) ) )
model = LinearSVC( random_state=0 )
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
train_matrix = train_matrix[0: train_matrix.shape[0], 0: train_matrix.shape[1] - model.intercept_.shape[0] ]
test_matrix = test_matrix[0: train_matrix.shape[0], 0: test_matrix.shape[1] - model.intercept_.shape[0] ]
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = KNN with word mover's distance as described in 'From Word Embeddings To Document Distances'")
model = WordMoversKNN(W_embed=embedding_weights , n_neighbors=3)
model.fit( train_matrix , train_labels )
results = model.predict( test_matrix )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = MLP with bag-of-words features")
np.random.seed(0)
model = Sequential()
model.add(Dense(embeddings_dim, input_dim=train_matrix.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(embeddings_dim, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit( train_matrix , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_matrix )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = Stack of two LSTMs")
np.random.seed(0)
model = Sequential()
model.add(Embedding(max_features, embeddings_dim, input_length=max_sent_len, mask_zero=True, weights=[embedding_weights] ))
model.add(Dropout(0.1))
model.add(LSTM(output_dim=embeddings_dim , activation='relu', return_sequences=True, init='zero'))
model.add(Dropout(0.1))
model.add(LSTM(output_dim=embeddings_dim , activation='relu', init='zero'))
model.add(Dense(1,init='zero',activation='linear'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='RMSProp', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='RMSProp')  
model.fit( train_sequences , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

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
model.add_node(Dense(1, input_dim=nb_filter * len([3, 5, 7])), name='dense', input='dropout')
model.add_node(Activation('sigmoid'), name='sigmoid', input='dense')
model.add_output(name='output', input='sigmoid')
if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam') 
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30)
results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

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
model.add_node(Dense(1, activation='sigmoid'), name='sigmoid', input='dropout')
model.add_output(name='output', input='sigmoid')
if num_classes == 2: model.compile(loss={'output': 'binary_crossentropy'}, optimizer='adam')
else: model.compile(loss={'output': 'categorical_crossentropy'}, optimizer='adam')
model.fit({'input': train_sequences, 'output': train_labels}, batch_size=32, nb_epoch=30)
results = np.array(model.predict({'input': test_sequences}, batch_size=32)['output'])
if num_classes != 2: results = results.argmax(axis=-1)
else: results = (results > 0.5).astype('int32')
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

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
model.add(Dense(1))
model.add(Activation('sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='adam')  
model.fit( train_sequences , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_sequences )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results ) ) )
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = Linear SVM with doc2vec features")
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
  try: model[w] = embeddings[w] 
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = LinearSVC( random_state=0 )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

print ("Method = Non-linear SVM with doc2vec features")
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
  try: model[w] = embeddings[w] 
  except : continue
for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
train_rep = np.array( [ model.docvecs[i] for i in range( train_matrix.shape[0] ) ] )
test_rep = np.array( [ model.docvecs[i + train_matrix.shape[0]] for i in range( test_matrix.shape[0] ) ] )
model = SVC( random_state=0 , kernel='poly' )
model.fit( train_rep , train_labels )
results = model.predict( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))

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
  try: model[w] = embeddings[w]
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
model.add(Dense(1, activation='sigmoid'))
if num_classes == 2: model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
else: model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit( train_rep , train_labels , nb_epoch=30, batch_size=32)
results = model.predict_classes( test_rep )
print ("Accuracy = " + repr( sklearn.metrics.accuracy_score( test_labels , results )  ))
print (sklearn.metrics.classification_report( test_labels , results ))
