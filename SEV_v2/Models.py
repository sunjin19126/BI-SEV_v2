from keras.layers import Convolution1D,MaxPooling1D
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Embedding,LSTM
from keras.layers.core import Dense,Dropout,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# ML_Random Forest Classifier
def RFwe():

    model = ensemble.RandomForestClassifier(n_estimators=1000,n_jobs=1)

    return model

# ML_Gaussian Naive_bayes Classifier
def NBwe():

    model = GaussianNB()

    return model

# ML_Multilayer Perceptron Classifer
def MLPwe():

    model = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(100,100), max_iter=1000, random_state=1)

    return model

# ML_Support Vector Machine Classifer
def SVMwe():

    model = SVC(C = 1e-05, random_state=0, probability = True)

    return model

# ML_k Nearest Neighbor algorithm
def kNNwe():

    model = KNeighborsClassifier(weights='distance', n_neighbors=4, p=3)

    return model

# DL_Convolutional Neutral Network 1dimensional
def CNN1Dwe():

    model = Sequential()

    model.add(Convolution1D(32, 3, input_shape=(24,20), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Convolution1D(64,3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

    return model

# DL_Convolutional Neutral Network 2dimensional
def CNN2Dwe():

    model = Sequential()

    model.add(Convolution2D(32,(3,3),input_shape=(1,24,20),data_format='channels_first',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

    return model

# DL_Long-Short Term Memory neutral Network
def LSTMwe():
    model = Sequential()
    model.add(Embedding(21, 5, input_length=31))

    model.add(LSTM(32, implementation=2))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Mono-layer Neutral Network: utilized to ensemble final results from different Models
def MonoNN():
    model = Sequential()
    model.add(Dense(1,activation='sigmoid',input_shape=(3,)))
    model.compile(optimizer=Adam(), loss = 'binary_crossentropy', metrics=['accuracy'])
    return model



