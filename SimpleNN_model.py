from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

def simpleNN(input_dim, nb_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model