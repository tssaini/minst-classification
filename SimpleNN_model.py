from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import optimizers

def simpleNN(input_dim, nb_classes):

    model = Sequential([
        keras.layers.Dense(300, activation="relu", input_dim=input_dim),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(nb_classes, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=["accuracy"])
    # model = Sequential()
    # model.add(Dense(300, input_dim=input_dim))
    # model.add(Activation('relu'))
    # model.add(Dense(200))
    # model.add(Activation('relu'))
    # model.add(Dense(nb_classes))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model