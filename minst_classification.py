from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import SimpleNN_model
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_x = train.iloc[:,1:].values.astype('float32')
train_y = train.iloc[:,0].values
train_y = np_utils.to_categorical(train_y) 

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

input_dim = train_x.shape[1]
nb_classes = train_y.shape[1]

simple_nn = SimpleNN_model.simpleNN(input_dim, nb_classes)

print("Training...")
simple_nn.fit(train_x, train_y, epochs=10, batch_size=10, validation_split=0.1, verbose=2)


test_x = test.values.astype('float32')

test_x = scaler.transform(test_x)

preds = simple_nn.predict_classes(test_x, verbose=0)

pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv("submission.csv", index=False)
