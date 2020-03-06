from tensorflow import keras
from tensorflow.keras import utils
import SimpleNN_model
from sklearn.preprocessing import StandardScaler
import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_x = train.iloc[:,1:].values.astype('float32')
train_y = train.iloc[:,0].values
train_y = utils.to_categorical(train_y) 

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

input_dim = train_x.shape[1]
nb_classes = train_y.shape[1]

simple_nn = SimpleNN_model.simpleNN(input_dim, nb_classes)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
print("Training...")
simple_nn.fit(train_x, train_y, epochs=20, batch_size=10, validation_split=0.1, verbose=2, callbacks=[early_stopping_cb, tensorboard_cb])


test_x = test.values.astype('float32')

test_x = scaler.transform(test_x)

preds = simple_nn.predict_classes(test_x, verbose=0)

pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv("submission.csv", index=False)
