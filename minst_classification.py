from tensorflow import keras
from tensorflow.keras import utils
import SimpleNN_model
from sklearn.preprocessing import StandardScaler
import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M")
    return os.path.join(root_logdir, run_id)

import pandas as pd
import numpy as np

# train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

(X_train_full, y_train_full), (_, _) = keras.datasets.mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = test / 255.
y_train = utils.to_categorical(y_train)
y_valid = utils.to_categorical(y_valid)

X_train = X_train.reshape((X_train.shape[0], -1))
X_valid = X_valid.reshape((X_valid.shape[0], -1))
input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

simple_nn = SimpleNN_model.simpleNN(input_dim, nb_classes)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
print("Training...")
simple_nn.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[early_stopping_cb, tensorboard_cb])

preds = simple_nn.predict_classes(X_test, verbose=0)

pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv("submission.csv", index=False)
