import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

ff = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16, )),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

urldata = pd.read_csv("https://cainvas-static.s3.amazonaws.com/media/user_data/cainvas-admin/urldata.csv")

#training_data
x = urldata[['hostname_length','path_length', 'fd_length', 'count-', 'count@', 'count?','count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits','count-letters', 'count_dir', 'use_of_ip']]

#label data
y = urldata['result']
y = y.values.ravel()

# splitting dataset
x_train = x[:int((1-0.2)*len(y))]
x_val = x[int((1-0.2)*len(y)):int((1-0.1)*len(y))]
x_test = x[int((1-0.1)*len(y)):]

y_train = y[:int((1-0.2)*len(y))]
y_val = y[int((1-0.2)*len(y)):int((1-0.1)*len(y))]
y_test = y[int((1-0.1)*len(y)):]

optim = tf.keras.optimizers.Adam(learning_rate=0.0001)
ff.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc'])

history = ff.fit(x_train, y_train, batch_size=256, epochs=10, validation_data=(x_val, y_val))
pred_test = ff.predict(x_test)

