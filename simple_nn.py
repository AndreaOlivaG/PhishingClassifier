import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

seed = 43
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

df = pd.read_csv("data/dataset_phishing.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}}, inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)

for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(87,)),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=100, verbose=0)
predictions = model.predict(X_test).round()

f1 = f1_score(y_test, predictions, average='macro')
acc = accuracy_score(y_test, predictions)
print("f1-score: %.4f, accuracy: %.4f" % (f1, acc))
