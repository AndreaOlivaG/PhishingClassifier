import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tensorflow.keras.layers import Dense, Dropout

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

df = pd.read_csv("dataset_phishing_1.0.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}},  inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)
for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

X_urls, test_urls, X_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=43)
train_urls, val_urls, train_labels, val_labels = train_test_split(X_urls, X_labels, test_size=0.2, random_state=43)

neurons = 16
max_dropout = 0.25
max_epochs = 100

model = Sequential([
  Dense(neurons, input_dim=87, activation='relu', name='input_layer'),
  Dropout(max_dropout),
  Dense(neurons, activation='relu', name='hidden_layer'),
  Dropout(max_dropout),
  Dense(1, activation='sigmoid', name='output_layer'),
])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_urls, train_labels, validation_data=[val_urls, val_labels], epochs=max_epochs, verbose=0)
predictions = model.predict(test_urls).round()

f1 = f1_score(test_labels, predictions, average='macro')
acc = accuracy_score(test_labels, predictions)
print("f1-score: %.4f, accuracy: %.4f" % (f1, acc))
