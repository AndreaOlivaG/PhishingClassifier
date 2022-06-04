import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

model = Sequential([
    LSTM(128, input_shape=(87, 1), activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_urls, train_labels, validation_data=[val_urls, val_labels], epochs=700, batch_size=32, verbose=2)
lstm_predictions = model.predict(test_urls).round()

print("Macro f1-score: %.4f" % (f1_score(test_labels, lstm_predictions, average='macro')))
print("Accuracy: %.4f\n" % (accuracy_score(test_labels, lstm_predictions)))
