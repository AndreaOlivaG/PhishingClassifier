import os
import random

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

seed = 43
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

df = pd.read_csv("data/dataset_phishing.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}}, inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)

for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

rf = RandomForestClassifier(random_state=seed)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

lr = LogisticRegression(max_iter=100, random_state=seed, solver='liblinear')
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)

svm = svm.SVC(random_state=seed)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

total_pred = np.zeros(len(X_test), dtype=int)
for i in range(len(y_test)):
    if lr_predictions[i] + rf_predictions[i] + svm_predictions[i] >= 2:
        total_pred[i] = 1
    else:
        total_pred[i] = 0

print("VOTING")
print("Macro f1-score %.4f" % f1_score(y_test, total_pred, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, total_pred))
