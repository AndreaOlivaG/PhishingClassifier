import os
import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

df = pd.read_csv("dataset_phishing_1.0.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}},  inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)
for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

train_urls, test_urls, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=43)

clf = RandomForestClassifier(random_state=0)
model = clf.fit(train_urls, train_labels)
rf_predictions = model.predict(test_urls)

lr = LogisticRegression(max_iter=1000, random_state=0)
lr.fit(train_urls, train_labels)
lr_predictions = lr.predict(test_urls)

clf = svm.SVC(random_state=0)
model = clf.fit(train_urls, train_labels)
svm_predictions = model.predict(test_urls)

total_pred = np.zeros(len(test_urls), dtype=int)
for i in range(len(test_urls)):
    if lr_predictions[i] + rf_predictions[i] + svm_predictions[i] >= 2:
        total_pred[i] = 1
    else:
        total_pred[i] = 0

print("VOTING")
print("Macro f1-score %.4f" % f1_score(test_labels, total_pred, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, total_pred))
