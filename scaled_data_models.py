import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

print("DECISION TREE")
dt = tree.DecisionTreeClassifier(random_state=seed)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
print("Macro f1-score %.4f" % f1_score(y_test, dt_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, dt_predictions))

print("\nRANDOM FOREST")
rf = RandomForestClassifier(random_state=seed)
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)
print("Macro f1-score %.4f" % f1_score(y_test, rf_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, rf_predictions))

print("\nLOGISTIC REGRESSION")
lr = LogisticRegression(max_iter=100, random_state=seed, solver='liblinear')
lr.fit(X_train, y_train)
lr_predictions = lr.predict(X_test)
print("Macro f1-score %.4f" % f1_score(y_test, lr_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, lr_predictions))

print("\nNAÏVE BAYES")
nv = GaussianNB()
nv.fit(X_train, y_train)
nv_predictions = nv.predict(X_test)
print("Macro f1-score %.4f" % f1_score(y_test, nv_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, nv_predictions))

print("\nSVM")
svm = svm.SVC(random_state=seed)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
print("Macro f1-score %.4f" % f1_score(y_test, svm_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(y_test, svm_predictions))

fig, ax = plt.subplots()
dec_tree = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=dt_predictions, ax=ax, name='Decision tree')
rand_for = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=rf_predictions, ax=ax, name='Random forest')
log_reg = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=lr_predictions, ax=ax, name='Logistic regression')
nai_bay = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=nv_predictions, ax=ax, name='Naïve Bayes')
sup_vec = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=svm_predictions, ax=ax, name='SVM')
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.savefig("roc_curve_scaled.png")
