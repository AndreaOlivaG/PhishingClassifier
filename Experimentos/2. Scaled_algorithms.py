import os
import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
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

print("DECISION TREE")
dt = tree.DecisionTreeClassifier(random_state=0)
dt.fit(train_urls, train_labels)
dt_predictions = dt.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, dt_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, dt_predictions))

print("\nRANDOM FOREST")
rf = RandomForestClassifier(random_state=0)
rf.fit(train_urls, train_labels)
rf_predictions = rf.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, rf_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, rf_predictions))

print("\nLOGISTIC REGRESSION")
lr = LogisticRegression(max_iter=1000, random_state=0)
lr.fit(train_urls, train_labels)
lr_predictions = lr.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, lr_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, lr_predictions))

print("\nNAÏVE BAYES")
clf = GaussianNB()
model = clf.fit(train_urls, train_labels)
nv_predictions = model.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, nv_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, nv_predictions))

print("\nSVM")
clf = svm.SVC(random_state=0)
model = clf.fit(train_urls, train_labels)
svm_predictions = model.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, svm_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, svm_predictions))

"""
fig, ax = plt.subplots()
dec_tree = metrics.RocCurveDisplay.from_predictions(y_true=test_labels, y_pred=dt_predictions, ax=ax, name='Decision tree')
rand_for = metrics.RocCurveDisplay.from_predictions(y_true=test_labels, y_pred=rf_predictions, ax=ax, name='Random forest')
log_reg = metrics.RocCurveDisplay.from_predictions(y_true=test_labels, y_pred=lr_predictions, ax=ax, name='Logistic regression')
nai_bay = metrics.RocCurveDisplay.from_predictions(y_true=test_labels, y_pred=nv_predictions, ax=ax, name='Naïve Bayes')
sup_vec = metrics.RocCurveDisplay.from_predictions(y_true=test_labels, y_pred=svm_predictions, ax=ax, name='SVM')
plt.ylabel("Tasa de verdaderos positivos (Positivo: phishing)")
plt.xlabel("Tasa de falsos positivos (Positivo: phishing)")
#plt.savefig("ROC_Curve_Exp2.pdf")
plt.show()
"""
