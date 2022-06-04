import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

cwd = os.getcwd()
df = pd.read_csv(cwd+"/../../Datasets/dataset_phishing_1.0.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}},  inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)
for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

fig, ax = plt.subplots(figsize=(3.69, 3.69))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

for i in range(len(X)):
    name = ""
    data = np.asarray(X.iloc[i]).reshape(87, 1)
    sns.heatmap(data, cbar=False, xticklabels=False, yticklabels=False, annot=False)

    for j in range(5-len(str(i))):
        name = "0" + name
    name = name + str(i)

    if y[i] == 1:
        plt.savefig("Figures/Phishing/"+name+".png")
    else:
        plt.savefig("Figures/Legitimate/"+name+".png")
