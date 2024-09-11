import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

seed = 43
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)

# Load dataset
df = pd.read_csv("data/dataset_phishing.csv")

# Replace categorical labels with numerical values
df.replace({'status': {'phishing': 1, 'legitimate': 0}}, inplace=True)

# Split features and target variable
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)

# Scale features
for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Create directories for saving figures
train_phishing_dir = "data/figures/train/phishing/"
train_legitimate_dir = "data/figures/train/legitimate/"
test_phishing_dir = "data/figures/test/phishing/"
test_legitimate_dir = "data/figures/test/legitimate/"

for directory in [train_phishing_dir, train_legitimate_dir, test_phishing_dir, test_legitimate_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to save heatmaps
def save_heatmaps(X_data, y_data, output_dir):
    for i in range(len(X_data)):
        data = np.asarray(X_data.iloc[i]).reshape(-1, 1)
        sns.heatmap(data, cbar=False, xticklabels=False, yticklabels=False, annot=False)

        # Generate filename with leading zeros
        filename = f"{i:04d}.png"

        if y_data[i] == 1:
            plt.savefig(os.path.join(output_dir, "phishing", filename))
        else:
            plt.savefig(os.path.join(output_dir, "legitimate", filename))
        plt.close()


# Save heatmaps for training and test sets
save_heatmaps(X_train, y_train, "data/figures/train")
save_heatmaps(X_test, y_test, "data/figures/test")
