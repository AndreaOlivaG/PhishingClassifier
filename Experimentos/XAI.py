import os
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_phishing_1.0.csv")
df.replace({'status': {'phishing': 1, 'legitimate': 0}},  inplace=True)
y = df['status'].values
X = df.drop(['status', 'url'], axis=1)
for col in X.columns:
    X[col] = MinMaxScaler().fit_transform(np.array(X[col]).reshape(-1, 1))

train_urls, test_urls, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=43)

rf = RandomForestClassifier(random_state=0)
rf.fit(train_urls, train_labels)
rf_predictions = rf.predict(test_urls)
print("Macro f1-score %.4f" % f1_score(test_labels, rf_predictions, average='macro'))
print("Accuracy %.4f" % accuracy_score(test_labels, rf_predictions))

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(train_urls)
shap.summary_plot(shap_values[1], train_urls, plot_type="dot", show=False, max_display=20)
plt.tight_layout()
plt.savefig('shap_summary_plot.pdf')

plt.clf()


def make_shap_waterfall_plot(shap_values, features, num_display=20):
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]

    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 1

    fig, ax1 = plt.subplots(figsize=(8, 10.3 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)

    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1) + 1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1) + 1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))


make_shap_waterfall_plot(shap_values[1], train_urls, 20)
plt.tight_layout()
plt.savefig('shap_waterfall_plot.pdf')

make_shap_waterfall_plot(shap_values[1], train_urls)
