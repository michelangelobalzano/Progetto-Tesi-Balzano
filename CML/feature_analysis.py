import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['valence', 'arousal']

features = pd.read_csv('CML\\features_NOT_STD.csv', header='infer')
X = features.drop(['segment_id', 'valence', 'arousal'], axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for i, label in enumerate(labels):
    y = features[label]

    correlations = []
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations.append((col, corr))
    correlation_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
    correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

    sns.barplot(x='Correlation', y='Feature', data=correlation_df, ax=axes[i], palette='coolwarm')
    axes[i].set_title(f'Correlazione delle features con {label.capitalize()}')

plt.tight_layout()
plt.show()