import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extraction import remove_neutrals

label = 'arousal'
labels = ['valence', 'arousal']
remove_neutral_data = True

features_df = pd.read_csv('classic_ml\\features.csv', header='infer')
if remove_neutral_data:
    features_df = remove_neutrals(features_df.copy(), label)

y = features_df[label]

majority_label = np.bincount(y).argmax()

y_pred_majority = np.full_like(y, majority_label)

accuracy = accuracy_score(y, y_pred_majority)
precision = precision_score(y, y_pred_majority, average='macro', zero_division=1)
recall = recall_score(y, y_pred_majority, average='macro', zero_division=1)
f1 = f1_score(y, y_pred_majority, average='macro', zero_division=1)

print(f'Majority class label: {majority_label}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')