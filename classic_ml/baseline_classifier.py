import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extraction import remove_neutrals
from imblearn.over_sampling import SMOTE

label = 'arousal'
labels = ['valence', 'arousal']
remove_neutral_data = True

features_df = pd.read_csv('classic_ml\\features.csv', header='infer')
if remove_neutral_data:
    features_df = remove_neutrals(features_df.copy(), label)

X = features_df.drop(['valence', 'arousal'], axis=1)
y = features_df[label]

minority_class = y.value_counts().idxmin()

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

majority_label = np.bincount(y_resampled).argmax()

y_pred_majority = np.full_like(y_resampled, majority_label)

accuracy = accuracy_score(y_resampled, y_pred_majority)
precision = precision_score(y_resampled, y_pred_majority, average='weighted', zero_division=1)
recall = recall_score(y_resampled, y_pred_majority, average='weighted', zero_division=1)
f1 = f1_score(y_resampled, y_pred_majority, average='weighted', zero_division=1)

print(f'Majority class label: {majority_label}')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')