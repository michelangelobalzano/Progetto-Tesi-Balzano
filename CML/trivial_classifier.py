import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extraction import remove_neutrals
from options import Options

def trivial_classification(config):

    # Lettura dataframe delle features
    features_df = pd.read_csv('CML\\features.csv', header='infer')

    # Rimozione etichette neutral dell'etichetta da predire
    features_df = remove_neutrals(features_df.copy(), config['label'])

    # Classificazione
    y = features_df[config['label']]
    # Determinazione etichetta di maggioranza
    majority_label = np.bincount(y).argmax()
    # Predizioni
    y_pred_majority = np.full_like(y, majority_label)

    accuracy = accuracy_score(y, y_pred_majority)
    precision = precision_score(y, y_pred_majority, average='weighted', zero_division=1)
    recall = recall_score(y, y_pred_majority, average='weighted', zero_division=1)
    f1 = f1_score(y, y_pred_majority, average='weighted', zero_division=1)

    print(f'Majority class label: {majority_label}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

args = Options().parse()
config = args.__dict__

trivial_classification(config)