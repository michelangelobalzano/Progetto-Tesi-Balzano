from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from datetime import datetime

from feature_extraction import remove_neutrals
from optimization_options import Options

grids = {
    'xgb': {
        'max_depth': [3, 5, 10, 20, 30],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.3, 0.5]
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'rf': {
        'max_depth': [None, 10, 20, 30],
        'n_estimators': [50, 100, 200],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'dt': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random']
    }
} # Griglie di valori degli iperparametri da ottimizzare

models = {
    'xgb': xgb.XGBClassifier(),
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'dt': DecisionTreeClassifier()
} # Modelli utilizzati

model_names = {
    'xgb': 'XGBoost',
    'knn': 'kNN',
    'rf': 'random forest',
    'dt': 'decision tree'
}

def main(config):

    # Lettura dataframe delle features
    features_df = pd.read_csv('CML\\features.csv', header='infer')

    # Rimozione etichette neutral dell'etichetta da predire
    features_df = remove_neutrals(features_df.copy(), config['label'])

    # Classificazione
    X = features_df.drop(['segment_id', 'valence', 'arousal', 'user_id'], axis=1)
    y = features_df[config['label']]

    print(f'Ricerca iperparametri per il modello: {model_names[config["model"]]} in corso...')
    grid_search = GridSearchCV(estimator=models[config['model']], param_grid=grids[config['model']], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(f"Migliori iperparametri: {best_params}")

args = Options().parse()
config = args.__dict__

main(config)