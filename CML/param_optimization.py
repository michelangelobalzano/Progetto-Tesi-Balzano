from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from datetime import datetime

from feature_extraction import feature_extraction, remove_neutrals

label = 'valence'
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

# Estrazione delle features (Commentare se già effettuata)
'''feature_extraction()'''

# Lettura dataframe delle features
features_df = pd.read_csv('CML\\features.csv', header='infer')

# Rimozione etichette neutral dell'etichetta da predire
features_df = remove_neutrals(features_df.copy(), label)

# Classificazione
X = features_df.drop(['segment_id', 'valence', 'arousal', 'user_id'], axis=1)
y = features_df[label]

current_datetime = datetime.now()
run_name = current_datetime.strftime("%m-%d_%H-%M")
with open(f'CML\\results\\{label}_optimization_{run_name}.txt', 'w') as file:
    for model in models.keys():
        print(f'hyperparameters search for model {model}...')
        grid_search = GridSearchCV(estimator=models[model], param_grid=grids[model], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        print(f"Best hyperparameters values: {best_params}")
        file.write(f"{model}: {best_params}\n")