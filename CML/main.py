import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tqdm import tqdm
from datetime import datetime

from training_methods import LOSO, LNSO, KF
from feature_extraction import feature_extraction, remove_neutrals

label = 'valence' # Etichetta da classificare
results_list = [] # Lista dei risultati
models = {
    'valence': {
        'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.01),
        'knn': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='uniform'),
        'rf': RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=50, random_state=42),
        'dt': DecisionTreeClassifier(criterion='gini', max_depth=30, min_samples_leaf=1, min_samples_split=10, splitter='random', random_state=42)
    },
    'arousal': {
        'xgb': xgb.XGBClassifier(max_depth=10, n_estimators=50, learning_rate=0.01),
        'knn': KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='distance'),
        'rf': RandomForestClassifier(max_depth=30, min_samples_leaf=1, min_samples_split=10, n_estimators=100, random_state=42),
        'dt': DecisionTreeClassifier(criterion='gini', max_depth=20, min_samples_leaf=10, min_samples_split=20, splitter='random', random_state=42)
    }
} # Modelli utilizzati con iperparametri ottimizzati per label

# Estrazione delle features (Commentare se già effettuata)
'''feature_extraction()'''

# Lettura dataframe delle features
features_df = pd.read_csv('CML\\features.csv', header='infer')

# Rimozione etichette neutral dell'etichetta da predire
features_df = remove_neutrals(features_df.copy(), label)

# Classificazione
X = features_df.drop(['segment_id', 'valence', 'arousal', 'user_id'], axis=1)
y = features_df[label]
groups = features_df['user_id']
for model_name, model in tqdm(models[label].items(), desc='Processing per model', leave=False):
    # Leave One Subject Out
    acc, prec, rec, f1 = LOSO(model, X, y, groups)
    results_list.append({
        'model': model_name,
        'val_type': 'LOSO',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    # Leave 2 Subject Out
    acc, prec, rec, f1 = LNSO(model, X, y, groups, 2)
    results_list.append({
        'model': model_name,
        'val_type': 'L2SO',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    # Leave 3 Subject Out
    acc, prec, rec, f1 = LNSO(model, X, y, groups, 3)
    results_list.append({
        'model': model_name,
        'val_type': 'L3SO',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    # K-fold Cross Validation con k = 5
    acc, prec, rec, f1 = KF(model, X, y, 5)
    results_list.append({
        'model': model_name,
        'val_type': 'KF(5)',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })
    # K-fold Cross Validation con k = 10
    acc, prec, rec, f1 = KF(model, X, y, 10)
    results_list.append({
        'model': model_name,
        'val_type': 'KF(10)',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

# Esportazione dei risultati
current_datetime = datetime.now()
run_name = current_datetime.strftime("%m-%d_%H-%M")
results_df = pd.DataFrame(results_list)
results_df.to_csv(f'CML\\results\\{label}_{run_name}.csv', index=False)