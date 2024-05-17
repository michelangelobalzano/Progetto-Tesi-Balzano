import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tqdm import tqdm

from training_methods import LOSO, LNSO, KF
from feature_extraction import process_data, do_pca

label = 'arousal'
labels = ['valence', 'arousal']
pca_components = [0,2,3,4,5,6,8,10,12,14,16,18,20]

#process_data()

results_list = []

for num_components in tqdm(pca_components, desc='Processing per PCA components', leave=False):

    features_df = pd.read_csv('classic_ml\\features.csv', header='infer')

    if num_components != 0:
        features_df = do_pca(num_components)

    X = features_df.drop(['segment_id', 'valence', 'arousal', 'user_id'], axis=1)
    y = features_df[label]
    groups = features_df['user_id']

    models = {
        'xgb': xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.01),
        'knn': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='uniform'),
        'rf': RandomForestClassifier(max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=50, random_state=42),
        'dt': DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=5, min_samples_split=2, splitter='random', random_state=42)
    }

    for model_name, model in tqdm(models.items(), desc='Processing per model', leave=False):

        acc, prec, rec, f1 = LOSO(model, X, y, groups)
        results_list.append({
            'model': model_name,
            'num_components': num_components,
            'val_type': 'LOSO',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        acc, prec, rec, f1 = LNSO(model, X, y, groups, 2)
        results_list.append({
            'model': model_name,
            'num_components': num_components,
            'val_type': 'L2SO',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        acc, prec, rec, f1 = LNSO(model, X, y, groups, 3)
        results_list.append({
            'model': model_name,
            'num_components': num_components,
            'val_type': 'L3SO',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        acc, prec, rec, f1 = KF(model, X, y, 5)
        results_list.append({
            'model': model_name,
            'num_components': num_components,
            'val_type': 'KF(5)',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        acc, prec, rec, f1 = KF(model, X, y, 10)
        results_list.append({
            'model': model_name,
            'num_components': num_components,
            'val_type': 'KF(10)',
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

results_df = pd.DataFrame(results_list)
results_df.to_csv('classic_ml\\model_results.csv', index=False)