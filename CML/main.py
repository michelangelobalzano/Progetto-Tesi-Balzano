import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from training_methods import LOSO, LNSO, KF
from feature_extraction import remove_neutrals
from options import Options

def main(config):

    if config['model'] == 'xgb':
        model = xgb.XGBClassifier(max_depth=config['xgb_max_depth'], 
                                  n_estimators=config['n_estimators'], 
                                  learning_rate=config['learning_rate'])
    elif config['model'] == 'knn':
        model = KNeighborsClassifier(metric=config['knn_metric'], 
                                     n_neighbors=config['knn_n_neighbors'], 
                                     weights=config['knn_weights'])
    elif config['model'] == 'rf':
        model = RandomForestClassifier(max_depth=config['rf_max_depth'], 
                                       min_samples_leaf=config['rf_min_samples_leaf'], 
                                       min_samples_split=config['rf_min_samples_split'], 
                                       n_estimators=config['rf_n_estimators'], 
                                       random_state=42)
    elif config['model'] == 'dt':
        model = DecisionTreeClassifier(criterion=config['dt_criterion'], 
                                       max_depth=config['dt_max_depth'], 
                                       min_samples_leaf=config['dt_min_samples_leaf'], 
                                       min_samples_split=config['dt_min_samples_split'], 
                                       splitter=config['dt_splitter'], 
                                       random_state=42)

    # Estrazione delle features (Commentare se già effettuata)
    features_df = pd.read_csv('CML\\features.csv', header='infer')

    # Rimozione etichette neutral dell'etichetta da predire
    features_df = remove_neutrals(features_df.copy(), config['label'])

    # Classificazione
    X = features_df.drop(['segment_id', 'valence', 'arousal', 'user_id'], axis=1)
    y = features_df[config['label']]
    groups = features_df['user_id']

    if config['split_type'] == 'LOSO':
        acc, prec, rec, f1 = LOSO(model, X, y, groups)
    elif config['split_type'] == 'L2SO':
        acc, prec, rec, f1 = LNSO(model, X, y, groups, 2)
    elif config['split_type'] == 'L3SO':
        acc, prec, rec, f1 = LNSO(model, X, y, groups, 3)
    elif config['split_type'] == 'KF5':
        acc, prec, rec, f1 = KF(model, X, y, 5)
    elif config['split_type'] == 'KF10':
        acc, prec, rec, f1 = KF(model, X, y, 10)

    print('Risultati classificazione:')
    print('Accuracy: ', acc)
    print('Precision: ', prec)
    print('Recall: ', rec)
    print('F1-score: ', f1)

args = Options().parse()
config = args.__dict__

main(config)