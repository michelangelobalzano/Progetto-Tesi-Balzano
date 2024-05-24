import argparse

class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='Classificazione CML.')
        
        self.parser.add_argument('--label', choices={'valence', 'arousal'}, default='valence',
                                 help='Etichetta del quale ottimizzare gli iperparametri.')
        self.parser.add_argument('--model', choices={'xgb', 'knn', 'rf', 'dt'}, default='xgb',
                                 help='Sigla del modello da utilizzare. xgb=XGBoost, knn=kNN, rf=random forest, dt=decision tree.')
        self.parser.add_argument('--split_type', choices={'LOSO', 'L2SO', 'L3SO', 'KF5', 'KF10'}, default='LOSO',
                                 help='Sigla tipo di split dei dati. LOSO=Leave One Subject Out, L2SO, L3SO=Leave 2, 3 subjects out, KF5, KF10=K-Fold Cross Validation k=5, 10.')
        
        
        self.parser.add_argument('--xgb_max_depth', type=int, choices={3, 5, 10, 20, 30}, default=3,
                                 help="Profondita' massima.")
        self.parser.add_argument('--xgb_n_estimators', type=int, choices={50, 100, 200}, default=50,
                                 help='Numero di alberi da valutare.')
        self.parser.add_argument('--xgb_learning_rate', type=float, choices={0.01, 0.1, 0.3, 0.5}, default=0.01,
                                 help='Tasso apprendimento.')
        
        
        self.parser.add_argument('--knn_n_neighbors', type=int, choices={1, 3, 5, 7, 9, 11, 13, 15}, default=3,
                                 help='Numero di vicini.')
        self.parser.add_argument('--knn_weights', choices={'uniform', 'distance'}, default='uniform',
                                 help='Metodo di peso dei vicini.')
        self.parser.add_argument('--knn_metric', choices={'euclidean', 'manhattan', 'minkowski'}, default='manhattan',
                                 help='Metrica per calcolare la distanza.')
        
        
        self.parser.add_argument('--rf_max_depth', type=int, choices={None, 10, 20, 30}, default=30,
                                 help="Profondita' massima.")
        self.parser.add_argument('--rf_n_estimators', type=int, choices={50, 100, 200}, default=50,
                                 help='Numero di alberi da valutare.')
        self.parser.add_argument('--rf_min_samples_split', type=int, choices={2, 5, 10}, default=10,
                                 help='Numero minimo di campioni richiesti per dividere un nodo interno.')
        self.parser.add_argument('--rf_min_samples_leaf', type=int, choices={1, 2, 4}, default=1,
                                 help='Numero minimo di campioni che deve avere un nodo foglia.')
        
        
        self.parser.add_argument('--dt_max_depth', type=int, choices={None, 10, 20, 30}, default=30,
                                 help="Profondita' massima.")
        self.parser.add_argument('--dt_min_samples_split', type=int, choices={2, 10, 20}, default=10,
                                 help='Numero minimo di campioni richiesti per dividere un nodo interno.')
        self.parser.add_argument('--dt_min_samples_leaf', type=int, choices={1, 5, 10}, default=1,
                                 help='Numero minimo di campioni che deve avere un nodo foglia.')
        self.parser.add_argument('--dt_criterion', choices={'gini', 'entropy'}, default='gini',
                                 help="Funzione di misurazione qualita' di una divisione.")
        self.parser.add_argument('--dt_splitter', choices={'best', 'random'}, default='random',
                                 help='Strategia di scelta della divisione.')
        
    def parse(self):

        args = self.parser.parse_args()

        return args