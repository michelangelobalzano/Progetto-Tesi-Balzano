import argparse

class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='Ottimizzazione iperparametri CML.')
        
        self.parser.add_argument('--label', choices={'valence', 'arousal'}, required=True,
                                 help='Etichetta del quale ottimizzare gli iperparametri.')
        self.parser.add_argument('--model', choices={'xgb', 'knn', 'rf', 'dt'}, required=True,
                                 help='Sigla del modello da utilizzare. xgb=XGBoost, knn=kNN, rf=random forest, dt=decision tree')
        
    def parse(self):

        args = self.parser.parse_args()

        return args