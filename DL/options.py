import argparse

class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='Run di addestramento.')
        
        # Preparazione dati
        self.parser.add_argument('--val_ratio', type=int, default=15,
                                 help='Rapporto dello split set di validazione.')
        self.parser.add_argument('--test_ratio', type=int, default=15,
                                 help='Rapporto dello split set di test.')
        self.parser.add_argument('--split_type', choices={'LOSO', 'L2SO', 'L3SO', 'segment'}, default='segment',
                                 help='Tipo di split dei dati in train/val/test.')
        # Opzioni di addestramento
        self.parser.add_argument('--num_epochs', type=int, default=500,
                                 help='Numero epoche di addestramento.')
        self.parser.add_argument('--num_epochs_to_save', type=int, default=25,
                                 help='Numero epoche per salvataggio automatico.')
        self.parser.add_argument('--learning_rate', type=float, default=0.0001,
                                 help='Learning rate.')
        self.parser.add_argument('--patience', type=int, default=10,
                                 help='Numero epoche pazienza di non miglioramento.')
        self.parser.add_argument('--max_lr_reductions', type=int, default=3,
                                 help='Numero massimo di riduzioni del learning rate.')
        self.parser.add_argument('--factor', type=float, default=0.3,
                                 help='Fattore riduzione learning rate.')
        self.parser.add_argument('--threshold', type=float, default=0.001,
                                 help='Soglia di tolleranza non miglioramento.')
        self.parser.add_argument('--label', choices={'valence', 'arousal'}, default='valence',
                                 help='Etichetta da classificare.')
        self.parser.add_argument('--num_optimization_trials', type=int, default=20,
                                 help='Numero di tentativi di ottimizzazione iperaparametri.')
        self.parser.add_argument('--num_optimization_epochs', type=int, default=20,
                                 help='Numero di epoche per trial di ottimizzazione iperparametri.')
        # Parametri modello
        self.parser.add_argument('--batch_size', type=int, choices={16,32,64,128,256}, default=256,
                                 help='Numero segmenti di un batch.')
        self.parser.add_argument('--d_model', type=int, choices={32,64,128,256}, default=64,
                                 help='Dimensione interna del modello.')
        self.parser.add_argument('--dim_feedforward', type=int, choices={128,256,512,1024,2048}, default=256,
                                 help='Dimensione feedforward network.')
        self.parser.add_argument('--dropout', type=float, choices={0.1,0.15,0.2,0.25,0.3,0.4,0.5}, default=0.25,
                                 help='Percentuale spegnimento neuroni.')
        self.parser.add_argument('--num_heads', type=int, choices={2,4,8}, default=2,
                                 help='Numero di teste modulo auto-attenzione.')
        self.parser.add_argument('--num_layers', type=int, choices={2,3,4,5,6}, default=3,
                                 help='Numero layers dell encoder.')
        self.parser.add_argument('--pe_type', choices={'fixed', 'learnable'}, default='learnable',
                                 help='Tipo di positional encoding.')
        
    def parse(self):

        args = self.parser.parse_args()

        return args