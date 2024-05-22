import argparse

class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='Run di addestramento.')
        
        # Percorsi
        self.parser.add_argument('--data_path', type=str, default='processed_data\\',
                                 help='Percorso directory dei dati preprocessati.')
        self.parser.add_argument('--info_path', type=str, default='sessions\\',
                                 help='Percorso directory salvataggio info sessioni.')
        self.parser.add_argument('--model_path', type=str, default='training\\pretrained_models\\',
                                 help='Percorso directory salvataggio modelli preaddestrati.')
        # Preparazione dati
        self.parser.add_argument('--val_ratio', type=int, default=15,
                                 help='Percorso directory salvataggio info sessioni.')
        self.parser.add_argument('--test_ratio', type=int, default=15,
                                 help='Percorso directory salvataggio modelli preaddestrati.')
        self.parser.add_argument('--split_type', choices={'LOSO', 'L2SO', 'L3SO'},
                                 help='Tipo di split dei dati in train/val/test.')
        self.parser.add_argument('--remove_neutral_data', action='store_true',
                                 help='Se impostato, rimuove i segmenti etichettati come neutral.')
        # Opzioni di addestramento
        self.parser.add_argument('--task', required=True, choices={'pretraining', 'classification'}, type=str,
                                 help='Task di training da eseguire.')
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
        self.parser.add_argument('--model_to_load', type=str, default='',
                                 help='Nome modello preaddestrato da caricare (solo per classificazione).')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='Se impostato, congela tutti i parametri del modello eccetto l output layer (solo per classificazione).')
        self.parser.add_argument('--label', choices={'valence', 'arousal'}, default='valence',
                                 help='Etichetta da predire (solo per classificazione).')
        self.parser.add_argument('--num_optimization_trials', type=int, default=20,
                                 help='Numero di tentativi di ottimizzazione iperaparametri.')
        self.parser.add_argument('--num_optimization_epochs', type=int, default=20,
                                 help='Numero di epoche per trial di ottimizzazione iperparametri.')
        # Parametri modello
        self.parser.add_argument('--batch_size', type=int, default=256,
                                 help='Numero segmenti di un batch.')
        self.parser.add_argument('--d_model', type=int, default=64,
                                 help='Dimensione interna del modello.')
        self.parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimensione feedforward network.')
        self.parser.add_argument('--dropout', type=float, default=0.25,
                                 help='Percentuale spegnimento neuroni.')
        self.parser.add_argument('--num_heads', type=int, default=2,
                                 help='Numero di teste modulo auto-attenzione.')
        self.parser.add_argument('--num_layers', type=int, default=3,
                                 help='Numero layers dell encoder.')
        self.parser.add_argument('--pe_type', choices={'fixed', 'learnable'}, default='learnable',
                                 help='Tipo di positional encoding.')
        # Parametri mascheramento
        self.parser.add_argument('--masking_ratio', type=float, default=0.15,
                                 help='Rapporto di valori mascherati.')
        self.parser.add_argument('--lm', type=int, default=3,
                                 help='Lunghezza sezioni mascherate.')
        
    def parse(self):

        args = self.parser.parse_args()

        return args