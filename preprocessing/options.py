import argparse

class Options(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            description='Preprocessing.')
        
        self.parser.add_argument('--segmentation_window_size', choices={30, 60, 120}, default=60,
                                 help='Dimensione della finestra di segmentazione (secondi).')
        self.parser.add_argument('--segmentation_step_size', choices={5, 10, 15}, default=10,
                                 help='Dimensione del passo di segmentazione (secondi).')
        self.parser.add_argument('--neutral_range', choices={0.2, 0.35, 0.5}, default=0.2,
                                 help='Range dalla media delle etichette neutral.')
        
    def parse(self):

        args = self.parser.parse_args()

        return args