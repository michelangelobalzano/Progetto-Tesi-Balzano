from WESAD_data_preprocessing import WESAD_preprocessing
from options import Options

def main(config):

    # Preprocessing dei dataset etichettati
    WESAD_preprocessing(config)

args = Options().parse()
config = args.__dict__

config['data_directory'] = 'WESAD\\' # Directory del dataset WESAD
config['signals'] = ['BVP', 'EDA', 'HR'] # Segnali considerati
config['signals_to_process'] = config['signals'].remove('BVP') # Segnali considerati

main(config)