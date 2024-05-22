from WESAD_data_preprocessing import WESAD_preprocessing
from merge_and_normalization import merge_and_normalize

signals = ['BVP', 'EDA', 'HR'] # Segnali da considerare per la classificazione
labeled_signals = ['EDA', 'HR']
w_size = 60 # Window size per segmentazione
w_step_size = 10 # Step size per segmentazione
target_freq = 4 # Frequenza di ricampionamento dei segnali
min_seconds = 600 # (10 minuti) tempo minimo di registrazioni valide
w_step_size = 10 # Step size per i dati etichettati

# Preprocessing dei dataset etichettati
WESAD_preprocessing(signals=labeled_signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)