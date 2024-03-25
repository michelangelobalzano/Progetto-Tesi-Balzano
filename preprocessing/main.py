signals = ['BVP', 'EDA', 'HR'] # Segnali da considerare per la classificazione
labeled_signals = ['EDA', 'HR']
df_names = ['data1, data2, data3, data4, data5, data6, data7, data8, data9'] # Nomi dei dataset non etichettati
labeled_df_names = ['labeled_data'] # Nomi dei dataset etichettati
data_directory = 'data\\' # Input e output directory
w_size = 60 # Window size per segmentazione
w_step_size = 10 # Step size per segmentazione
target_freq = 4 # Frequenza di ricampionamento dei segnali
user_max_segments = 500 # Numero massimo di segmenti per utente

from data1_preprocessing import data1_preprocessing
from labeled_data_preprocessing import labeled_data_preprocessing
from merge_and_normalization import normalization

#data1_preprocessing(data_directory=data_directory, df_name='data1', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size, user_max_segments=user_max_segments)

#labeled_data_preprocessing(data_directory=data_directory, df_name='labeled_data', signals=labeled_signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size, user_max_segments=user_max_segments)

normalization(data_directory, labeled_df_names, signals, labeled=True)