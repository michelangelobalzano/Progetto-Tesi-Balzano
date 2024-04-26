signals = ['BVP', 'EDA', 'HR'] # Segnali da considerare per la classificazione
labeled_signals = ['EDA', 'HR']
df_names = ['5', '6', '7', '8', '9'] # Nomi dei dataset non etichettati                    '1', '2', '3', '4', 
labeled_df_name = '10' # Nome del dataset etichettato
data_directory = 'data\\' # Input e output directory
w_size = 60 # Window size per segmentazione
w_step_size = 10 # Step size per segmentazione
target_freq = 4 # Frequenza di ricampionamento dei segnali
user_max_segments = 1000 # Numero massimo di segmenti per utente
min_seconds = 600 # (10 minuti) tempo minimo di registrazioni valide

from preprocessing import preprocessing
from labeled_data_preprocessing import labeled_data_preprocessing
from merge_and_normalization import merge_and_normalize

# Preprocessing dei dataset non etichettati
for data in ['9']:
    preprocessing(data_directory=data_directory+data+'\\', df_name=data, signals=signals, min_seconds=min_seconds, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)

# Preprocessing dei dataset etichettati
#labeled_data_preprocessing(data_directory=data_directory+labeled_df_name+'\\', df_name=labeled_df_name, signals=labeled_signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)

merge_and_normalize(data_directory, ['9'], signals, user_max_segments=user_max_segments, labeled=False)
#merge_and_normalize(data_directory, [labeled_df_name], signals, user_max_segments, labeled=True)