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
from data2_preprocessing import data2_preprocessing
from data3_preprocessing import data3_preprocessing
from data4_preprocessing import data4_preprocessing
from data5_preprocessing import data5_preprocessing
from data6_preprocessing import data6_preprocessing
from data7_preprocessing import data7_preprocessing
from data8_preprocessing import data8_preprocessing
from data9_preprocessing import data9_preprocessing
from labeled_data_preprocessing import labeled_data_preprocessing
from merge_and_normalization import normalization

# Preprocessing dei dataset non etichettati
#data1_preprocessing(data_directory=data_directory, df_name='data1', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data2_preprocessing(data_directory=data_directory, df_name='data2', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data3_preprocessing(data_directory=data_directory, df_name='data3', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data4_preprocessing(data_directory=data_directory, df_name='data4', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data5_preprocessing(data_directory=data_directory, df_name='data5', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data6_preprocessing(data_directory=data_directory, df_name='data6', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data7_preprocessing(data_directory=data_directory, df_name='data7', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data8_preprocessing(data_directory=data_directory, df_name='data8', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)
#data9_preprocessing(data_directory=data_directory, df_name='data9', signals=signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)

# Preprocessing dei dataset etichettati
labeled_data_preprocessing(data_directory=data_directory, df_name='labeled_data', signals=labeled_signals, target_freq=target_freq, w_size=w_size, w_step_size=w_step_size)

#normalization(data_directory, ['data1'], signals, labeled=False)
normalization(data_directory, labeled_df_names, signals, labeled=True)