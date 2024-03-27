from os.path import join
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
from preprocessing_methods import necessary_signals, structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments

####################################################################################################################
# DATASET 8: dataset composto da 11 utenti con un numero di registrazioni differente per ognuno
####################################################################################################################

# Id degli utenti
users = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
min_seconds = 600 # (10 minuti) tempo minimo di registrazioni valide


####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data(data_directory, df_name, signals):

    data = {}

    # Creazione struttura dati
    for user_id in users:
        data[user_id] = {}
        for signal in set(signals) | set(necessary_signals):
            data[user_id][signal] = []

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        directory = data_directory + df_name + '\\' + user_id + '\\'

        for reg_directory in os.listdir(directory):

            file_path = os.path.join(directory, reg_directory)
            files = [f for f in os.listdir(file_path) if os.path.isfile(join(file_path, f))]

            # Considero validi i df che abbiano almeno 10 minuti di registrazione
            # Prendo i file TEMP per il calcolo (4 Hz)
            valid = True
            for file in files:
                if(file.endswith('TEMP.csv') and (len(pd.read_csv(os.path.join(file_path, file), header=None)) < min_seconds * 4)):
                    valid = False
                    break

            # Lettura dei df validi
            if(valid):
                for signal in set(signals) | set(necessary_signals):
                    for file in files:
                        
                        if file.endswith(f'{signal}.csv'):
                            data[user_id][signal].append(pd.read_csv(os.path.join(file_path, file), header=None))
                            break

        progress_bar.update(1)
    progress_bar.close()

    return data















####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
def data8_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size):
    
    # Lettura del dataset
    data = read_sensor_data(data_directory, df_name, signals)

    # Preprocessing dei dataframe
    progress_bar = tqdm(total=len(users), desc="User preprocessing")
    for user_id in users:

        dim = len(data[user_id][signals[0]])
        for i in range(dim):

            # Modifica dei dataframe
            for signal in set(signals) | set(necessary_signals):
                data[user_id][signal][i] = structure_modification(data[user_id][signal][i].copy(), signal, target_freq)

            # Determinazione momenti di off-body e sleep
            data_temp = {}
            for signal in set(signals) | set(necessary_signals):
                data_temp[signal] = data[user_id][signal][i]
            data_temp = off_body_detection(data_temp, signals)
            data_temp = sleep_detection(data_temp, signals)
            for signal in set(signals) | set(necessary_signals):
                data[user_id][signal][i] = data_temp[signal]

        progress_bar.update(1)
    progress_bar.close()

    # Rimozione dei dati di EDA e ACC se non utili alla classificazione
    for signal in necessary_signals:
        if signal not in signals:
            for user_id in users:
                del data[user_id][signal]

    # Creazione dizionario dei df totali segmentati
    segmented_data = {}
    for signal in signals:
        segmented_data[signal] = pd.DataFrame()

    # Segmentazione dei df
    progress_bar = tqdm(total=len(users), desc="Segmentation")
    for user_id in users:

        dim = len(data[user_id][signals[0]])
        for i in range(dim):

            # Produzione dei segmenti
            data_temp = {}
            for signal in signals:
                data_temp[signal] = segmentation(data[user_id][signal][i], segment_prefix=f'{df_name}_{user_id}_{i}_', w_size=w_size, w_step_size=w_step_size)

            # Eliminazione segmenti di off-body e sleep
            data_temp = delete_off_body_and_sleep_segments(data_temp, signals)
            
            # Concatenazione dei segmenti dell'user ai segmenti totali
            for signal in signals:
                segmented_data[signal] = pd.concat([segmented_data[signal], data_temp[signal]], axis=0, ignore_index=True)

        progress_bar.update(1)
    progress_bar.close()

    # Eliminazione colonne inutili
    for signal in signals:
        segmented_data[signal] = segmented_data[signal].drop(['time','off-body', 'sleep'], axis=1)

   # Esportazione delle features del dataset
    for signal in signals:
        print(f"Esportazione {signal}...")
        export_df(segmented_data[signal], data_directory, df_name, signal)