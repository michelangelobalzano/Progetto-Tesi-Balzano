import os
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import necessary_signals, structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments

####################################################################################################################
# DATASET 2: dataset composto da 27 utenti e registrazioni suddivise per data. 
# In ogni data possono esserci più registrazioni della stessa persona.
####################################################################################################################

# Nome del dataset
dataset_name = 'data2'
# Directory del dataset
data_directory = f'data\{dataset_name}\\'
# Id degli utenti
days = ['20190902', '20190903', '20190904', '20190905', '20190906',
         '20190909', '20190910', '20190911', '20190912', '20190913',
         '20191028', '20191029', '20191030', '20191031', '20191101',
         '20191118', '20191119', '20191120', '20191121', '20191122']
min_seconds = 600 # (10 minuti) tempo minimo di registrazioni valide




def get_users(data_directory, df_name):

    users = set()
    for day in days:
        directory = data_directory + df_name + '\\' + day + '\\'
        for user_directory in os.listdir(directory):
            user_id = user_directory.split('_')[1]
            users.add(user_id)

    return list(users)



####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da una lista di dataset per ogni utente
####################################################################################################################
def read_sensor_data(data_directory, df_name, signals):

    data = {}
    
    # Determinazione della lista degli utenti
    users = get_users(data_directory, df_name)

    # Creazione struttura dati
    for user_id in users:
        data[user_id] = {}
        for signal in set(signals) | set(necessary_signals):
            data[user_id][signal] = []

    progress_bar = tqdm(total=len(days), desc="Data reading")
    for day in days:

        directory = data_directory + df_name + '\\' + day + '\\'

        for user_directory in os.listdir(directory):

            user_id = user_directory.split('_')[1]

            file_path = os.path.join(directory, user_directory)
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

        progress_bar.update(1)
    progress_bar.close()

    return data, users










####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
# Lettura del dataset
def data2_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size, user_max_segments):
    
    # Lettura del dataset e della lista degli utenti
    data, users = read_sensor_data(data_directory, df_name, signals)

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
            for signal in signals:
                data_temp[signal] = data[user_id][signal][i]
            data_temp = off_body_detection(data_temp, signals)
            data_temp = sleep_detection(data_temp, signals)
            for signal in signals:
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