import os
from os.path import join
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import necessary_signals, structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df

def get_users(data_directory):

    users = set()
    for user_directory in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, user_directory)):
            user_id = user_directory
            users.add(user_id)

    return list(users)

def read_sensor_data(data_directory, users, signals, min_seconds):

    data = {}

    # Creazione struttura dati
    for user_id in users:
        data[user_id] = {}
        for signal in set(signals) | set(necessary_signals):
            data[user_id][signal] = []

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_directory in os.listdir(data_directory):

        if os.path.isdir(os.path.join(data_directory, user_directory)):
            user_directory_path = os.path.join(data_directory, user_directory)
            user_id = user_directory

            for reg_directory in os.listdir(user_directory_path):

                reg_directory_path = os.path.join(user_directory_path, reg_directory)
                files = [f for f in os.listdir(reg_directory_path) if os.path.isfile(join(reg_directory_path, f))]

                # Considero validi i df che abbiano almeno 10 minuti di registrazione
                # Prendo i file TEMP per il calcolo (4 Hz)
                valid = True
                for file in files:
                    if(file.endswith('TEMP.csv') and (len(pd.read_csv(os.path.join(reg_directory_path, file), header=None)) < min_seconds * 4)):
                        valid = False
                        break

                # Lettura dei df validi
                if(valid):
                    for signal in set(signals) | set(necessary_signals):
                        for file in files:
                            if file.endswith(f'{signal}.csv'):
                                data[user_id][signal].append(pd.read_csv(os.path.join(reg_directory_path, file), header=None))
                                break

            progress_bar.update(1)
    progress_bar.close()

    return data

####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
def preprocessing(data_directory, df_name, signals, min_seconds, target_freq, w_size, w_step_size):
    
    users = get_users(data_directory)
    data = read_sensor_data(data_directory, users, signals, min_seconds)

    # Preprocessing dei dataframe
    progress_bar = tqdm(total=len(users), desc="Df modification")
    for user_id in users:
        dim = len(data[user_id][signals[0]])
        for i in range(dim):
            # Modifica dei dataframe
            for signal in set(signals) | set(necessary_signals):
                data[user_id][signal][i] = structure_modification(data[user_id][signal][i].copy(), signal, target_freq)
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm(total=len(users), desc="Off-body and sleep detection")
    for user_id in users:
        dim = len(data[user_id][signals[0]])
        for i in range(dim):
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
                data_temp[signal] = segmentation(data[user_id][signal][i], segment_prefix=f'{df_name}{user_id}{i}', w_size=w_size, w_step_size=w_step_size)

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

    # Eliminazione dei segmenti creati che non contengono frequency * window_size valori
    for signal in signals:
        segmented_data[signal] = segmented_data[signal].groupby('segment_id').filter(lambda x: len(x) == target_freq * w_size)

    # Eliminazione segmenti non appartenenti all'intersezione dei segment_id
    segment_set = set(segmented_data[signals[0]].groupby('segment_id').groups.keys())
    for signal in signals:
        segment_set = segment_set & set(segmented_data[signal].groupby('segment_id').groups.keys())
    for signal in signals:
        segmented_data[signal] = segmented_data[signal][segmented_data[signal]['segment_id'].isin(segment_set)]

    '''# Controllo che i segment_id dei tre dataset coincidono
    segmenti = set(segmented_data[signals[0]].groupby('segment_id').groups.keys())
    for signal in signals:
        segmenti2 = set(segmented_data[signal].groupby('segment_id').groups.keys())
        if (segmenti != segmenti2):
            break'''

    # Esportazione delle features del dataset
    for signal in signals:
        print(f"Esportazione {signal}...")
        export_df(segmented_data[signal], data_directory, signal)