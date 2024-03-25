from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments, necessary_signals

####################################################################################################################
# DATASET 1: dataset composto da 10 utenti, ognuno avente 3 registrazioni: Midterm 1, Midterm 2 e Final
####################################################################################################################

users = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
# Tipologie di dataframe
df_types = ['Midterm 1', 'Midterm 2', 'Final']





####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data(data_directory, df_name, signals):

    data = {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        data[user_id] = {}

        for df_type in df_types:

            data[user_id][df_type] = {}

            directory = data_directory + df_name + '\\' + user_id + '\\' + df_type + '\\'
            file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

            for signal in set(signals) | set(necessary_signals):
            
                for file in file_path:

                    if file.endswith(f'{signal}.csv'):
                        data[user_id][df_type][signal] = pd.read_csv(join(directory, file), header=None)
                        break

        progress_bar.update(1)
    progress_bar.close()

    return data





####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
def data1_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size, user_max_segments): 

    # Lettura del dataset
    data = read_sensor_data(data_directory, df_name, signals)

    # Preprocessing dei dataframe
    progress_bar = tqdm(total=len(users), desc="User preprocessing")
    for user_id in users:

        for df_type in df_types:

            # Modifica dei dataframe
            for signal in set(signals) | set(necessary_signals):

                data[user_id][df_type][signal] = structure_modification(data[user_id][df_type][signal].copy(), signal, target_freq)

            # Determinazione momenti di off-body e sleep
            data[user_id][df_type] = off_body_detection(data[user_id][df_type], signals)
            data[user_id][df_type] = sleep_detection(data[user_id][df_type], signals)

        progress_bar.update(1)
    progress_bar.close()

    # Rimozione dei dati di EDA e ACC se non utili alla classificazione
    for signal in necessary_signals:
        if signal not in signals:
            for user_id in users:
                for df_type in df_types:
                    del data[user_id][df_type][signal]

    # Creazione dizionario dei df totali segmentati
    segmented_data = {}
    for signal in signals:
        segmented_data[signal] = pd.DataFrame()

    # Segmentazione dei df
    progress_bar = tqdm(total=len(users), desc="Segmentation")
    for user_id in users:

        for df_type in df_types:

            # Produzione dei segmenti
            data_temp = {}
            for signal in signals:
                data_temp[signal] = segmentation(data[user_id][df_type][signal], segment_prefix=f'{df_name}_{user_id}_{df_type}_', w_size=w_size, w_step_size=w_step_size)

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