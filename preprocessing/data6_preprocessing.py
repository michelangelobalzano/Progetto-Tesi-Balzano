from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import necessary_signals, structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments

####################################################################################################################
# DATASET 6: dataset composto da 35 utenti, ognuno avente una registrazione
####################################################################################################################

# Id degli utenti
users = ['S01', 'S02', 'S03', 'S04', 'S05', 
        'S06', 'S07', 'S08', 'S09', 'S10',
        'S11', 'S12', 'S13', 'S14', 'S15',
        'S16', 'S17', 'S18', 'S19', 'S20',
        'S21', 'S22', 'S23', 'S24', 'S25',
        'S26', 'S27', 'S28', 'S29', 'S30',
        'S31', 'S32', 'S33', 'S34', 'S35']





####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data(data_directory, df_name, signals):

    data = {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        data[user_id] = {}

        directory = data_directory + df_name + '\\' + user_id + '\\'
        file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

        for signal in set(signals) | set(necessary_signals):
            
            for file in file_path:

                if file.endswith(f'{signal}.csv'):
                    data[user_id][signal] = pd.read_csv(join(directory, file), header=None)
                    break

        progress_bar.update(1)
    progress_bar.close()

    return data













####################################################################################################################
# Preprocessing del dataset 6
####################################################################################################################
def data6_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size):
    
    # Lettura del dataset
    data = read_sensor_data(data_directory, df_name, signals)

    # Preprocessing dei dataframe
    progress_bar = tqdm(total=len(users), desc="User preprocessing")
    for user_id in users:

        # Modifica dei dataframe
        for signal in set(signals) | set(necessary_signals):

            data[user_id][signal] = structure_modification(data[user_id][signal].copy(), signal, target_freq)

        # Determinazione momenti di off-body e sleep
        data[user_id] = off_body_detection(data[user_id], signals)
        data[user_id] = sleep_detection(data[user_id], signals)
        
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

    progress_bar = tqdm(total=len(users), desc="Segmentation")
    for user_id in users:

        # Produzione dei segmenti
        data_temp = {}
        for signal in signals:
            data_temp[signal] = segmentation(data[user_id][signal], segment_prefix=f'{df_name}_{user_id}_', w_size=w_size, w_step_size=w_step_size)

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