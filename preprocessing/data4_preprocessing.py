from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import necessary_signals, structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df

####################################################################################################################
# DATASET 4: dataset composto da 1 utente con 30 registrazioni
####################################################################################################################

# Id degli utenti
users = ['S1']
user_id = 'S1'
# Nomi delle registrazioni
reg_names = ['R1', 'R2', 'R3', 'R4', 'R5',
             'R6', 'R7', 'R8', 'R9', 'R10',
             'R11', 'R12', 'R13', 'R14', 'R15',
             'R16', 'R17', 'R18', 'R19', 'R20',
             'R21', 'R22', 'R23', 'R24', 'R25',
             'R26', 'R27', 'R28', 'R29', 'R30',]





####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data(data_directory, df_name, signals):

    data = {}

    progress_bar = tqdm(total=len(reg_names), desc="Data reading")
    for reg_name in reg_names:

        data[reg_name] = {}

        directory = data_directory + df_name + '\\' + reg_name + '\\'
        file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

        for file in file_path:

            for signal in set(signals) | set(necessary_signals):
            
                for file in file_path:

                    if file.endswith(f'{signal}.csv'):
                        data[reg_name][signal] = pd.read_csv(join(directory, file), header=None)
                        break

        progress_bar.update(1)
    progress_bar.close()

    return data












####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
def data4_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size): 
    
    # Lettura del dataset
    data = read_sensor_data(data_directory, df_name, signals)

    # Preprocessing dei dataframe
    progress_bar = tqdm(total=len(reg_names), desc="User preprocessing")
    for reg_name in reg_names:

        # Modifica dei dataframe
        for signal in set(signals) | set(necessary_signals):

            data[reg_name][signal] = structure_modification(data[reg_name][signal].copy(), signal, target_freq)

        # Determinazione momenti di off-body e sleep
        data[reg_name] = off_body_detection(data[reg_name], signals)
        data[reg_name] = sleep_detection(data[reg_name], signals)

        progress_bar.update(1)
    progress_bar.close()

    # Rimozione dei dati di EDA e ACC se non utili alla classificazione
    for signal in necessary_signals:
        if signal not in signals:
            for reg_name in reg_names:
                del data[reg_name][signal]

    # Creazione dizionario dei df totali segmentati
    segmented_data = {}
    for signal in signals:
        segmented_data[signal] = pd.DataFrame()

    # Segmentazione dei df
    progress_bar = tqdm(total=len(reg_names), desc="Segmentation")
    for reg_name in reg_names:

        # Produzione dei segmenti
        data_temp = {}
        for signal in signals:
            data_temp[signal] = segmentation(data[reg_name][signal], segment_prefix=f'{user_id}_{reg_name}_', w_size=w_size, w_step_size=w_step_size)

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