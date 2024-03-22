from os import listdir
from os.path import isfile, join
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from preprocessing import structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments, cut_and_label

####################################################################################################################
# DATASET 9: dataset composto da 15 utenti, ognuno avente una registrazione
####################################################################################################################

# Nome del dataset
dataset_name = 'data9'
# Directory del dataset
data_directory = f'data\{dataset_name}\\'
# Id degli utenti
users = ['S2', 'S3', 'S4', 'S5', 'S6',
          'S7', 'S8', 'S9', 'S10', 'S11', 
          'S13', 'S14', 'S15', 'S16', 'S17']
# Numero degli intervalli di tempo etichettati
num_intervals = 5





####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data():

    acc, bvp, eda, hr = {}, {}, {}, {}
    start_times = {}
    end_times = {}
    valence = {}
    arousal = {}
    reg_start_times = {}
    intervals = {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        directory = data_directory + user_id + '\\'

        file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

        for file in file_path:

            if file.endswith('ACC.csv'):
                acc[user_id] = pd.read_csv(join(directory, file), header=None)
                
            elif file.endswith('BVP.csv'):
                bvp[user_id] = pd.read_csv(join(directory, file), header=None)

            elif file.endswith('EDA.csv'):
                eda[user_id] = pd.read_csv(join(directory, file), header=None)

            elif file.endswith('HR.csv'):
                hr[user_id] = pd.read_csv(join(directory, file), header=None)

            # Lettura delle etichette e degli intervalli al quale sono associate
            # Gli intervalli sono in formato <minuto>:<secondo> passati dall'inizio
            # della registrazione con il dispositivo respiban
            elif file.endswith(f'{user_id}_quest.csv'):
                df = pd.read_csv(join(directory, file), sep=';', header=None)

                start_times[user_id] = df.iloc[2, 1:6].tolist()
                end_times[user_id] = df.iloc[3, 1:6].tolist()

                for i in range(num_intervals):
                    s_minuti, *s_secondi = map(int, start_times[user_id][i].split('.'))
                    start_times[user_id][i] = timedelta(minutes=s_minuti, seconds=s_secondi[0] if s_secondi else 0)
                    e_minuti, *e_secondi = map(int, end_times[user_id][i].split('.'))
                    end_times[user_id][i] = timedelta(minutes=e_minuti, seconds=e_secondi[0] if e_secondi else 0)

                valence[user_id] = df.iloc[17:22, 1].tolist()
                arousal[user_id] = df.iloc[17:22, 2].tolist()

            # Lettura tempi di inizio registrazione dispositivo respiban
            # Gli intervalli delle etichette partono da questi tempi
            # I tempi di inizio registrazione sono in formato <ora>:<minuto>
            elif file.endswith(f'{user_id}_respiban.txt'):
                with open(join(directory, file), "r") as f:
                    f.seek(0)
                    seconda_riga = f.readline()
                    seconda_riga = f.readline()
                    # Recupero data
                    inizio_delimitatore_data = 'date": "'
                    fine_delimitatore_data = '", "mode'
                    indice_inizio = seconda_riga.find(inizio_delimitatore_data)
                    indice_fine = seconda_riga.find(fine_delimitatore_data)
                    data = seconda_riga[indice_inizio + len(inizio_delimitatore_data):indice_fine]

                    # Recupero orario
                    inizio_delimitatore_tempo = 'time": "'
                    fine_delimitatore_tempo = ':1.0", "comments'
                    indice_inizio = seconda_riga.find(inizio_delimitatore_tempo)
                    indice_fine = seconda_riga.find(fine_delimitatore_tempo)
                    orario = seconda_riga[indice_inizio + len(inizio_delimitatore_tempo):indice_fine]
                    
                    # Conversione in datetime
                    reg_start = f'{data} {orario}'
                    reg_start_times[user_id] = datetime.strptime(reg_start, '%Y-%m-%d %H:%M')

        # Calcolo dei datetime corrispondenti agli intervalli etichettati
        intervals[user_id] = []
        for i in range(num_intervals):
            start = reg_start_times[user_id] + start_times[user_id][i]
            end = reg_start_times[user_id] + end_times[user_id][i]
            intervals[user_id].append([start, end])

        progress_bar.update(1)

    progress_bar.close()

    return acc, bvp, eda, hr, valence, arousal, intervals















####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
# Lettura del dataset
acc, bvp, eda, hr, valence, arousal, intervals = read_sensor_data()

# Dataframe di segmenti uniti
acc_df, bvp_df, eda_df, hr_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

progress_bar = tqdm(total=len(users), desc="User preprocessing")

for user_id in users:

    # Modifica dei dataframe
    acc[user_id] = structure_modification(acc[user_id].copy(), 'acc')
    bvp[user_id] = structure_modification(bvp[user_id].copy(), 'bvp')
    eda[user_id] = structure_modification(eda[user_id].copy(), 'eda')
    hr[user_id] = structure_modification(hr[user_id].copy(), 'hr')

    # Determinazione momenti di off-body e sleep
    eda[user_id] = off_body_detection(eda[user_id])
    eda[user_id] = sleep_detection(eda[user_id], acc[user_id])

    progress_bar.update(1)
progress_bar.close()

del acc_df

# Ritaglio degli intervalli etichettati
for user_id in users:
    bvp[user_id] = cut_and_label(bvp[user_id], intervals[user_id], valence[user_id], arousal[user_id])
    eda[user_id] = cut_and_label(eda[user_id], intervals[user_id], valence[user_id], arousal[user_id])
    hr[user_id] = cut_and_label(hr[user_id], intervals[user_id], valence[user_id], arousal[user_id])

progress_bar = tqdm(total=len(users), desc="Segmentation")
for user_id in users:

    # Dataframe di segmenti temporanei per utente
    bvp_temp, eda_df_temp, hr_temp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Produzione dei segmenti
    bvp_temp = segmentation(bvp[user_id], segment_prefix=f'{dataset_name}_{user_id}_')
    eda_temp = segmentation(eda[user_id], segment_prefix=f'{dataset_name}_{user_id}_')
    hr_temp = segmentation(hr[user_id], segment_prefix=f'{dataset_name}_{user_id}_')

    # Eliminazione segmenti di off-body e sleep
    bvp_temp, eda_temp, hr_temp = delete_off_body_and_sleep_segments(bvp_temp, eda_temp, hr_temp) 
    
    bvp_df = pd.concat([bvp_df, bvp_temp], axis=0, ignore_index=True)
    eda_df = pd.concat([eda_df, eda_temp], axis=0, ignore_index=True)
    hr_df = pd.concat([hr_df, hr_temp], axis=0, ignore_index=True)

    progress_bar.update(1)
progress_bar.close()

# Applicazione delle etichette ai dataframes


acc_df = acc_df.drop(['sleep'], axis=1)
eda_df = eda_df.drop(['off-body'], axis=1)

print('Cancellazione segmenti random...')
acc_df, bvp_df, eda_df, hr_df = delete_random_segments(users, dataset_name, acc_df, bvp_df, eda_df, hr_df)

# Esportazione delle features del dataset
print("Esportazione ACC")
export_df(acc_df, f'\\{dataset_name}\\acc')
print("Esportazione BVP")
export_df(bvp_df, f'\\{dataset_name}\\bvp')
print("Esportazione EDA")
export_df(eda_df, f'\\{dataset_name}\\eda')
print("Esportazione HR")
export_df(hr_df, f'\\{dataset_name}\\hr')
print("Esportazione Completata")