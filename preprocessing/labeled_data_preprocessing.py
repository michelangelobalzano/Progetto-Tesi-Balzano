from os import listdir
from os.path import isfile, join
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from preprocessing import structure_modification, segmentation, export_df, frequency, w_size

####################################################################################################################
# DATASET ETICHETTATO: dataset composto da 15 utenti, ognuno avente una registrazione
####################################################################################################################

# Nome del dataset
dataset_name = 'labeled_data'
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

    bvp, eda, hr = {}, {}, {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        directory = data_directory + user_id + '\\'

        file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

        for file in file_path:
                
            if file.endswith('BVP_LABELED.csv'):
                bvp[user_id] = pd.read_csv(join(directory, file), header=None)
                bvp[user_id].columns = ['time', 'bvp', 'valence', 'arousal']
                bvp[user_id] = bvp[user_id].iloc[1:]

            elif file.endswith('EDA.csv'):
                eda[user_id] = pd.read_csv(join(directory, file), header=None)

            elif file.endswith('HR.csv'):
                hr[user_id] = pd.read_csv(join(directory, file), header=None)

        progress_bar.update(1)

    progress_bar.close()

    return bvp, eda, hr















####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
# Lettura del dataset
bvp, eda, hr = read_sensor_data()

# Dataframe di segmenti uniti
bvp_df, eda_df, hr_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

progress_bar = tqdm(total=len(users), desc="User preprocessing")
for user_id in users:

    # Modifica dei dataframe
    eda[user_id] = structure_modification(eda[user_id].copy(), 'eda')
    hr[user_id] = structure_modification(hr[user_id].copy(), 'hr')

    progress_bar.update(1)
progress_bar.close()

# Ritaglio degli intervalli etichettati
progress_bar = tqdm(total=len(users), desc="Labeling")
for user_id in users:

    # Recupero dei time-stamp etichettati
    labeled_times = bvp[user_id].loc[bvp[user_id]['valence'].notnull(), 'time']
    # Ritaglio dei df lasciando solo gli intervalli etichettati
    bvp[user_id]['time'] = pd.to_datetime(bvp[user_id]['time'])
    eda[user_id]['time'] = pd.to_datetime(eda[user_id]['time'])
    hr[user_id]['time'] = pd.to_datetime(hr[user_id]['time'])
    bvp[user_id] = bvp[user_id][bvp[user_id]['time'].isin(labeled_times)]
    eda[user_id] = eda[user_id][eda[user_id]['time'].isin(labeled_times)]
    hr[user_id] = hr[user_id][hr[user_id]['time'].isin(labeled_times)]

    progress_bar.update(1)
progress_bar.close()
    

progress_bar = tqdm(total=len(users), desc="Segmentation")
for user_id in users:

    # Dataframe di segmenti temporanei per utente
    bvp_temp, eda_df_temp, hr_temp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Produzione dei segmenti
    bvp_temp = segmentation(bvp[user_id], segment_prefix=f'{dataset_name}_{user_id}_')
    eda_temp = segmentation(eda[user_id], segment_prefix=f'{dataset_name}_{user_id}_')
    hr_temp = segmentation(hr[user_id], segment_prefix=f'{dataset_name}_{user_id}_')
    
    bvp_df = pd.concat([bvp_df, bvp_temp], axis=0, ignore_index=True)
    eda_df = pd.concat([eda_df, eda_temp], axis=0, ignore_index=True)
    hr_df = pd.concat([hr_df, hr_temp], axis=0, ignore_index=True)

    progress_bar.update(1)
progress_bar.close()

# Eliminazione dei segmenti creati che non contengono frequency * window_size valori
bvp_df = bvp_df.groupby('segment_id').filter(lambda x: len(x) == frequency * w_size)
eda_df = eda_df.groupby('segment_id').filter(lambda x: len(x) == frequency * w_size)
hr_df = hr_df.groupby('segment_id').filter(lambda x: len(x) == frequency * w_size)

# Controllo che i segment_id dei tre dataset coincidono
#valori_df1 = set(bvp_df.groupby('segment_id').groups.keys())
#valori_df2 = set(eda_df.groupby('segment_id').groups.keys())
#valori_df3 = set(hr_df.groupby('segment_id').groups.keys())
#coincidono = valori_df1 == valori_df2 == valori_df3

# Esportazione delle features del dataset
print("Esportazione BVP")
export_df(bvp_df, f'\\{dataset_name}\\bvp')
print("Esportazione EDA")
export_df(eda_df, f'\\{dataset_name}\\eda')
print("Esportazione HR")
export_df(hr_df, f'\\{dataset_name}\\hr')
print("Esportazione Completata")