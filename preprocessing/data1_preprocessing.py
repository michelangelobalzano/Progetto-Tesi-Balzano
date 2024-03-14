from os import listdir
from os.path import isfile, join
import pandas as pd
from tqdm import tqdm
from preprocessing import structure_modification, off_body_detection, sleep_detection, segmentation, delete_off_body_and_sleep_segments, export_df, delete_random_segments

####################################################################################################################
# DATASET 1: dataset composto da 10 utenti, ognuno avente 3 registrazioni: Midterm 1, Midterm 2 e Final
####################################################################################################################

# Nome del dataset
dataset_name = 'data1'
# Directory del dataset
data_directory = f'data\{dataset_name}\\'
# Id degli utenti
users = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
# Tipologie di dataframe
df_types = ['Midterm 1', 'Midterm 2', 'Final']





####################################################################################################################
# Lettura dei dataset e creazione di un dizionario per ogni sensore
# Ogni dizionario è composto da un dataset per ogni utente
####################################################################################################################
def read_sensor_data():

    acc, bvp, eda, hr = {}, {}, {}, {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_id in users:

        for df_type in df_types:

            directory = data_directory + user_id + '\\' + df_type + '\\'

            file_path = [f for f in listdir(directory) if isfile(join(directory, f))]

            for file in file_path:

                if file.endswith('ACC.csv'):
                    acc.setdefault(user_id, {})[df_type] = pd.read_csv(join(directory, file), header=None)
                    
                elif file.endswith('BVP.csv'):
                    bvp.setdefault(user_id, {})[df_type] = pd.read_csv(join(directory, file), header=None)

                elif file.endswith('EDA.csv'):
                    eda.setdefault(user_id, {})[df_type] = pd.read_csv(join(directory, file), header=None)

                elif file.endswith('HR.csv'):
                    hr.setdefault(user_id, {})[df_type] = pd.read_csv(join(directory, file), header=None)

        progress_bar.update(1)
    progress_bar.close()

    return acc, bvp, eda, hr





####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
# Lettura del dataset
acc, bvp, eda, hr = read_sensor_data()

# Dataframe di segmenti uniti
acc_df, bvp_df, eda_df, hr_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

progress_bar = tqdm(total=len(users), desc="User preprocessing")
for user_id in users:

    for df_type in df_types:

        # Modifica dei dataframe
        acc[user_id][df_type] = structure_modification(acc[user_id][df_type].copy(), user_id, dataset_name, 'acc')
        bvp[user_id][df_type] = structure_modification(bvp[user_id][df_type].copy(), user_id, dataset_name, 'bvp')
        eda[user_id][df_type] = structure_modification(eda[user_id][df_type].copy(), user_id, dataset_name, 'eda')
        hr[user_id][df_type] = structure_modification(hr[user_id][df_type].copy(), user_id, dataset_name, 'hr')

        # Determinazione momenti di off-body e sleep
        eda[user_id][df_type] = off_body_detection(eda[user_id][df_type])
        eda[user_id][df_type] = sleep_detection(eda[user_id][df_type], acc[user_id][df_type])

    progress_bar.update(1)
progress_bar.close()

del acc_df

progress_bar = tqdm(total=len(users), desc="Segmentation")
for user_id in users:

    for df_type in df_types:

        # Dataframe di segmenti temporanei per utente
        bvp_temp, eda_df_temp, hr_temp = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Produzione dei segmenti
        bvp_temp = segmentation(bvp[user_id][df_type], segment_prefix=f'{dataset_name}_{user_id}_{df_type}_')
        eda_temp = segmentation(eda[user_id][df_type], segment_prefix=f'{dataset_name}_{user_id}_{df_type}_')
        hr_temp = segmentation(hr[user_id][df_type], segment_prefix=f'{dataset_name}_{user_id}_{df_type}_')

        # Eliminazione segmenti di off-body e sleep
        bvp_temp, eda_temp, hr_temp = delete_off_body_and_sleep_segments(bvp_temp, eda_temp, hr_temp) 
        
        bvp_df = pd.concat([bvp_df, bvp_temp], axis=0, ignore_index=True)
        eda_df = pd.concat([eda_df, eda_temp], axis=0, ignore_index=True)
        hr_df = pd.concat([hr_df, hr_temp], axis=0, ignore_index=True)

    progress_bar.update(1)
progress_bar.close()

bvp_df = bvp_df.drop(['time'], axis=1)
eda_df = eda_df.drop(['time','off-body', 'sleep'], axis=1)
hr_df = hr_df.drop(['time'], axis=1)

print('Cancellazione segmenti random...')
bvp_df, eda_df, hr_df = delete_random_segments(users, dataset_name, bvp_df, eda_df, hr_df)

# Esportazione delle features del dataset
print("Esportazione BVP")
export_df(bvp_df, f'\\{dataset_name}\\bvp')
print("Esportazione EDA")
export_df(eda_df, f'\\{dataset_name}\\eda')
print("Esportazione HR")
export_df(hr_df, f'\\{dataset_name}\\hr')
print("Esportazione Completata")