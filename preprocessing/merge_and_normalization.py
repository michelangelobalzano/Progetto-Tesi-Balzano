import os
from os.path import join
import pandas as pd
from tqdm import tqdm
import numpy as np

from preprocessing_methods import precision

# Colonne da normalizzare per ogni segnale
cols_to_normalize = {
    'ACC': ['x', 'y', 'z'],
    'BVP': ['bvp'],
    'EDA': ['eda'],
    'HR': ['hr'],
    'TEMP': ['temp']
}

def get_users(data_directory):

    users = set()
    for user_directory in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, user_directory)):
            user_id = user_directory
            users.add(user_id)

    return list(users)

def merge_and_normalize(data_directory, df_names, signals, labeled=False):

    data = {} # Dizionario dei segnali
    valence_df = pd.DataFrame() # Df delle etichette
    arousal_df = pd.DataFrame() # Df delle etichette

    # Inizializzazione del dizionario dei df
    for signal in signals:
        data[signal] = pd.DataFrame()

    progress_bar = tqdm(total=len(df_names), desc="Data reading")
    for df_name in df_names:
        n_segmenti_tot = 0 # Provvisorio
        n_segmenti_da_cancellare_tot = 0 # Provvisorio
        directory = f'{data_directory}{df_name}\\'

        data_temp = {}
        for signal in signals:
            data_temp[signal] = pd.read_csv(f'{directory}{signal}.csv', header=None, low_memory=False)
            c = cols_to_normalize[signal].copy()
            c.append('segment_id')
            data_temp[signal].columns = c
            data_temp[signal] = data_temp[signal].iloc[1:]
            for col in cols_to_normalize[signal]:
                data_temp[signal][col] = data_temp[signal][col].astype(float)
            data_temp[signal]['segment_id'] = data_temp[signal]['segment_id'].astype(np.int64)
        if labeled:
            valence_df_temp = pd.read_csv(f'{directory}VALENCE.csv', header=None, low_memory=False)
            arousal_df_temp = pd.read_csv(f'{directory}AROUSAL.csv', header=None, low_memory=False)
            c = ['segment_id', 'valence', 'arousal']
            valence_df_temp.columns = ['segment_id', 'valence']
            arousal_df_temp.columns = ['segment_id', 'arousal']
            valence_df_temp = valence_df_temp.iloc[1:]
            arousal_df_temp = arousal_df_temp.iloc[1:]

        # Se è fissato un numero massimo di segmenti per utente si cancellano se sono di numero maggiore
        # NON FUNZIONA CON IL SEGMENT_ID DI TIPO INTERO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''if not user_max_segments == None:
            if not labeled:
                # Determinazione utenti
                users = get_users(directory)
                # Riordinamento decrescente degli utenti per cercare i segmenti utente con il segment_id
                users = list(map(int, users))
                users.sort(reverse=True)
                users = [str(x) for x in users]

                # Determinazione segmenti da mantenere
                segments_to_delete = set()
                for user_id in users:
                    user_segments = data_temp[signals[0]].loc[data_temp[signals[0]]['segment_id'].astype(str).str.startswith(str(df_name+user_id)), 'segment_id'].unique()
                    n_segmenti_tot += len(user_segments) # Provvisorio
                    print(f'numero segmenti di {user_id}: {len(user_segments)}')

                    if len(user_segments) > user_max_segments:
                        segments_to_delete.update(np.random.choice(user_segments, size=len(user_segments)-user_max_segments, replace=False))
                        
                n_segmenti_da_cancellare_tot += len(segments_to_delete) # Provvisorio
                
                if len(segments_to_delete) > 0:
                    for signal in signals:
                        data_temp[signal] = data_temp[signal][~data_temp[signal]['segment_id'].isin(segments_to_delete)]

        print('TOTALE SEGMENTI: ', n_segmenti_tot)
        print('TOTALE SEGMENTI DA CANCELLARE: ', n_segmenti_da_cancellare_tot)
        print('NUMERO SEGMENTI RISULTANTI: ', data_temp[signals[0]]['segment_id'].nunique())'''


        # Concatenazione del dataset
        for signal in signals:
            data[signal] = pd.concat([data[signal], data_temp[signal]], axis=0, ignore_index=True)
        if labeled:
            valence_df = pd.concat([valence_df, valence_df_temp], axis=0, ignore_index=True)
            arousal_df = pd.concat([arousal_df, arousal_df_temp], axis=0, ignore_index=True)

        progress_bar.update(1)
    progress_bar.close()

    ######################################################
    # Normalization
    progress_bar = tqdm(total=len(signals), desc="Normalization")
    for signal in signals:
        for col in cols_to_normalize[signal]:
            mean = data[signal][col].mean()
            std = data[signal][col].std()
            data[signal][col] = round((data[signal][col] - mean) / std, precision[signal])
        progress_bar.update(1)
    progress_bar.close()

    progress_bar = tqdm(total=len(signals), desc="Extraction")
    for signal in signals:
        if labeled:
            data[signal].to_csv(f'processed_data\\{signal}_LABELED.csv', index=False)
            valence_df.to_csv(f'processed_data\\VALENCE.csv', index=False)
            arousal_df.to_csv(f'processed_data\\AROUSAL.csv', index=False)
        else:
            data[signal].to_csv(f'processed_data\\{signal}.csv', index=False)
        progress_bar.update(1)
    progress_bar.close()