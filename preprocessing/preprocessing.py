import os
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

output_directory = 'data\\'
# Colonne dei dataframe
acc_cols = ['x', 'y', 'z']
bvp_cols = ['bvp']
eda_cols = ['eda']
hr_cols = ['hr']
# Variabili segmentazione
w_size = 60
w_step_size = 15

# Frequenza di ricampionamento dei segnali
frequency = 4 #Hz

# Numero massimo di segmenti per utente
n_max_segmenti = 500










####################################################################################################################
# Modifica della struttura dei dataset:
# Aggiunta di una colonna contenente il tempo della singola registrazione
# Rimozione delle prime due righe
# Aggiunta delle intestazioni alle colonne
# Aggiunta dell'accelerazione totale
####################################################################################################################
def structure_modification(df, df_type):

    if(df_type == 'acc'):
        cols = acc_cols
    elif(df_type == 'bvp'):
        cols = bvp_cols
    elif(df_type == 'eda'):
        cols = eda_cols
    elif(df_type == 'hr'):
        cols = hr_cols
    
    # Calcolo del vettore dei tempi delle singole misurazioni
    df_time = time_calculation(df)
    # Rimozione delle prime due righe
    df = df.iloc[2:, :]
    # Aggiunta delle intestazioni alle colonne
    df.columns = cols
    # Aggiunta della colonna dei tempi
    df['time'] = pd.to_datetime(df_time, unit='s')

    # Resampling o interpolazione a 4 Hz
    if(df_type == 'hr'):
        df = interpolazione(df, target_freq=frequency)
    else:
        df = resampling(df, target_freq=frequency)

    return df

# Calcolo della colonna dei tempi in base alle due prime righe dei dataset
def time_calculation(df):

    start = df.iloc[0, 0]
    samp_freq = df.iloc[1, 0]
    data_slice = df.iloc[2:, :]
    stop = start + (len(data_slice) / samp_freq)
    time = np.linspace(start, stop, num=len(data_slice)).tolist()

    return time








####################################################################################################################
# Ricampionamento dei dataframe a target_freq Hz
# I tempi diventano multipli perfetti di 1/target_freq secondi
####################################################################################################################
# Resampling applicato ai dati con frequenza maggiore di 4 Hz ovvero ACC, BVP e EDA
def resampling(df, target_freq):

    # Impostazione della colonna 'time' come indice
    df = df.set_index('time')
    # Resampling
    df = df.resample(f'{1/target_freq}s').mean()
    # Reset dell'indice impostato sul tempo
    df = df.reset_index()

    return df

# Interpolazione applicato ai dati con frequenza minore di 4 Hz ovvero HR
def interpolazione(df, target_freq):

    # Impostazione della colonna 'time' come indice
    df = df.set_index('time')
    # Interpolazione
    df = df.resample(f'{1/target_freq}s').interpolate(method='linear')
    # Reset dell'indice impostato sul tempo
    df = df.reset_index()

    return df







####################################################################################################################
# Determinazione momenti di off-body in base ai valori di eda
# Il metodo aggiunge una colonna booleana ai df di EDA
# True: off-body
# False: on-body
####################################################################################################################
def off_body_detection(eda):

    # Creazione colonna off-body
    eda['off-body'] = False

    # Determinazione dei momenti di off-body in base al valore di EDA
    eda.loc[(eda['eda'] < 0.05) | (eda['eda'] > 100), 'off-body'] = True

    return eda







####################################################################################################################
# Determinazione dei periodi di sleep in base ai cambiamenti dell'angolazione del braccio
# Aggiunge una colonna booleana ai df di ACC
# True: sleep
# False: wake
####################################################################################################################
def sleep_detection(eda, acc, median_windows_size=5, angle_degree_threshold=5, sleep_windows_size=300):

    # median_windows_size = 5: Finestra scorrevole per il calcolo delle accelerazioni medie
    # angle_degree = 5: Angolo di tolleranza per la determinazione delle sue variazioni
    # sleep_windows_size = 300: (5 minuti) Finestra scorrevole per la determinazione di assenza di variazione di angolo

    # Impostazione del tempo come indice
    acc = acc.set_index('time')

    # Calcolo delle accelerazioni medie in finestre scorrevoli di 5 secondi
    acc['ax'] = acc['x'].rolling(window = f'{median_windows_size}s').mean()
    acc['ay'] = acc['y'].rolling(window = f'{median_windows_size}s').mean()
    acc['az'] = acc['z'].rolling(window = f'{median_windows_size}s').mean()

    # Calcolo dell'angolazione del braccio
    acc['angle'] = np.arctan2(acc['az'], np.sqrt(acc['ax']**2 + acc['ay']**2)) * 180 / np.pi

    # Calcolo delle angolazioni del braccio medie in finestre scorrevoli di 5 secondi
    acc['medium_angle'] = acc['angle'].rolling(window = f'{median_windows_size}s').mean()

    # Calcolo del valore minimo e massimo di medium_angle per ogni finestra
    acc['medium_angle_min'] = acc['medium_angle'].rolling(window = f'{sleep_windows_size}s').min()
    acc['medium_angle_max'] = acc['medium_angle'].rolling(window = f'{sleep_windows_size}s').max()
    acc['angle_difference'] = acc['medium_angle_max'] - acc['medium_angle_min']

    # Determinazione dei momenti di sonno
    eda['sleep'] = acc['angle_difference'] <= angle_degree_threshold

    # Reset dell'indice del tempo
    acc = acc.reset_index()

    # Eliminazione delle colonne intermedie
    acc = acc.drop(['ax', 'ay', 'az', 'medium_angle', 'angle', 'medium_angle_min', 'medium_angle_max', 'angle_difference'], axis=1)

    return eda












####################################################################################################################
# Segmentazione con metodo della finestra scorrevole
# Produzione di un dataset per ogni utente contenente tutti i segmenti
# Aggiunta della colonna 'segment_id'
# La singola riga è una singola registrazione appartenente al segmento 'segment_id'
####################################################################################################################
def segmentation(df, segment_prefix, window_size = w_size, step_size= w_step_size):

    segments_df = pd.DataFrame()

    start_time = df['time'].min()
    end_time = df['time'].max()
    start_timestamp = start_time
    segments = []
    segment_number = 0

    while start_timestamp + timedelta(seconds=window_size) <= end_time:

        end_timestamp = start_timestamp + timedelta(seconds=window_size)

        segment = df[(df['time'] >= start_timestamp) & (df['time'] < end_timestamp)].copy()
        segment['segment_id'] = f'{segment_prefix}{segment_number}'
        segments.append(segment)

        start_timestamp += timedelta(seconds=step_size)
        segment_number += 1

    if(segment_number > 0):
        segments_df = pd.concat(segments, ignore_index=True)

    return segments_df







####################################################################################################################
# Metodo di cancellazione dei segmenti di off-body e sleep
####################################################################################################################
def delete_off_body_and_sleep_segments(bvp_df, eda_df, hr_df):

    # Trova segment_id con sleep=True in eda_df
    segmenti_sleep = set(eda_df[eda_df['sleep'] == True]['segment_id'])
    
    # Trova segment_id con off-body=True in eda_df
    segmenti_offbody = set(eda_df[eda_df['off-body'] == True]['segment_id'])

    # Unione dei segment_id con sleep a True in acc_df e con off-body a True in eda_df
    segmenti_da_rimuovere = segmenti_sleep.union(segmenti_offbody)

    # Rimuovi le righe con segment_id presenti nell'insieme segmenti_da_rimuovere
    bvp_df = bvp_df[~bvp_df['segment_id'].isin(segmenti_da_rimuovere)]
    eda_df = eda_df[~eda_df['segment_id'].isin(segmenti_da_rimuovere)]
    hr_df = hr_df[~hr_df['segment_id'].isin(segmenti_da_rimuovere)]

    return bvp_df, eda_df, hr_df












####################################################################################################################
# Esportazione del dataframe finale
####################################################################################################################
def export_df(df, dataset_name):

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    df.to_csv(f'{output_directory}{dataset_name}.csv', index=False)









####################################################################################################################
# Cancellazione di segmenti random se un utente ne ha più di limit=50
####################################################################################################################
def delete_random_segments(users, dataset_name, bvp, eda, hr, limit=n_max_segmenti):

    num_segmenti_cancellati = 0

    for user_id in users:

        user_segments = bvp[bvp['segment_id'].str.split('_').str[1] == str(user_id)]
        num_segments = user_segments['segment_id'].nunique()

        print(f'numero di segmenti di {user_id}: {num_segments}')

        if num_segments > limit:
            
            segments_to_delete = np.random.choice(user_segments['segment_id'], size=num_segments - limit, replace=False)
            num_segmenti_cancellati += len(segments_to_delete)
            bvp = bvp[~bvp['segment_id'].isin(segments_to_delete)]
            eda = eda[~eda['segment_id'].isin(segments_to_delete)]
            hr = hr[~hr['segment_id'].isin(segments_to_delete)]

    print(f'numero di segmenti eliminati: {num_segmenti_cancellati}')

    return bvp, eda, hr





####################################################################################################################
# Ritaglio del dataset negli intervalli etichettati
####################################################################################################################
def cut_and_label(df, intervals, valence, arousal):
    
    df_output = pd.DataFrame()
    #df.to_csv(f'{output_directory}_originale.csv', index=False)
    print(df['time'])
    
    for i, (start, end) in enumerate(intervals):
        
        df_ritagliato =  df.loc[(df['time'] >= start) & (df['time'] <= end)]
        df_ritagliato['valence'] = valence[i]
        df_ritagliato['arousal'] = arousal[i]

        df_ritagliato.to_csv(f'{output_directory}_prova.csv', index=False)
        input()

        df_output = pd.concat([df_output, df_ritagliato])
    
    return df_output