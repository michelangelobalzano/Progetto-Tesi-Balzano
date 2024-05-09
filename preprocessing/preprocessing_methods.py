import os
import pandas as pd
import numpy as np
from datetime import timedelta

df_cols = {'ACC': ['x', 'y', 'z'], 'BVP': ['bvp'], 'EDA': ['eda'], 'HR': ['hr']} # Colonne dei df
# data['ACC'] serve per la determinazione dei momenti di sonno
# EDA serve per la determinazione dei momenti di off-body
# Bisogna considerarli anche se non servono per la classificazione
necessary_signals = ['ACC', 'EDA']
precision = {'ACC': 0, 'BVP': 2, 'EDA': 6, 'HR': 2} # Numero di cifre decimali

####################################################################################################################
# Modifica della struttura dei dataset:
# Aggiunta di una colonna contenente il tempo della singola registrazione
# Rimozione delle prime due righe
# Aggiunta delle intestazioni alle colonne
# Aggiunta dell'data['ACC']elerazione totale
####################################################################################################################
def structure_modification(df, signal, target_freq):

    cols = df_cols[signal]
    original_freq = int(df.iloc[1, 0])
    # Calcolo del vettore dei tempi delle singole misurazioni
    df_time = time_calculation(df)
    # Rimozione delle prime due righe
    df = df.iloc[2:, :]
    # Aggiunta delle intestazioni alle colonne
    df.columns = cols
    # Aggiunta della colonna dei tempi
    df['time'] = pd.to_datetime(df_time, unit='s')

    # Resampling se la frequenza originale è maggiore della frequenza target
    # Interpolazione se la frequenza originale è minore della frequenza target
    if original_freq >= target_freq:
        df = resampling(df, target_freq)
    else:
        df = interpolazione(df, target_freq)

    for col in cols:
        df[col] = df[col].round(precision[signal])

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
def resampling(df, target_freq):

    # Impostazione della colonna 'time' come indice
    df = df.set_index('time')
    # Resampling
    df = df.resample(f'{1/target_freq}s').mean()
    # Reset dell'indice impostato sul tempo
    df = df.reset_index()

    return df

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
def off_body_detection(data, signals):

    # Creazione colonna off-body
    for signal in signals:
        data[signal]['off-body'] = False

    data['EDA'].loc[(data['EDA']['eda'] < 0.05) | (data['EDA']['eda'] > 100), 'off-body'] = True

    for signal in signals:
        # Determinazione dei momenti di off-body in base al valore di EDA
        data[signal]['off-body'] = data['EDA']['off-body']

    return data







####################################################################################################################
# Determinazione dei periodi di sleep in base ai cambiamenti dell'angolazione del brdata['ACC']io
# Aggiunge una colonna booleana ai df di data['ACC']
# True: sleep
# False: wake
####################################################################################################################
def sleep_detection(data, signals, median_windows_size=5, angle_degree_threshold=5, sleep_windows_size=300):

    # median_windows_size = 5: Finestra scorrevole per il calcolo delle data['ACC']elerazioni medie
    # angle_degree = 5: Angolo di tolleranza per la determinazione delle sue variazioni
    # sleep_windows_size = 300: (5 minuti) Finestra scorrevole per la determinazione di assenza di variazione di angolo

    # Impostazione del tempo come indice
    data['ACC'] = data['ACC'].set_index('time')

    # Calcolo delle data['ACC']elerazioni medie in finestre scorrevoli di 5 secondi
    data['ACC']['ax'] = data['ACC']['x'].rolling(window = f'{median_windows_size}s').mean()
    data['ACC']['ay'] = data['ACC']['y'].rolling(window = f'{median_windows_size}s').mean()
    data['ACC']['az'] = data['ACC']['z'].rolling(window = f'{median_windows_size}s').mean()

    # Calcolo dell'angolazione del brdata['ACC']io
    data['ACC']['angle'] = np.arctan2(data['ACC']['az'], np.sqrt(data['ACC']['ax']**2 + data['ACC']['ay']**2)) * 180 / np.pi

    # Calcolo delle angolazioni del brdata['ACC']io medie in finestre scorrevoli di 5 secondi
    data['ACC']['medium_angle'] = data['ACC']['angle'].rolling(window = f'{median_windows_size}s').mean()

    # Calcolo del valore minimo e massimo di medium_angle per ogni finestra
    data['ACC']['medium_angle_min'] = data['ACC']['medium_angle'].rolling(window = f'{sleep_windows_size}s').min()
    data['ACC']['medium_angle_max'] = data['ACC']['medium_angle'].rolling(window = f'{sleep_windows_size}s').max()
    data['ACC']['angle_difference'] = data['ACC']['medium_angle_max'] - data['ACC']['medium_angle_min']
    
    data['ACC']['sleep'] = data['ACC']['angle_difference'] <= angle_degree_threshold

    # Reset dell'indice del tempo
    data['ACC'] = data['ACC'].reset_index()

    # Determinazione dei momenti di sonno
    for signal in signals:
        data[signal]['sleep'] = data['ACC']['sleep']

    # Eliminazione delle colonne intermedie
    data['ACC'] = data['ACC'].drop(['ax', 'ay', 'az', 'medium_angle', 'angle', 'medium_angle_min', 'medium_angle_max', 'angle_difference'], axis=1)

    return data












####################################################################################################################
# Segmentazione con metodo della finestra scorrevole
# Produzione di un dataset per ogni utente contenente tutti i segmenti
# Aggiunta della colonna 'segment_id'
# La singola riga è una singola registrazione appartenente al segmento 'segment_id'
####################################################################################################################
def segmentation(df, segment_prefix, w_size, w_step_size, user_id=None):

    segments_df = pd.DataFrame()

    start_time = df['time'].min()
    end_time = df['time'].max()
    start_timestamp = start_time
    segments = []
    segment_number = 0

    # Aggiunta colonna user_id per poter rimuovere segmenti casuali per user_id
    if user_id is not None:
        df['user_id'] = user_id

    while start_timestamp + timedelta(seconds=w_size) <= end_time:

        end_timestamp = start_timestamp + timedelta(seconds=w_size)

        segment = df[(df['time'] >= start_timestamp) & (df['time'] < end_timestamp)].copy()
        segment['segment_id'] = int(f'{segment_prefix}{segment_number}')
        segments.append(segment)

        start_timestamp += timedelta(seconds=w_step_size)
        segment_number += 1

    if(segment_number > 0):
        segments_df = pd.concat(segments, ignore_index=True)

    return segments_df







####################################################################################################################
# Metodo di cancellazione dei segmenti di off-body e sleep
####################################################################################################################
def delete_off_body_and_sleep_segments(data, signals):

    # Trova segment_id con sleep=True in eda_df
    segmenti_sleep = set(data['EDA'][data['EDA']['sleep'] == True]['segment_id'])
    
    # Trova segment_id con off-body=True in eda_df
    segmenti_offbody = set(data['EDA'][data['EDA']['off-body'] == True]['segment_id'])

    # Unione dei segment_id con sleep a True in data['ACC']_df e con off-body a True in eda_df
    segmenti_da_rimuovere = segmenti_sleep.union(segmenti_offbody)

    # Rimuovi le righe con segment_id presenti nell'insieme segmenti_da_rimuovere
    for signal in signals:
        data[signal] = data[signal][~data[signal]['segment_id'].isin(segmenti_da_rimuovere)]

    return data












####################################################################################################################
# Esportazione del dataframe finale
####################################################################################################################
def export_df(df, output_dir, signal):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(f'{output_dir}\\{signal}.csv', index=False)









####################################################################################################################
# Cancellazione di segmenti random se un utente ne ha più di limit=50
####################################################################################################################
def delete_random_segments(users, dataset_name, bvp, eda, hr, n_max_segmenti):

    num_segmenti_cancellati = 0

    for user_id in users:

        user_segments = bvp[bvp['segment_id'].str.split('_').str[1] == str(user_id)]
        num_segments = user_segments['segment_id'].nunique()

        print(f'numero di segmenti di {user_id}: {num_segments}')

        if num_segments > n_max_segmenti:
            
            segments_to_delete = np.random.choice(user_segments['segment_id'], size=num_segments - n_max_segmenti, replace=False)
            num_segmenti_cancellati += len(segments_to_delete)
            bvp = bvp[~bvp['segment_id'].isin(segments_to_delete)]
            eda = eda[~eda['segment_id'].isin(segments_to_delete)]
            hr = hr[~hr['segment_id'].isin(segments_to_delete)]

    print(f'numero di segmenti eliminati: {num_segmenti_cancellati}')

    return bvp, eda, hr