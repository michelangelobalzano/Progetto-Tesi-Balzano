import os
from os.path import join
import pandas as pd
from tqdm import tqdm
from preprocessing_methods import structure_modification, segmentation, export_df, necessary_signals

def get_users(data_directory):

    users = set()
    for user_directory in os.listdir(data_directory):
        if os.path.isdir(os.path.join(data_directory, user_directory)):
            user_id = user_directory
            users.add(user_id)

    return list(users)

def read_sensor_data(data_directory, users, signals):

    data = {}

    # Creazione struttura dati
    for user_id in users:
        data[user_id] = {}

    progress_bar = tqdm(total=len(users), desc="Data reading")
    for user_directory in os.listdir(data_directory):

        if os.path.isdir(os.path.join(data_directory, user_directory)):
            user_directory_path = os.path.join(data_directory, user_directory)
            user_id = user_directory

            files = [f for f in os.listdir(user_directory_path) if os.path.isfile(join(user_directory_path, f))]

            for signal in set(signals) | set(necessary_signals) | set(['BVP_LABELED']):
                for file in files:
                    if file.endswith(f'{signal}.csv'):
                        data[user_id][signal] = pd.read_csv(join(user_directory_path, file), header=None)
                        break

        data[user_id]['BVP_LABELED'].columns = ['time', 'bvp', 'valence', 'arousal']
        data[user_id]['BVP_LABELED'] = data[user_id]['BVP_LABELED'].iloc[1:]

        progress_bar.update(1)
    progress_bar.close()

    return data















####################################################################################################################
# Esecuzione del preprocessing
####################################################################################################################
# Lettura del dataset
def labeled_data_preprocessing(data_directory, df_name, signals, target_freq, w_size, w_step_size):
    
    users = get_users(data_directory)
    data = read_sensor_data(data_directory, users, signals)

    progress_bar = tqdm(total=len(users), desc="User preprocessing")
    for user_id in users:
        # Modifica dei dataframe
        for signal in set(signals) | set(necessary_signals):
            data[user_id][signal] = structure_modification(data[user_id][signal].copy(), signal, target_freq)
        progress_bar.update(1)
    progress_bar.close()

    # Rimozione dei dati di EDA e ACC se non utili alla classificazione
    for signal in necessary_signals:
        if signal not in signals:
            for user_id in users:
                del data[user_id][signal]

    # Ritaglio degli intervalli etichettati
    progress_bar = tqdm(total=len(users), desc="Labeling")
    for user_id in users:
        # Recupero dei time-stamp etichettati
        labeled_times = data[user_id]['BVP_LABELED'].loc[data[user_id]['BVP_LABELED']['valence'].notnull(), 'time']
        # Ritaglio dei df lasciando solo gli intervalli etichettati
        for signal in set(signals) | set(['BVP_LABELED']):
            data[user_id][signal]['time'] = pd.to_datetime(data[user_id][signal]['time'])
            data[user_id][signal] = data[user_id][signal][data[user_id][signal]['time'].isin(labeled_times)]
        progress_bar.update(1)
    progress_bar.close()

    # Creazione dizionario dei df totali segmentati
    segmented_data = {}
    for signal in set(signals) | set(['BVP_LABELED']):
        segmented_data[signal] = pd.DataFrame()
        
    # Segmentazione dei df
    progress_bar = tqdm(total=len(users), desc="Segmentation")
    for user_id in users:
        # Produzione dei segmenti
        data_temp = {}
        for signal in set(signals) | set(['BVP_LABELED']):
            data_temp[signal] = segmentation(data[user_id][signal], segment_prefix=f'{df_name}{user_id}', w_size=w_size, w_step_size=w_step_size)
            segmented_data[signal] = pd.concat([segmented_data[signal], data_temp[signal]], axis=0, ignore_index=True)
        progress_bar.update(1)
    progress_bar.close()

    # Eliminazione dei segmenti creati che non contengono frequency * window_size valori
    for signal in set(signals) | set(['BVP_LABELED']):
        segmented_data[signal] = segmented_data[signal].groupby('segment_id').filter(lambda x: len(x) == target_freq * w_size)

    # Applicazione delle etichette di maggioranza ad ogni segmento
    label_df = pd.DataFrame(columns=['segment_id', 'valence', 'arousal'])
    for segment_id, segment in segmented_data['BVP_LABELED'].groupby('segment_id'):
        row = {
            'segment_id': segment_id,
            'valence': segment['valence'].mode().iloc[0],
            'arousal': segment['arousal'].mode().iloc[0]
        }
        row_df = pd.DataFrame([row])
        label_df = pd.concat([label_df, row_df], ignore_index=True)

    # Controllo che i segment_id dei tre dataset coincidono
    #valori_df1 = set(bvp_df.groupby('segment_id').groups.keys())
    #valori_df2 = set(eda_df.groupby('segment_id').groups.keys())
    #valori_df3 = set(hr_df.groupby('segment_id').groups.keys())
    #coincidono = valori_df1 == valori_df2 == valori_df3
        
    # Eliminazione colonne inutili
    for signal in signals:
        segmented_data[signal] = segmented_data[signal].drop(['time'], axis=1)
    segmented_data['BVP_LABELED'] = segmented_data['BVP_LABELED'].drop(['time', 'valence', 'arousal'], axis=1)

    # Esportazione delle features del dataset
    for signal in signals:
        print(f"Esportazione {signal}...")
        export_df(segmented_data[signal], data_directory, signal)
    print(f"Esportazione BVP...")
    export_df(segmented_data['BVP_LABELED'], data_directory, 'BVP')
    print(f"Esportazione etichette...")
    export_df(label_df, data_directory, 'LABELS')
    