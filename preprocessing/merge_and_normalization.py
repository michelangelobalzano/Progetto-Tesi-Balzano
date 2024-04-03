import pandas as pd
from tqdm import tqdm
import numpy as np

# Colonne da normalizzare per ogni segnale
cols_to_normalize = {
    'ACC': ['x', 'y', 'z'],
    'BVP': ['bvp'],
    'EDA': ['eda'],
    'HR': ['hr'],
    'TEMP': ['temp']
}

def merge_and_normalize(data_directory, df_names, signals, user_max_segments, labeled=False):

    data = {} # Dizionario dei segnali
    label_df = pd.DataFrame() # Df delle etichette

    # Inizializzazione del dizionario dei df
    for signal in signals:
        data[signal] = pd.DataFrame()

    progress_bar = tqdm(total=len(df_names), desc="Data reading")
    for df_name in df_names:
        n_segmenti_tot = 0 # Provvisorio
        n_segmenti_da_cancellare_tot = 0 # Provvisorio
        directory = f'{data_directory}\\{df_name}\\'

        data_temp = {}
        for signal in signals:
            data_temp[signal] = pd.read_csv(f'{directory}{signal}.csv')
        if labeled:
            label_df_temp = pd.read_csv(f'{directory}LABELS.csv')

        if not labeled:
            # Determinazione utenti
            users = {'_'.join(segment_id.split('_')[:-1]) for segment_id in data_temp[signals[0]]['segment_id']}

            # Determinazione segmenti da mantenere
            segments_to_delete = set()
            for user_id in users:
                user_segments = data_temp[signals[0]].loc[data_temp[signals[0]]['segment_id'].str.startswith(user_id), 'segment_id'].unique()
                n_segmenti_tot += len(user_segments) # Provvisorio

                if len(user_segments) > user_max_segments:
                    segments_to_delete.update(np.random.choice(user_segments, size=len(user_segments)-user_max_segments, replace=False))
                    
            n_segmenti_da_cancellare_tot += len(segments_to_delete) # Provvisorio
            
            if len(segments_to_delete) > 0:
                for signal in signals:
                    data_temp[signal] = data_temp[signal][~data_temp[signal]['segment_id'].isin(segments_to_delete)]

        print('TOTALE SEGMENTI: ', n_segmenti_tot)
        print('TOTALE SEGMENTI DA CANCELLARE: ', n_segmenti_da_cancellare_tot)
        print('NUMERO SEGMENTI RISULTANTI: ', data_temp[signals[0]]['segment_id'].nunique())


        # Concatenazione del dataset
        for signal in signals:
            data[signal] = pd.concat([data[signal], data_temp[signal]], axis=0, ignore_index=True)
            if labeled:
                label_df = pd.concat([label_df, label_df_temp], axis=0, ignore_index=True)

        progress_bar.update(1)
    progress_bar.close()

    ######################################################
    # Normalization
    for signal in signals:
        for col in cols_to_normalize[signal]:
            mean = data[signal][col].mean()
            std = data[signal][col].std()
            data[signal][col] = (data[signal][col] - mean) / std

    for signal in signals:
        if labeled:
            data[signal].to_csv(f'processed_data\\{signal}_LABELED.csv', index=False)
            label_df.to_csv(f'processed_data\\LABELS.csv', index=False)
        else:
            data[signal].to_csv(f'processed_data\\{signal}.csv', index=False)