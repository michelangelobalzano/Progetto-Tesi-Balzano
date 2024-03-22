import pickle
import pandas as pd
from collections import Counter
from tqdm import tqdm

from preprocessing import time_calculation, resampling

# Ricostruzione dei

users = ['S2', 'S3', 'S4', 'S5', 'S6',
          'S7', 'S8', 'S9', 'S10', 'S11', 
          'S13', 'S14', 'S15', 'S16', 'S17']


progress_bar = tqdm(total=len(users), desc="User preprocessing")
for user_id in users:

    # Lettura file BVP
    bvp = pd.read_csv(f'data\\data9\\{user_id}\\BVP.csv', header=None)

    # Calcolo del vettore dei tempi delle singole misurazioni
    df_time = time_calculation(bvp)
    # Rimozione delle prime due righe
    bvp = bvp.iloc[2:, :]
    # Aggiunta delle intestazioni alle colonne
    bvp.columns = ['bvp']
    # Aggiunta della colonna dei tempi
    bvp['time'] = pd.to_datetime(df_time, unit='s')

    # Lettura file pkl
    with open(f'data\\data9\\{user_id}\\{user_id}.pkl', 'rb') as f:
        file_pkl = pickle.load(f, encoding='latin1')

    # Conversione dati bvp e etichette in liste
    #csv_time_list = bvp['time'].tolist()
    #csv_data_list = bvp['bvp'].tolist()
    pkl_data_list = file_pkl['signal']['wrist']['BVP'].tolist()
    pkl_data_list = [elemento for sublist in pkl_data_list for elemento in sublist]
    labels = file_pkl['label']

    # Ritaglio dei dati BVP del file csv in base ai valori BVP del file pkl
    # Il file csv contiene la sottosequenza dei valori del file pkl
    # Il dispositivo E4 è stato acceso prima e spento dopo il dispositivo respiban
    # Vengono scelte sottosequenze di 20 valori per determinare l'inizio e la fine
    lunghezza_sequenza = 20
    sotto_sequenza = pkl_data_list[-lunghezza_sequenza:]
    for i in range((len(bvp) - 1) - lunghezza_sequenza + 1):
        if bvp['bvp'][i:i + lunghezza_sequenza].tolist() == sotto_sequenza:
            bvp = bvp[:i+lunghezza_sequenza]
            break
    sotto_sequenza = pkl_data_list[:lunghezza_sequenza]
    for i in range((len(bvp) - 1) - lunghezza_sequenza + 1):
        if bvp['bvp'][i:i + lunghezza_sequenza].tolist() == sotto_sequenza:
            bvp = bvp[i:]
            break

    bvp = resampling(bvp, target_freq=4)

    # Associazione delle etichette con ricampionamento da 700 a 4 Hz
    rapporto = 700/4
    
    len_bvp = int((len(bvp)- (len(bvp)%4)))
    bvp = bvp.iloc[:len_bvp]

    len_labels = int(len(bvp) * rapporto)
    labels = labels[:len_labels]
    
    nuove_etichette = []
    pos = 0
    while (pos < len(labels)):
        conteggio = Counter(labels[int(pos):int(pos+rapporto)])
        valore_piu_frequente, _ = conteggio.most_common(1)[0]
        nuove_etichette.append(valore_piu_frequente)
        pos += rapporto

    # Esportazione del nuovo df
    bvp['label'] = nuove_etichette
    bvp.to_csv(f'data\\data9\\{user_id}\\BVP_LABELED.csv', index=False)

    progress_bar.update(1)
progress_bar.close()