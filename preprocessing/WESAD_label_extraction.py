import pickle
import pandas as pd
from collections import Counter
from tqdm import tqdm
import numpy as np

from preprocessing_methods import time_calculation, resampling, get_users

task_numbers = {
    'Base': 1,
    'TSST': 2,
    'Fun': 3,
    'Medi 1' :4,
    'Medi 2' :5
}
valori_task_inutili = [5, 6, 7]

def extract_WESAD_labels (config, users):

    progress_bar = tqdm(total=len(users), desc="Estrazione etichette", leave=False)
    for user_id in users:

        print(f'utente: {user_id}')

        # Lettura file BVP
        bvp = pd.read_csv(f'{config["data_directory"]}{user_id}\\BVP.csv', header=None)

        # Calcolo del vettore dei tempi delle singole misurazioni
        df_time = time_calculation(bvp)
        # Rimozione delle prime due righe
        bvp = bvp.iloc[2:, :]
        # Aggiunta delle intestazioni alle colonne
        bvp.columns = ['bvp']
        # Aggiunta della colonna dei tempi
        bvp['time'] = pd.to_datetime(df_time, unit='s')

        # Lettura file pkl
        with open(f'{config["data_directory"]}{user_id}\\{user_id}.pkl', 'rb') as f:
            file_pkl = pickle.load(f, encoding='latin1')

        pkl_data_list = file_pkl['signal']['wrist']['BVP'].tolist()
        pkl_data_list = [elemento for sublist in pkl_data_list for elemento in sublist]
        tasks = file_pkl['label']

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

        bvp = resampling(bvp, target_freq=config['resampling_frequency'])
        bvp['bvp'] = bvp['bvp'].round(2)

        # Associazione delle etichette con ricampionamento da 700 a resampling_frequency Hz
        rapporto = 700/config['resampling_frequency']
        
        # Ricalcolo lunghezza segnale bvp per farlo terminare al secondo preciso
        len_bvp = int((len(bvp)- (len(bvp)%config['resampling_frequency'])))
        bvp = bvp.iloc[:len_bvp]

        # Ricalcolo lunghezza tasks per farle terminare al secondi preciso di bvp
        len_tasks = int(len(bvp) * rapporto)
        tasks = tasks[:len_tasks]
        
        # Ricalcolo delle etichette da 700Hz a resampling_frequency Hz in base al valore più frequente per ogni intervallo
        nuove_etichette = []
        pos = 0
        while (pos < len(tasks)):
            conteggio = Counter(tasks[int(pos):int(pos+rapporto)])
            valore_piu_frequente, _ = conteggio.most_common(1)[0]
            nuove_etichette.append(valore_piu_frequente)
            pos += rapporto

        # Esportazione del nuovo df
        bvp['task'] = nuove_etichette

        # Sostituzione valori non utili con uno 0 (nessuna task)
        bvp['task'] = bvp['task'].replace(valori_task_inutili, 0)

        # Per ogni utente ci sono due task di meditazione contrassegnate con valore 4
        # Alle due task corrispondono etichette diverse. Occorre differenziarle+
        # Sostituzione valore seconda task di meditazione (4) con un 5
        stato = 0
        idx_inizio_sostituzione = None
        for i, valore in enumerate(bvp['task']):
            nuovo_stato = valore
            if nuovo_stato != stato and stato == 4:
                idx_inizio_sostituzione = i + 2
                break
            stato = nuovo_stato
        bvp.loc[idx_inizio_sostituzione:, 'task'] = bvp.loc[idx_inizio_sostituzione:, 'task'].replace(4, 5)

        # Lettura etichette e task corrispondenti da file csv
        valence = []
        arousal = []
        tasks_names = []
        tasks = []
        label_df = pd.read_csv(f'{config["data_directory"]}{user_id}\\{user_id}_quest.csv', sep=';', header=None)
        tasks_names = label_df.iloc[1, 1:6].tolist()
        valence = label_df.iloc[17:22, 1].tolist()
        arousal = label_df.iloc[17:22, 2].tolist()
        for i in range(len(tasks_names)):
            tasks.append(task_numbers[tasks_names[i]])

        # Standardizzazione delle etichette di valenza e attivazione
        valence = list(map(int, valence))
        arousal = list(map(int, arousal))
        valence_std = []
        val_mean = np.mean(valence)
        for i in range(len(valence)):
            if (valence[i] >= val_mean + config['neutral_range']):
                valence_std.append('positive')
            elif (valence[i] <= val_mean - config['neutral_range']):
                valence_std.append('negative')
            else:
                valence_std.append('neutral')
        arousal_std = []
        aro_mean = np.mean(arousal)
        for i in range(len(arousal)):
            if (arousal[i] >= aro_mean + config['neutral_range']):
                arousal_std.append('positive')
            elif (arousal[i] <= aro_mean - config['neutral_range']):
                arousal_std.append('negative')
            else:
                arousal_std.append('neutral')
            
        # Associazione delle etichette ai task corrispondenti
        bvp_not_std = bvp
        for i, valore in enumerate(bvp['task']):
            if valore in tasks:
                bvp.loc[i, 'valence'] = valence_std[tasks.index(valore)]
                bvp.loc[i, 'arousal'] = arousal_std[tasks.index(valore)]
                bvp_not_std.loc[i, 'valence'] = valence[tasks.index(valore)]
                bvp_not_std.loc[i, 'arousal'] = arousal[tasks.index(valore)]
            else:
                bvp.loc[i, 'valence'] = None
                bvp_not_std.loc[i, 'valence'] = None

        bvp = bvp.drop(['task'], axis=1)
        bvp_not_std = bvp_not_std.drop(['task'], axis=1)

        bvp.to_csv(f'{config["data_directory"]}{user_id}\\BVP_LABELED.csv', index=False)
        bvp_not_std.to_csv(f'{config["data_directory"]}{user_id}\\BVP_LABELED_NOT_STD.csv', index=False)

        progress_bar.update(1)
    progress_bar.close()