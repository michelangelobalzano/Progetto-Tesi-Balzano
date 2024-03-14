import pandas as pd
import numpy as np
import torch

# Caricamento dei dati
def load_data(file_path, signals):

    data = {}
    for signal in signals:
        data[signal] = pd.read_csv(file_path[signal])
    return data

# Preparazione dei dati
# Conversione di ogni segmento di dati in un tensore
def prepare_data(data):
    
    prepared_data = {}
    for signal, df in data.items():
        segments = []
        for segment in df.groupby('segment_id'):
            segments.append(torch.tensor(segment[1].iloc[:, :-1].values, dtype=torch.float32))
        prepared_data[signal] = segments
    return prepared_data

# Generazione della maschera booleana
def generate_mask(segment_length, masking_ratio, lm, lu):
    masks = []
    for _ in range(segment_length):
        # Genera casualmente una maschera binaria basata sulla proporzione di mascheramento
        mask = np.random.choice([0, 1], size=segment_length, p=[masking_ratio, 1 - masking_ratio])
        # Cerca la prima occorrenza di uno nella maschera
        first_one_index = np.argmax(mask)
        # Genera casualmente la lunghezza degli zeri seguendo una distribuzione geometrica con media lm
        zero_length = np.random.geometric(1 / lm)
        # Modifica la maschera in modo che ci siano zeri fino alla lunghezza calcolata o fino alla prima occorrenza di uno
        mask[:first_one_index] = 0
        mask[first_one_index:first_one_index + zero_length] = 0
        # La parte rimanente della maschera sarà 1 (non mascherata)
        mask[first_one_index + zero_length:] = 1
        masks.append(mask)
    return masks