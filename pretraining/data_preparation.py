import pandas as pd
import torch

# Caricamento dei dati
def load_data(file_path, signals):

    data = {}
    for signal in signals:
        data[signal] = pd.read_csv(file_path[signal])
    return data

# Preparazione dei dati
# Conversione di ogni segmento di dati in un tensore
# L'output è un dizionario avente come chiave il nome del sensore e come valore la lista di segmenti convertita in tensori
# {
#     'bvp': [tensor_segment_1, tensor_segment_2, ...],
#     'eda': [tensor_segment_1, tensor_segment_2, ...],
#     'hr': [tensor_segment_1, tensor_segment_2, ...]
# }
def prepare_data(data):
    
    prepared_data = {}
    for signal, df in data.items():
        segments = []
        for segment in df.groupby('segment_id'):
            segments.append(torch.tensor(segment[1].iloc[:, :-1].values, dtype=torch.float32))
        prepared_data[signal] = segments
    return prepared_data