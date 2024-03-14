import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np

# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo dell'RMSE
def masked_prediction_loss(outputs, targets, masks):
    loss = 0
    for output, target, mask in zip(outputs, targets, masks):
        loss += torch.sqrt(criterion(output * mask, target * mask))
    return loss / len(outputs)

# Suddivisione dei segmenti in train e val
def data_split(data, split_ratio, signals):
    segment_ids = data['bvp']['segment_id'].unique()
    train_segment_ids, val_segment_ids = train_test_split(segment_ids, train_size=split_ratio, random_state=42)

    train_data = {}
    val_data = {}
    for signal in signals:
        train_data[signal] = data[signal][data[signal]['segment_id'].isin(train_segment_ids)].reset_index(drop=True)
        val_data[signal] = data[signal][data[signal]['segment_id'].isin(val_segment_ids)].reset_index(drop=True)

    return train_data, val_data

# Generazione della maschera booleana
def generate_mask(segment_length, masking_ratio, lm):
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