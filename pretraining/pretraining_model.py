import torch
import torch.nn as nn

from utils import masked_prediction_loss

# Metodo per l'addestramento del modello
def train_model(model, train_input_data, mask_bvp, mask_eda, mask_hr, optimizer):
    model.train()
    train_loss = 0.0

    for segment_data in zip(train_input_data['bvp'], train_input_data['eda'], train_input_data['hr']):
        optimizer.zero_grad()  # Azzeramento dei gradienti

        # Passaggio dei segmenti mascherati al modello
        output = model({
            'bvp': segment_data[0] * mask_bvp,
            'eda': segment_data[1] * mask_eda,
            'hr': segment_data[2] * mask_hr
        })

        # Calcola la loss e aggiorna i pesi del modello
        loss = masked_prediction_loss(output, segment_data, [mask_bvp, mask_eda, mask_hr])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # Accumula la loss

    return train_loss / len(train_input_data['bvp'])

# Metodo per la validazione del modello
def validate_model(model, val_input_data, mask_bvp, mask_eda, mask_hr):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for segment_data in zip(val_input_data['bvp'], val_input_data['eda'], val_input_data['hr']):
            
            # Passaggio dei segmenti non mascherati al modello
            output = model({
                'bvp': segment_data[0],
                'eda': segment_data[1],
                'hr': segment_data[2]
            })

            # Calcola la loss di validazione
            loss = masked_prediction_loss(output, segment_data, [mask_bvp, mask_eda, mask_hr])
            val_loss += loss.item()

    return val_loss / len(val_input_data['bvp'])

# Metodo per il criterio di stop anticipato
def early_stopping(val_losses, patience=10):
    if len(val_losses) < patience + 1:
        return False

    for i in range(1, patience + 1):
        if val_losses[-i] < val_losses[-i - 1]:
            return False

    return True