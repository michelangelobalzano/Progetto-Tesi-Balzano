import torch
import torch.nn as nn


# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo dell'RMSE
def masked_prediction_loss(output, target, mask):

    masked_output = output * mask
    masked_target = target * mask
    loss = torch.sqrt(criterion(masked_output, masked_target))

    return loss


# Metodo per l'addestramento del modello
def train_model(model, train_data, masked_train_data, train_masks, optimizer):

    model.train()
    train_loss = 0.0

    # Calcolo del numero dei segmenti
    num_segments = len(train_data[next(iter(train_data))])

    print(f'numero segmenti: {num_segments}')

    for i in range(num_segments):

        # Recupero dell'i-esima tripla di segmenti (uno per segnale)
        segment_data = {}
        segment_masked_data = {}
        for signal, segments in train_data.items():
            segment_data[signal] = segments[i]
        for signal, segments in masked_train_data.items():
            segment_masked_data[signal] = segments[i]

        # Azzeramento dei gradienti
        optimizer.zero_grad()

        # Passaggio della tripla dei segmenti mascherati al modello
        output = model(segment_masked_data)

        # Calcolo della loss e aggiornamento dei pesi
        loss = masked_prediction_loss(output, segment_data, train_masks)
        loss.backward()
        optimizer.step()

        # Accumulo della loss
        train_loss += loss.item()

    return train_loss / num_segments







# Metodo per la validazione del modello
def validate_model(model, val_data, masked_val_data, val_masks):

    model.eval()
    val_loss = 0.0
    num_segments = len(val_data[next(iter(val_data))])

    print(f'numero segmenti: {num_segments}')

    with torch.no_grad():

        for i in range(num_segments):

            # Recupero dell'i-esima tripla di segmenti (uno per segnale)
            segment_data = {}
            segment_masked_data = {}
            for signal, segments in val_data.items():
                segment_data[signal] = segments[i]
            for signal, segments in masked_val_data.items():
                segment_masked_data[signal] = segments[i]

            # Passaggio della tripla di segmenti mascherati al modello
            output = model(segment_masked_data)

            # Calcolo della loss
            loss = masked_prediction_loss(output, segment_data, val_masks)
            
            # Accumulo della loss
            val_loss += loss.item()

    # Calcola la loss media per la validazione
    return val_loss / num_segments







# Metodo per il criterio di stop anticipato
def early_stopping(val_losses, patience=10):
    if len(val_losses) < patience + 1:
        return False

    for i in range(1, patience + 1):
        if val_losses[-i] < val_losses[-i - 1]:
            return False

    return True