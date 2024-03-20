import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks
from tqdm import tqdm
from graphs_methods import try_graph


# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss(reduction='mean')

# Calcolo della loss
def masked_prediction_loss(predictions, true, masks):
    
    masked_predictions = predictions * masks
    masked_true = true * masks

    return criterion(masked_predictions, masked_true)

# Train di un epoca
def train_model(model, dataloader, num_signals, segment_length, iperparametri, optimizer, device):
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(total=num_batches, desc="Train batch analizzati")
    for batch in dataloader:

        # Azzeramento dei gradienti
        optimizer.zero_grad()

        # Generazione delle maschere
        masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], iperparametri['lm'], num_signals, segment_length, device)

        # Applicazione della maschera
        masked_batch = batch * masks

        # Passaggio del batch al modello
        predictions = model(masked_batch)

        # Stampa dimensioni tensori per prova
        #print('input size: ', batch.size())
        #print('masked batch size: ', masked_batch.size())
        #print('predictions size: ', predictions.size())
        # Stampa dati del primo segmento del batch del primo segnale per prova
        #print('input: ', batch[0,0,:])
        #print('mask: ', masks[0,0,:])
        #input()
        #print('masked input: ', masked_batch[0,0,:])
        #print('predictions: ', predictions[0,0,:])

        # Calcolo della loss
        loss = masked_prediction_loss(predictions, batch, masks)

        # Aggiornamento dei pesi
        loss.backward()
        optimizer.step()

        # Accumulo della loss
        train_loss += loss.item()

        progress_bar.update(1)
    progress_bar.close()

    return train_loss / num_batches, model

# Validation di un epoca
def validate_model(model, dataloader, num_signals, segment_length, iperparametri, device):
    
    model.eval()
    val_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        
        progress_bar = tqdm(total=num_batches, desc="Val batch analizzati")
        for batch in dataloader:
            
            # Generazione delle maschere
            masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], iperparametri['lm'], num_signals, segment_length, device)

            # Applicazione della maschera
            masked_batch = batch * masks

            # Passaggio del batch al modello
            predictions = model(masked_batch)

            # Calcolo della loss
            loss = masked_prediction_loss(predictions, batch, masks)

            # Accumulo della loss
            val_loss += loss.item()

            progress_bar.update(1)
        progress_bar.close()

    # Calcola la loss media per la validazione
    return val_loss / num_batches, model
    
# Metodo per il criterio di stop anticipato
def early_stopping(val_losses, patience=10):
    if len(val_losses) < patience + 1:
        return False

    for i in range(1, patience + 1):
        if val_losses[-i] < val_losses[-i - 1]:
            return False

    return True

def try_model(model, dataloader, num_signals, segment_length, iperparametri, device):
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], iperparametri['lm'], num_signals, segment_length, device)
        masked_batch = first_batch * masks
        predictions = model(masked_batch)
        try_graph(first_batch[0,:,:], masks[0,:,:], predictions[0,:,:], num_signals, segment_length)