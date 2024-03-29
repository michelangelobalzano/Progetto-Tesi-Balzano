import torch
import torch.nn as nn
import numpy as np
from utils import generate_masks
from tqdm import tqdm
from graphs_methods import try_graph
from torch.nn import functional as F


# Calcolo della loss per il pre-training con masked prediction
def masked_prediction_loss(predictions, true, masks):

    criterion = nn.MSELoss(reduction='mean')# Criterio di minimizzazione dell'errore
    
    masked_true = torch.masked_select(true, masks)
    masked_predictions = torch.masked_select(predictions, masks)

    return criterion(masked_predictions, masked_true)

# Calcolo della loss per la classificazione
def cross_entropy_loss(self, inp, target):
    return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                            ignore_index=self.ignore_index, reduction=self.reduction)

# Train di un epoca
def train_pretrain_model(model, dataloader, num_signals, segment_length, iperparametri, optimizer, device):
    
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(total=num_batches, desc="Train batch analizzati")
    for batch in dataloader:
        
        optimizer.zero_grad() # Azzeramento dei gradienti
        masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], 
                               iperparametri['lm'], num_signals, segment_length, device) # Generazione delle maschere
        masked_batch = batch * masks # Applicazione della maschera
        predictions = model(masked_batch) # Passaggio del batch al modello
        loss = masked_prediction_loss(predictions, batch, masks) # Calcolo della loss
        loss.backward() # Aggiornamento dei pesi
        optimizer.step()
        train_loss += loss.item() # Accumulo della loss

        progress_bar.update(1)
    progress_bar.close()

    return train_loss / num_batches, model

# Validation di un epoca
def validate_pretrain_model(model, dataloader, num_signals, segment_length, iperparametri, device):
    
    model.eval()
    val_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        
        progress_bar = tqdm(total=num_batches, desc="Val batch analizzati")
        for batch in dataloader:
            
            masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], 
                                   iperparametri['lm'], num_signals, segment_length, device) # Generazione delle maschere
            masked_batch = batch * masks # Applicazione della maschera
            predictions = model(masked_batch) # Passaggio del batch al modello
            loss = masked_prediction_loss(predictions, batch, masks) # Calcolo della loss
            val_loss += loss.item() # Accumulo della loss

            progress_bar.update(1)
        progress_bar.close()

    # Calcola la loss media per la validazione
    return val_loss / num_batches, model

''' 
# Metodo per il criterio di stop anticipato
def early_stopping(val_losses, patience=10):
    if len(val_losses) < patience + 1:
        return False

    for i in range(1, patience + 1):
        if val_losses[-i] < val_losses[-i - 1]:
            return False

    return True
'''

# Prova del modello con stampa del grafico delle previsioni del primo segmento del primo batch
def try_model(model, dataloader, num_signals, segment_length, iperparametri, device):
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], iperparametri['lm'], num_signals, segment_length, device)
        masked_batch = first_batch * masks
        predictions = model(masked_batch)
        try_graph(first_batch[0,:,:], masks[0,:,:], predictions[0,:,:], num_signals, segment_length)