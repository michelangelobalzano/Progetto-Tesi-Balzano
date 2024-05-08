import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from utils import generate_masks
from tqdm import tqdm
from graphs_methods import try_graph

# Calcolo della loss per la classificazione
def classification_loss(predictions, labels):

    criterion = nn.CrossEntropyLoss()

    return criterion(predictions, labels)

# Calcolo della loss per le soli sezioni mascherate
def pretraining_loss(predictions, true, masks):

    criterion = nn.MSELoss(reduction='mean')
    
    masks = ~masks # Logica maschera inversa: 1 = calcolare loss, 0 = non calcolare loss
    masked_true = torch.masked_select(true, masks)
    masked_predictions = torch.masked_select(predictions, masks)

    mse_loss = criterion(masked_predictions, masked_true)
    rmse_loss = torch.sqrt(mse_loss)

    return rmse_loss

# Train di un epoca
def train_pretrain_model(model, dataloader, optimizer, epoch):
    
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1} training", leave=False)
    for batch in dataloader:
        
        optimizer.zero_grad() # Azzeramento dei gradienti
        predictions, masks = model(batch) # Passaggio del batch al modello
        loss = pretraining_loss(predictions, batch, masks) # Calcolo della loss
        loss.backward() # Aggiornamento dei pesi
        optimizer.step()
        train_loss += loss.item() # Accumulo della loss

        progress_bar.update(1)
    progress_bar.close()

    return train_loss / num_batches

# Validation di un epoca
def validate_pretrain_model(model, dataloader, epoch):
    
    model.eval()
    val_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        
        progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1} validation", leave=False)
        for batch in dataloader:
            
            predictions, masks = model(batch) # Passaggio del batch al modello
            loss = pretraining_loss(predictions, batch, masks) # Calcolo della loss
            val_loss += loss.item() # Accumulo della loss

            progress_bar.update(1)
        progress_bar.close()

    # Calcola la loss media per la validazione
    return val_loss / num_batches

# Train di un epoca
def train_classification_model(model, dataloader, optimizer, device, epoch):
    
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(total=num_batches, desc=f"Epoch {epoch + 1} training", leave=False)
    for batch in dataloader:
        X, labels = batch
        X = X.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() # Azzeramento dei gradienti
        predictions = model(X) # Passaggio del batch al modello
        loss = classification_loss(predictions, labels) # Calcolo della loss
        loss.backward() # Aggiornamento dei pesi
        optimizer.step()
        train_loss += loss.item() # Accumulo della loss

        progress_bar.update(1)
    progress_bar.close()

    average_loss = round(train_loss / num_batches, 4)

    return average_loss

def val_classification_model(model, dataloader, device, epoch=None, task=''):

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    num_batches = len(dataloader)

    if epoch is not None:
        desc = f'Epoch {epoch + 1} {task}'
    else:
        desc = 'testing'

    progress_bar = tqdm(total=num_batches, desc=desc, leave=False)
    with torch.no_grad():
        for batch in dataloader:
            X, labels = batch
            X = X.to(device)
            labels = labels.to(device)
            predictions = model(X)
            loss = classification_loss(predictions, labels)

            val_loss += loss.item()

            _, predicted = torch.max(predictions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            progress_bar.update(1)
    progress_bar.close()

    average_loss = round(val_loss / num_batches, 4)
    accuracy = round(correct / total, 4)
    precision = round(precision_score(all_labels, all_predictions, average='weighted'), 4)
    recall = round(recall_score(all_labels, all_predictions, average='weighted'), 4)
    f1 = round(f1_score(all_labels, all_predictions, average='weighted'), 4)

    return average_loss, accuracy, precision, recall, f1

# Prova del modello con stampa del grafico delle previsioni del primo segmento del primo batch
def try_model(model, dataloader, num_signals, segment_length, iperparametri, device):
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        masks = generate_masks(iperparametri['batch_size'], iperparametri['masking_ratio'], iperparametri['lm'], num_signals, segment_length, device)
        masked_batch = first_batch * masks
        predictions = model(masked_batch)
        try_graph(first_batch[0,:,:], masks[0,:,:], predictions[0,:,:], num_signals, segment_length)