import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# Calcolo della loss
def loss_function(predictions, labels):

    criterion = nn.CrossEntropyLoss()

    return criterion(predictions, labels)

# Training di un epoca
def train_model(model, dataloader, optimizer, device, epoch):
    
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
        loss = loss_function(predictions, labels) # Calcolo della loss
        loss.backward() # Aggiornamento dei pesi
        optimizer.step()
        train_loss += loss.item() # Accumulo della loss

        progress_bar.update(1)
    progress_bar.close()

    average_loss = round(train_loss / num_batches, 4)

    return average_loss

# Validation di un epoca
def val_model(model, dataloader, device, epoch=None, task=''):

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
            loss = loss_function(predictions, labels)

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
    precision = round(precision_score(all_labels, all_predictions, average='weighted', zero_division=1), 4)
    recall = round(recall_score(all_labels, all_predictions, average='weighted', zero_division=1), 4)
    f1 = round(f1_score(all_labels, all_predictions, average='weighted', zero_division=1), 4)

    return average_loss, accuracy, precision, recall, f1