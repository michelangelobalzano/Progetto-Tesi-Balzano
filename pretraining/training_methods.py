import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_mask
from tqdm import tqdm


# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo della loss
def masked_prediction_loss(outputs, target, masks, batch_size):
    
    loss = 0
    num_signals = len(outputs)

    for i in range(batch_size):
        for signal, output in outputs.items():
            masked_output = output[:, i, :] * (1 - masks[i, :])
            masked_target = target[signal][:, i, :] * (1 - masks[i, :])
            loss += torch.sqrt(criterion(masked_output, masked_target))

    return loss / (num_signals * batch_size)

# Train di un epoca
def train_model(model, dataloader, batch_size, optimizer, device):
    model.train()
    train_loss = 0.0
    num_batches = len(dataloader)

    progress_bar = tqdm(total=num_batches, desc="Train batch analizzati")
    for batch in dataloader:
        batch = {signal: data.to(device) for signal, data in batch.items()}

        # Azzeramento dei gradienti
        optimizer.zero_grad()

        masked_batch, masks = apply_mask(batch, batch_size, device)

        # Passaggio del batch al modello
        output_segments = model(masked_batch)

        # Stampa i grafici
        '''
        for i in range(32):
            segment_data = {signal: data[i] for signal, data in batch.items()}
            segment_outputs = {signal: data[i] for signal, data in output_segments.items()}
            stampa_grafico(segment_data, segment_outputs, masks[i])
        '''
        # Calcolo della loss
        loss = masked_prediction_loss(output_segments, batch, masks, batch_size)

        # Aggiornamento dei pesi
        loss.backward()
        optimizer.step()

        # Accumulo della loss
        train_loss += loss.item()

        progress_bar.update(1)
    progress_bar.close()

    return train_loss / num_batches, model

# Validation di un epoca
def validate_model(model, dataloader, batch_size, device):
    model.eval()
    val_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        
        progress_bar = tqdm(total=num_batches, desc="Val batch analizzati")
        for batch in dataloader:
            batch = {signal: data.to(device) for signal, data in batch.items()}

            masked_batch, masks = apply_mask(batch, batch_size, device)

            # Passaggio del batch al modello
            output_segments = model(masked_batch)

            # Calcolo della loss
            loss = masked_prediction_loss(output_segments, batch, masks, batch_size)

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

# Metodo provvisorio per la stampa dei dati di un singolo segmento
def stampa_grafico(segment_data, outputs, mask):

    print(segment_data['bvp'])
    print(outputs['bvp'])
    print(mask)

    time = np.arange(240) / 4

    zero_intervals = []
    start_idx = None
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    for i, val in enumerate(mask):
        if val == 0 and start_idx is None:
            start_idx = i
        elif val == 1 and start_idx is not None:
            zero_intervals.append((start_idx, i - 1))
            start_idx = None

    if start_idx is not None:
        zero_intervals.append((start_idx, len(mask) - 1))

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, segment_data['bvp'].cpu().numpy(), color='blue')
    plt.plot(time, outputs['bvp'].detach().cpu().numpy().flatten(), color='red')
    min = np.min([np.min(segment_data['bvp'].cpu().numpy()[:, 0]), np.min(outputs['bvp'].detach().cpu().numpy()[:, 0])])
    max = np.max([np.max(segment_data['bvp'].cpu().numpy()[:, 0]), np.max(outputs['bvp'].detach().cpu().numpy()[:, 0])])
    y_values = [min, max]
    for start, end in zero_intervals:
        plt.fill_betweenx(y_values, start/4, end/4, color='grey')
    plt.title('Blood Volume Pulse (BVP)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('BVP')

    plt.subplot(3, 1, 2)
    plt.plot(time, segment_data['eda'].cpu().numpy(), color='blue')
    plt.plot(time, outputs['eda'].detach().cpu().numpy().flatten(), color='red')
    min = np.min([np.min(segment_data['eda'].cpu().numpy()[:, 0]), np.min(outputs['eda'].detach().cpu().numpy()[:, 0])])
    max = np.max([np.max(segment_data['eda'].cpu().numpy()[:, 0]), np.max(outputs['eda'].detach().cpu().numpy()[:, 0])])
    y_values = [min, max]
    for start, end in zero_intervals:
        plt.fill_betweenx(y_values, start/4, end/4, color='grey')
    plt.title('Electrodermal Activity (EDA)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('EDA')

    plt.subplot(3, 1, 3)
    plt.plot(time, segment_data['hr'].cpu().numpy(), color='blue')
    min = np.min([np.min(segment_data['hr'].cpu().numpy()[:, 0]), np.min(outputs['hr'].detach().cpu().numpy()[:, 0])])
    max = np.max([np.max(segment_data['hr'].cpu().numpy()[:, 0]), np.max(outputs['hr'].detach().cpu().numpy()[:, 0])])
    y_values = [min, max]
    for start, end in zero_intervals:
        plt.fill_betweenx(y_values, start/4, end/4, color='grey')
    plt.plot(time, outputs['hr'].detach().cpu().numpy().flatten(), color='red')
    plt.title('Heart Rate (HR)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('HR')

    plt.tight_layout()
    plt.show()