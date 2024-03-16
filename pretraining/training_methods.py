import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_mask


# Criterio di minimizzazione dell'errore
criterion = nn.MSELoss()

# Calcolo della loss
def masked_prediction_loss(outputs, target, mask):

    loss = 0

    for signal, output in outputs.items():

        masked_output = output * mask
        masked_target = target[signal] * mask
        loss += torch.sqrt(criterion(masked_output, masked_target))

    return loss

# Train di un epoca
def train_model(model, train_data, optimizer):
    model.train()
    train_loss = 0.0

    # Calcolo del numero dei segmenti
    num_segments = len(train_data[next(iter(train_data))])

    for i in range(num_segments):

        print('Segmento ', i)

        # Recupero dell'i-esima tripla di segmenti (uno per segnale)
        segment_data = {}
        for signal, segments in train_data.items():
            segment_data[signal] = segments[i]

        segment_masked_data, mask = apply_mask(segment_data)

        stampa_grafico(segment_data, segment_masked_data)

        # Azzeramento dei gradienti
        optimizer.zero_grad()

        # Passaggio della tripla dei segmenti mascherati al modello
        output_segments = model(segment_masked_data)

        # Calcolo della loss e aggiornamento dei pesi
        loss = masked_prediction_loss(output_segments, segment_data, mask)
        loss.backward()
        optimizer.step()

        # Accumulo della loss
        train_loss += loss.item()

    return train_loss / num_segments

# Validation di un epoca
def validate_model(model, val_data, masked_val_data, val_masks):

    model.eval()
    val_loss = 0.0
    num_segments = len(val_data[next(iter(val_data))])

    with torch.no_grad():

        for i in range(num_segments):

            # Recupero dell'i-esima tripla di segmenti (uno per segnale)
            segment_data = {}
            segment_masked_data = {}
            segment_masks = {}
            for signal, segments in val_data.items():
                segment_data[signal] = segments[i]
            for signal, segments in masked_val_data.items():
                segment_masked_data[signal] = segments[i]
            for signal, segments in val_masks.items():
                segment_masks[signal] = segments[i]

            # Passaggio della tripla di segmenti mascherati al modello
            outputs = model(segment_masked_data)

            # Calcolo della loss
            loss = masked_prediction_loss(outputs, segment_data, segment_masks)
            
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

# Metodo provvisorio per la stampa dei dati di un singolo segmento
def stampa_grafico(segment_data, segment_masked_data):

    time = np.arange(240) / 4

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, segment_data['bvp'], color='blue')
    plt.plot(time, segment_masked_data['bvp'], color='green')
    #plt.plot(time, outputs['bvp'], color='red')
    plt.title('Blood Volume Pulse (BVP)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('BVP')

    plt.subplot(3, 1, 2)
    plt.plot(time, segment_data['eda'], color='blue')
    plt.plot(time, segment_masked_data['eda'], color='green')
    #plt.plot(time, outputs['eda'], color='red')
    plt.title('Electrodermal Activity (EDA)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('EDA')

    plt.subplot(3, 1, 3)
    plt.plot(time, segment_data['hr'], color='blue')
    plt.plot(time, segment_masked_data['hr'], color='green')
    #plt.plot(time, outputs['hr'], color='red')
    plt.title('Heart Rate (HR)')
    plt.xlabel('Tempo (s)')
    plt.ylabel('HR')

    plt.tight_layout()
    plt.show()