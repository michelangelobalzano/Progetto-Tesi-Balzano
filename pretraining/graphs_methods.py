import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np


def losses_graph(epoch_info, path):

    # Plot delle curve di loss di training e validazione
    plt.plot(epoch_info['train_losses'], label='Train Loss')
    plt.plot(epoch_info['val_losses'], label='Val Loss')

    # Aggiungi etichette agli assi e una legenda
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Salvataggio grafico
    plt.savefig(path)

    # Mostra il grafico
    plt.show()

def try_graph(original, mask, prediction, num_signals, segment_length):

    frequency = 4
    time = np.arange(segment_length) / frequency
    zero_intervals = [[] for _ in range(num_signals)]
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask

    for s in range(num_signals):
        start_idx = None
        for i in range(segment_length):
            if mask[s,i] == False and start_idx is None:
                start_idx = i
            elif mask[s,i] == True and start_idx is not None:
                zero_intervals[s].append((start_idx, i - 1))
                start_idx = None

        if start_idx is not None:
            zero_intervals[s].append((start_idx, segment_length - 1))

    plt.figure(figsize=(10, 18))

    signal_names = ['BVP', 'EDA', 'HR']
    for s in range(num_signals):
        plt.subplot(num_signals, 1, s+1)

        '''# Sezioni originali non mascherate
        original_unmasked = np.ma.masked_array(original[s].cpu().numpy(), ~mask[s])
        plt.plot(time, original_unmasked, color='blue', linewidth=1.0, label='Valori non mascherati')

        # Sezioni originali mascherate
        original_masked = np.ma.masked_array(original[s].cpu().numpy(), mask[s])
        plt.plot(time, original_masked, color='green', linewidth=1.0, label='Valori non mascherati')

        # Previsioni delle sezioni mascherate
        prediction_masked = np.ma.masked_array(prediction[s].detach().cpu().numpy().flatten(), mask[s])
        plt.plot(time, prediction_masked, color='red', linewidth=1.0, label='Predizioni')

        # Area grigia sezioni mascherate'''

        plt.plot(time, original[s].cpu().numpy(), color='blue', linewidth=0.5)
        plt.plot(time, prediction[s].detach().cpu().numpy().flatten(), color='red', linewidth=0.5)
        min = np.min([np.min(original[s].cpu().numpy()), np.min(prediction[s].detach().cpu().numpy())])
        max = np.max([np.max(original[s].cpu().numpy()), np.max(prediction[s].detach().cpu().numpy())])
        y_values = [min, max]
        for start, end in zero_intervals[s]:
            plt.fill_betweenx(y_values, start/4, end/4, color='grey')
        
        plt.title(signal_names[s])
        plt.xlabel('Tempo (s)')
        plt.ylabel(signal_names[s])

    plt.tight_layout()
    plt.show()