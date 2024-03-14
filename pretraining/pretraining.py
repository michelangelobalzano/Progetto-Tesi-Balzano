import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from data_preparation import load_data, prepare_data, generate_mask
from pretraining_model import train_model, validate_model, early_stopping
from transformer import Transformer

# Percorsi dei dati
data_path = {
    'bvp': 'data/bvp.csv',
    'eda': 'data/eda.csv',
    'hr': 'data/hr.csv'
}
# Segnali considerati
signals = ['bvp', 'eda', 'hr']
# Variabili per il mascheramento
segment_length = 240 # Lunghezza dei segmenti in time steps
masking_ratio = 0.15 # Rapporto di valori mascherati
lm = 3 # Media della lunghezza dei segmenti mascherati in secondi

# Proporzione dati per la validazione sul totale 
val_data_ratio = 0.15










# MAIN

# Caricamento e preparazione dei dati
data = load_data(data_path, signals)
prepared_data = prepare_data(data)

# Suddivisione dei dati in set di training e validazione
train_data, val_data = train_test_split(prepared_data, test_size=0.15, random_state=42)

# Definizione dei dati di input per il training
train_input_data = {
    'bvp': train_data['bvp'],
    'eda': train_data['eda'],
    'hr': train_data['hr']
}

# Definizione dei dati di input per la validazione
val_input_data = {
    'bvp': val_data['bvp'],
    'eda': val_data['eda'],
    'hr': val_data['hr']
}

# Definizione dei target per il training e la validazione
train_target_data = train_data
val_target_data = val_data

# Generazione maschere
mask_bvp = generate_mask(segment_length, masking_ratio, lm)
mask_eda = generate_mask(segment_length, masking_ratio, lm)
mask_hr = generate_mask(segment_length, masking_ratio, lm)

# Definizione del modello
input_sizes = {'bvp': segment_length, 'eda': segment_length, 'hr': segment_length}
channel_embedding_output_size = 16
representation_hidden_size = 256
representation_num_heads = 2
transformation_output_size = 10
model = Transformer(input_sizes, channel_embedding_output_size, representation_hidden_size, representation_num_heads, transformation_output_size)

# Definizione dell'ottimizzatore (AdamW) e dello scheduler della velocità di apprendimento
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=True)

num_epochs = 300
best_loss = float('inf')
no_improvement_count = 0
val_losses = []  # Lista per memorizzare le loss di validazione per il controllo dell'arresto anticipato

# Ciclo di training
for epoch in range(num_epochs):
    train_loss = train_model(model, train_input_data, mask_bvp, mask_eda, mask_hr, optimizer)
    val_loss = validate_model(model, val_input_data, mask_bvp, mask_eda, mask_hr)

    # Aggiungi la loss di validazione alla lista
    val_losses.append(val_loss)

    # Stampa le informazioni sull'addestramento
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Controllo dell'arresto anticipato
    if early_stopping(val_losses):
        print("Early stopping triggered. Training halted.")
        break

    # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
    scheduler.step(val_loss)