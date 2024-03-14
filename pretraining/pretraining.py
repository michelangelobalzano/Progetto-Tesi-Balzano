import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_preparation import load_data, prepare_data
from pretraining_model import train_model, validate_model, early_stopping
from transformer import Transformer
from utils import data_split, generate_mask, apply_mask

# Percorsi dei dati
data_path = {
    'bvp': 'processed_data/bvp.csv',
    'eda': 'processed_data/eda.csv',
    'hr': 'processed_data/hr.csv'
}
# Segnali considerati
signals = ['bvp', 'eda', 'hr']
sampling_frequency = 4 # Hz
# Variabili per il mascheramento
segment_length = 240 # Lunghezza dei segmenti in time steps
masking_ratio = 0.15 # Rapporto di valori mascherati
lm = 12 # Media della lunghezza dei segmenti mascherati in secondi

# Proporzione dati per la validazione sul totale 
train_data_ratio = 0.85










# MAIN

# Caricamento e preparazione dei dati
data = load_data(data_path, signals)

# Suddivisione dei dati in train e val
train_data, val_data = data_split(data, train_data_ratio, signals)

# Prepara i dati suddivisi
prepared_train_data = prepare_data(train_data)
prepared_val_data = prepare_data(val_data)

'''# Definizione dei dati di input per il training
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
'''


# Definizione del modello
input_sizes = {'bvp': segment_length, 'eda': segment_length, 'hr': segment_length}
channel_embedding_output_size = 16
representation_hidden_size = 256
representation_num_heads = 2
transformation_output_size = 10
model = Transformer(input_sizes, sampling_frequency, channel_embedding_output_size, representation_hidden_size, representation_num_heads, transformation_output_size)

# Definizione dell'ottimizzatore (AdamW) e dello scheduler della velocità di apprendimento
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=True)

num_epochs = 300
val_losses = []  # Lista per memorizzare le loss di validazione per il controllo dell'arresto anticipato

# Ciclo di training
for epoch in range(num_epochs):

    masked_data = apply_mask(prepared_train_data)
    train_loss = train_model(model, prepared_train_data, masked_data, optimizer)
    val_loss = validate_model(model, prepared_val_data, mask_bvp, mask_eda, mask_hr)

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