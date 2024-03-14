import torch
import torch.nn as nn
import torch.optim as optim
from data_preparation import load_data, prepare_data, generate_mask
from utils import masked_prediction_loss
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










# MAIN

# Caricamento e preparazione dei dati
data = load_data(data_path, signals)
prepared_data = prepare_data(data)

# Creazione delle maschere
mask_bvp = generate_mask(segment_length, masking_ratio, lm)
mask_eda = generate_mask(segment_length, masking_ratio, lm)
mask_hr = generate_mask(segment_length, masking_ratio, lm)

# Mascheramento dei dati
masked_input_data = {
    'bvp': [segment * mask_bvp for segment in prepared_data['bvp']],
    'eda': [segment * mask_eda for segment in prepared_data['eda']],
    'hr': [segment * mask_hr for segment in prepared_data['hr']]
}

# Definizione del modello
input_sizes = {'bvp': segment_length, 'eda': segment_length, 'hr': segment_length}
channel_embedding_output_size = 16
representation_hidden_size = 256
representation_num_heads = 2
transformation_output_size = 10
model = Transformer(input_sizes, channel_embedding_output_size, representation_hidden_size, representation_num_heads, transformation_output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Ciclo di training
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad() # Azzeramento gradienti
    outputs = model(masked_input_data) # Passaggio dati al modello
    masks = [mask_bvp, mask_eda, mask_hr] # Vettore delle maschere
    loss = masked_prediction_loss(outputs, prepared_data, masks) # Calcolo della loss
    loss.backward() # Back-propagation
    optimizer.step() # Aggiornamento pesi
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")