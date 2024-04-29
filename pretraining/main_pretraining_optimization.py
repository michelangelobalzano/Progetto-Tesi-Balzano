import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformer
import torch
import optuna
from data_preparation import load_unlabeled_data, prepare_data, pretrain_collate_fn, pretrain_data_split
from training_methods import train_pretrain_model, validate_pretrain_model

def objective(trial, model_params, masking_params, train_dataset, val_dataset, device):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512, 1024, 2048])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    pe_type = trial.suggest_categorical('pe_type', ['fixed', 'learnable'])
    
    model_params['batch_size'] = batch_size
    model_params['d_model'] = d_model
    model_params['dim_feedforward'] = dim_feedforward
    model_params['dropout'] = dropout
    model_params['num_heads'] = num_heads
    model_params['num_layers'] = num_layers
    model_params['pe_type'] = pe_type

    train_dataloader = DataLoader(train_dataset, batch_size=model_params['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=model_params['batch_size'], shuffle=True, drop_last=True, collate_fn=pretrain_collate_fn)

    # Costruzione del modello con i parametri suggeriti
    model = TSTransformer(num_signals, segment_length, model_params, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 
                                  mode='min', 
                                  factor=0.3, 
                                  patience=10, 
                                  threshold=1e-4, 
                                  threshold_mode='rel', 
                                  cooldown=0, 
                                  min_lr=0, 
                                  eps=1e-8)
    
    # Addestramento del modello per un numero fisso di epoche
    for _ in range(num_epochs):
        _ = train_pretrain_model(model, 
                                 train_dataloader, 
                                 num_signals, 
                                 segment_length, 
                                 model_params['batch_size'],
                                 masking_params, 
                                 optimizer, 
                                 device)
        val_loss = validate_pretrain_model(model, 
                                           val_dataloader, 
                                           num_signals, 
                                           segment_length, 
                                           model_params['batch_size'], 
                                           masking_params, 
                                           device)
        
        scheduler.step(val_loss)
    
    # Restituisci l'ultimo valore di perdita di validazione
    return val_loss

signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
data_directory = 'processed_data\\' # Percorso dei dati
split_ratios = [85, 15] # Split ratio dei segmenti (train/val)
num_epochs = 10 # Numero epoche task pretraining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_unlabeled_data(data_directory, signals)
train, val = pretrain_data_split(data, split_ratios, signals)
num_train_segments = len(train[next(iter(train))].groupby('segment_id'))
num_val_segments = len(val[next(iter(val))].groupby('segment_id'))
train_data = prepare_data(train, num_signals, num_train_segments, segment_length)
val_data = prepare_data(val, num_signals, num_val_segments, segment_length)
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

model_params = {}
masking_params = {'lm': 12, 'masking_ratio': 0.15}
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, 
                                       model_params, 
                                       masking_params, 
                                       train_dataset, 
                                       val_dataset, 
                                       device), n_trials=10)
print('Migliori iperparametri:', study.best_params)
print('Miglior valore di perdita:', study.best_value)