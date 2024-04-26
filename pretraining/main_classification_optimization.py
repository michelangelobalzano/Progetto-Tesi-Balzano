import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from transformer import TSTransformerClassifier
import torch
import optuna
from tqdm import tqdm
from data_preparation import load_labeled_data, prepare_classification_data, classification_collate_fn, classification_data_split
from training_methods import train_classification_model, val_classification_model

def objective(trial, iperparametri, train_dataset, val_dataset, device):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)

    iperparametri['batch_size'] = batch_size
    iperparametri['d_model'] = d_model
    iperparametri['dropout'] = dropout
    iperparametri['num_heads'] = num_heads
    iperparametri['num_layers'] = num_layers

    train_dataloader = DataLoader(train_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=iperparametri['batch_size'], shuffle=True, drop_last=True, collate_fn=classification_collate_fn)

    model = TSTransformerClassifier(num_signals, segment_length, iperparametri, num_classes, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)
    val_losses = []
    accuracy = []

    progress_bar = tqdm(total=num_epochs, desc=f"epoche")
    for _ in range(num_epochs):
        _, model = train_classification_model(model, train_dataloader, optimizer, device)
        val_loss, val_accuracy, model = val_classification_model(model, val_dataloader, device, task='Validation')
        val_losses.append(val_loss)
        accuracy.append(val_accuracy)
        scheduler.step(val_loss)
        progress_bar.update(1)
    progress_bar.close()
    
    return accuracy[-1]

signals = ['BVP', 'EDA', 'HR'] # Segnali considerati
num_signals = len(signals) # Numero dei segnali
segment_length = 240 # Lunghezza dei segmenti in time steps
num_classes = 3 # 'negative', 'positive', 'neutral'
data_directory = 'processed_data\\' # Percorso dei dati
split_ratios = [70, 15, 15] # Split ratio dei segmenti (train/val/test)
num_epochs = 15 # Numero epoche task classification
label = 'valence' # Etichetta da classificare ('valence'/'arousal')
num_trials = 15

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, labels = load_labeled_data(data_directory, signals, label)
train, train_labels, val, val_labels, _, _ = classification_data_split(data, labels, split_ratios, signals)
num_train_segments = len(train[signals[0]].groupby('segment_id'))
num_val_segments = len(val[signals[0]].groupby('segment_id'))
train_data, train_labels = prepare_classification_data(train, train_labels, num_signals, num_train_segments, segment_length, label)
val_data, val_labels = prepare_classification_data(val, val_labels, num_signals, num_val_segments, segment_length, label)
train_data = train_data.permute(1, 0, 2).to(device)
val_data = val_data.permute(1, 0, 2).to(device)
train_labels = train_labels.to(device)
val_labels = val_labels.to(device)
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

iperparametri = {}
iperparametri['lm'] = 12
iperparametri['masking_ratio'] = 0.15

study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, iperparametri, train_dataset, val_dataset, device), n_trials=num_trials)
print('Migliori iperparametri:', study.best_params)
print('Miglior valore di perdita:', study.best_value)