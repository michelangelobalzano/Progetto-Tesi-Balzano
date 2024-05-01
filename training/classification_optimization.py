import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import TSTransformerClassifier
import torch
import optuna
from data_preparation import get_classification_dataloaders
from training_methods import train_classification_model, val_classification_model
from options import Options

def objective(trial, model_params, device):
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
    
    train_dataloader, val_dataloader, _ = get_classification_dataloaders(config, device)

    model = TSTransformerClassifier(config, device)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min', 
                                factor=config['factor'], 
                                patience=config['patience'], 
                                threshold=config['threshold'], 
                                threshold_mode='rel', 
                                cooldown=0, 
                                min_lr=0, 
                                eps=1e-8)

    for _ in range(config['num_optimization_epochs']):
        _ = train_classification_model(model, train_dataloader, optimizer, device)
        val_loss, val_accuracy = val_classification_model(model, val_dataloader, device, task='Validation')
        
        scheduler.step(val_loss)
    
    return val_accuracy

def main(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, 
                                        config, 
                                        device), 
                                        n_trials=config['num_optimization_trials'])
    print('Migliori iperparametri:', study.best_params)
    print('Miglior valore di perdita:', study.best_value)

args = Options().parse()
config = args.__dict__
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
config['num_classes'] = 3
main(config)