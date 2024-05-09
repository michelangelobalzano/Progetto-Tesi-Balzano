import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformer import TSTransformerClassifier
import torch
import time
from datetime import datetime
from data_preparation import get_classification_dataloaders
from training_methods import train_classification_model, val_classification_model
from graphs_methods import losses_graph
from utils import load_pretrained_model, load_pretrained_model_params, save_classification_info, save_model
from options import Options

def main(config):

    # Check disponibilità GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU disponibili: {device_count}")
        
        current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"GPU in uso: {current_device_name}")
    else:
        print("GPU non disponibile. Si sta utilizzando la CPU.")

    # Caricamento dati e creazione dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_classification_dataloaders(config, device)

    # Caricamento modello preaddestrato
    if config['model_to_load'] != '':
        model_name = config['model_to_load']
        # Caricamento parametri modello ed etichetta da classificare
        config = load_pretrained_model_params(config)
        model = TSTransformerClassifier(config, device)
        # Caricamento modello
        model = load_pretrained_model(model, config)
    else:
        model = TSTransformerClassifier(config, device)
        current_datetime = datetime.now()
        model_name = current_datetime.strftime("%m-%d_%H-%M")
    model = model.to(device)

    # Definizione dell'ottimizzatore (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Definizione dello scheduler di apprendimento
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min', 
                                factor=config['factor'], 
                                patience=config['patience'], 
                                threshold=config['threshold'], 
                                threshold_mode='rel', 
                                cooldown=0, 
                                min_lr=0, 
                                eps=1e-8)

    # Ciclo di training
    epoch_info = {'train_losses': [], 'val_losses': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1score': []}
    test_info = {}
    start_time = time.time()
    num_lr_reductions = 0
    current_lr = config['learning_rate']

    for epoch in range(config['num_epochs']):
        # Training
        train_loss = train_classification_model(model, 
                                                train_dataloader, 
                                                optimizer, 
                                                device, 
                                                epoch)
        # Validation
        val_loss, accuracy, precision, recall, f1 = val_classification_model(model, 
                                                        val_dataloader, 
                                                        device, 
                                                        epoch=epoch, 
                                                        task='validation')

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1Score: {f1}")

        # Salvataggio delle informazioni dell'epoca
        epoch_info['train_losses'].append(train_loss)
        epoch_info['val_losses'].append(val_loss)
        epoch_info['accuracy'].append(accuracy)
        epoch_info['precision'].append(precision)
        epoch_info['recall'].append(recall)
        epoch_info['f1score'].append(f1)

        # Aggiorna lo scheduler della velocità di apprendimento in base alla loss di validazione
        scheduler.step(val_loss)
        if scheduler._last_lr != current_lr:
            num_lr_reductions += 1
            current_lr = scheduler._last_lr
        if num_lr_reductions > config['max_lr_reductions']:
            print('Early stopped')
            break

        # Ogni tot epoche effettua un salvataggio del modello
        if config['num_epochs_to_save'] is not None:
            if (epoch + 1) % config['num_epochs_to_save'] == 0 and epoch > 0:
                now = time.time()
                elapsed_time = now - start_time
                save_classification_info(model_name, 
                                        config, 
                                        epoch_info, 
                                        epoch+1, 
                                        elapsed_time)

    # Test
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = val_classification_model(model, 
                                                        test_dataloader, 
                                                        device, 
                                                        task='testing')
    test_info['test_loss'] = test_loss
    test_info['accuracy'] = test_accuracy
    test_info['precision'] = test_precision
    test_info['recall'] = test_recall
    test_info['f1score'] = test_f1
    print(f"Test Loss: {test_loss}, Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1Score: {test_f1}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    save_classification_info(model_name, 
                            config,
                            epoch_info, 
                            epoch+1, 
                            elapsed_time, 
                            test_info)
        
    losses_graph(epoch_info, save_path=f'graphs\\losses_plot_{model_name}.png')

args = Options().parse()
config = args.__dict__
config['signals'] = ['BVP', 'EDA', 'HR']
config['num_signals'] = 3
config['segment_length'] = 240
config['num_classes'] = 3
main(config)