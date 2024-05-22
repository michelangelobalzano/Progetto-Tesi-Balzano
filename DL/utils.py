import torch
import random
import csv

# Salvataggio info classificazione
def save_session_info(model_name, 
                             config,
                             epoch_info,
                             num_epochs,
                             elapsed_time,
                             test_info=None):
    with open(config['info_path'] + config['label'] + '_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['task', 'classification'])
        writer.writerow(['label', config['label']])
        # Parametri modello
        model_params = ['batch_size', 'd_model', 'dim_feedforward', 'dropout', 'num_heads', 'num_layers', 'pe_type']
        for param in model_params:
            writer.writerow([param, config[param]])
        writer.writerow(['num. epochs', num_epochs])
        writer.writerow(['elapsed time', elapsed_time])
        writer.writerow(['time per epoch', elapsed_time / num_epochs])
        # Loss training e validation
        writer.writerow(epoch_info.keys())
        for i in range(num_epochs):
            values = []
            for _, vettore in epoch_info.items():
                values.append(vettore[i])
            writer.writerow(values)
        if test_info is not None:
            writer.writerow(test_info.keys())
            writer.writerow(test_info.values())
        else: 
            writer.writerow(['early stopped'])
