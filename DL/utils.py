import csv

# Salvataggio info classificazione
def save_session_info(model_name, config):
    with open(config['info_path'] + config['label'] + '_classification_' + model_name + '.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['label', config['label']])
        writer.writerow(['split type', config['split_type']])
        # Parametri modello
        model_params = ['batch_size', 'd_model', 'dim_feedforward', 'dropout', 'num_heads', 'num_layers', 'pe_type']
        for param in model_params:
            writer.writerow([param, config[param]])
        writer.writerow(['iterazione','accuracy','precision','recall','f1-score'])

# Salvataggio info test i-esimo
def save_test_info(model_name, config, i, acc, prec, rec, f1):
    with open(config['info_path'] + config['label'] + '_classification_' + model_name + '.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i,acc,prec,rec,f1])

# Salvataggio risultato finale
def save_final_info(model_name, config, acc, prec, rec, f1):
    with open(config['info_path'] + config['label'] + '_classification_' + model_name + '.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Finale',acc,prec,rec,f1])