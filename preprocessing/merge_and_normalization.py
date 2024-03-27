import pandas as pd

# Colonne da normalizzare per ogni segnale
cols_to_normalize = {
    'ACC': ['x', 'y', 'z'],
    'BVP': ['bvp'],
    'EDA': ['eda'],
    'HR': ['hr'],
    'TEMP': ['temp']
}

def normalization(data_directory, df_names, signals, labeled=False):

    ######################################################
    # Merge

    data = {}
    for signal in signals:
        data[signal] = pd.DataFrame()

    for df_name in df_names:
        print(f'{df_name} processing...')
        directory = f'{data_directory}\\{df_name}\\'

        for signal in signals:
            df_temp = pd.read_csv(f'{directory}{signal}.csv')
            data[signal] = pd.concat([data[signal], df_temp], axis=0, ignore_index=True)

    ######################################################
    # Normalization
            
    for signal in signals:
        for col in cols_to_normalize[signal]:
            mean = data[signal][col].mean()
            std = data[signal][col].std()
            data[signal][col] = (data[signal][col] - mean) / std
        if labeled:
            data[signal].to_csv(f'processed_data\\{signal}_LABELED.csv', index=False)
        else:
            data[signal].to_csv(f'processed_data\\{signal}.csv', index=False)