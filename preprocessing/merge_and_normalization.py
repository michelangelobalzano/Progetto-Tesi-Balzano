import pandas as pd

dataset = ['data1']# , 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10'
file_names = ['bvp.csv', 'eda.csv', 'hr.csv']
output_directory = 'processed_data\\'

bvp, eda, hr = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()



######################################################
# Merge
for data in dataset:
    print(f'{data} processing...')
    directory = f'data\\{data}\\'

    bvp_df = pd.read_csv(f'{directory}bvp.csv')
    eda_df = pd.read_csv(f'{directory}eda.csv')
    hr_df = pd.read_csv(f'{directory}hr.csv')

    bvp = pd.concat([bvp, bvp_df], axis=0, ignore_index=True)
    eda = pd.concat([eda, eda_df], axis=0, ignore_index=True)
    hr = pd.concat([hr, hr_df], axis=0, ignore_index=True)



######################################################
# Normalization
bvp_mean = bvp['bvp'].mean()
eda_mean = eda['eda'].mean()
hr_mean = hr['hr'].mean()

bvp_std = bvp['bvp'].std()
eda_std = eda['eda'].std()
hr_std = hr['hr'].std()

bvp['bvp'] = (bvp['bvp'] - bvp_mean) / bvp_std
eda['eda'] = (eda['eda'] - eda_mean) / eda_std
hr['hr'] = (hr['hr'] - hr_mean) / hr_std

# Esportazione
bvp.to_csv(f'{output_directory}bvp.csv', index=False)
eda.to_csv(f'{output_directory}eda.csv', index=False)
hr.to_csv(f'{output_directory}hr.csv', index=False)