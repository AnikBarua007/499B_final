import pandas as pd
base = pd.read_csv('benchmark_exp/eval/metrics/multi/mymodel_200.csv')
files = pd.read_csv('Datasets/File_List/TSB-AD-M-first15.csv')['file_name'].tolist()
sel = base[base['file'].isin(files)]
print('Found', len(sel), 'matching rows')
print('Baseline average VUS-PR:', float(sel['VUS-PR'].mean()))
print(sel[['file','VUS-PR']].to_string(index=False))
