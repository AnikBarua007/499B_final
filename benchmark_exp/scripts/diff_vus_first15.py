import pandas as pd
new = pd.read_csv('benchmark_exp/eval/metrics/mymodel_first15_metrics.csv')
base = pd.read_csv('benchmark_exp/eval/metrics/multi/mymodel_200.csv')
# merge on file
merged = pd.merge(new, base[['file','VUS-PR']], on='file', how='left', suffixes=('_new','_base'))
merged['VUS-PR_base'] = merged['VUS-PR_base'].astype(float)
merged['delta'] = merged['VUS-PR_new'] - merged['VUS-PR_base']
print('Rows with missing baseline:', merged['VUS-PR_base'].isna().sum())
print('\nPer-file VUS-PR comparison (sorted by delta ascending):')
print(merged[['file','VUS-PR_new','VUS-PR_base','delta']].sort_values('delta').to_string(index=False))
print('\nAverage VUS-PR (new):', merged['VUS-PR_new'].mean())
print('Average VUS-PR (baseline rows present):', merged['VUS-PR_base'].mean())
