#!/usr/bin/env python3
import csv
from pathlib import Path
p=Path('benchmark_exp/eval/metrics/multi/mymodel_200.csv')
if not p.exists():
    print('MISSING',p)
    raise SystemExit(1)
with p.open() as f:
    r=csv.reader(f)
    header=next(r)
    cols=header[1:]
    sums=[0.0]*len(cols)
    cnt=0
    for row in r:
        if not row or len(row)<2:
            continue
        vals=row[1:1+len(cols)]
        try:
            nums=[float(x) for x in vals]
        except Exception:
            continue
        for i,v in enumerate(nums):
            sums[i]+=v
        cnt+=1
    if cnt==0:
        print('no rows')
        raise SystemExit(1)
    means=[s/cnt for s in sums]
    out_p=Path('benchmark_exp/eval/metrics/multi/mymodel_200_summary.csv')
    out_p.parent.mkdir(parents=True,exist_ok=True)
    with out_p.open('w',newline='') as wf:
        w=csv.writer(wf)
        w.writerow(['metric','mean','nrows'])
        for name,m in zip(cols,means):
            w.writerow([name,repr(m),cnt])
    print('Saved summary to',out_p)
    print('nrows',cnt)
    for name,m in zip(cols,means):
        print(f'{name}: {m}')
