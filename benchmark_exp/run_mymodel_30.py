#!/usr/bin/env python3
"""Run MyModel on 30 random files from TSB-AD-M and evaluate metrics."""

import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(r"D:\Learning\499B\Final_get_results_capstone\499B_final")
sys.path.insert(0, str(REPO_ROOT))

from TSB_AD.models.MyModel import MyModel
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

DATASET_DIR = REPO_ROOT / "Datasets" / "TSB-AD-M"
NUM_FILES = 30
SEED = 42

def main():
    random.seed(SEED)
    all_files = sorted([f.name for f in DATASET_DIR.glob("*.csv")])
    print(f"Total files in TSB-AD-M: {len(all_files)}")

    selected = random.sample(all_files, min(NUM_FILES, len(all_files)))
    print(f"Selected {len(selected)} random files\n")

    results = []
    for i, filename in enumerate(selected, 1):
        print(f"[{i:02d}/{len(selected)}] {filename}")
        try:
            df = pd.read_csv(DATASET_DIR / filename).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df["Label"].astype(int).to_numpy()

            # Extract train index from filename
            try:
                train_index = int(Path(filename).stem.split("_")[-3])
            except Exception:
                train_index = max(1, int(0.2 * len(data)))

            data_train = data[:train_index, :]

            # Compute sliding window for metrics
            if data.shape[1] == 1:
                slidingWindow = find_length_rank(data, rank=1)
            else:
                slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)

            # Fit and score
            t0 = time.time()
            clf = MyModel(verbose=0)
            clf.fit(data_train)
            scores = clf.decision_function(data)
            elapsed = time.time() - t0

            # Evaluate — use percentile threshold (top 5% flagged as anomalies)
            threshold = np.percentile(scores, 95)
            pred = scores > threshold
            metrics = get_metrics(scores, label, slidingWindow=slidingWindow, pred=pred)
            metrics["file"] = filename
            metrics["time"] = round(elapsed, 2)
            results.append(metrics)

            print(f"   Time: {elapsed:.1f}s | VUS-PR: {metrics['VUS-PR']:.4f} | Std-F1: {metrics['Standard-F1']:.4f} | PA-F1: {metrics['PA-F1']:.4f} | Ev-F1: {metrics['Event-based-F1']:.4f} | R-F1: {metrics['R-based-F1']:.4f} | Aff-F: {metrics['Affiliation-F']:.4f}")

            # Clear TF session
            if hasattr(clf, "keras") and clf.keras is not None:
                clf.keras.backend.clear_session()
            del clf

        except Exception as e:
            print(f"   ERROR: {e}")
            traceback.print_exc()
            continue

    # Summary
    if results:
        df_results = pd.DataFrame(results)
        out_path = REPO_ROOT / "benchmark_exp" / "eval" / "metrics" / "multi" / "mymodel_improved_30.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(out_path, index=False)

        print("\n" + "=" * 70)
        print(f"RESULTS ({len(results)} files)")
        print("=" * 70)
        metric_cols = ["AUC-PR", "AUC-ROC", "VUS-PR", "VUS-ROC",
                       "Standard-F1", "PA-F1", "Event-based-F1",
                       "R-based-F1", "Affiliation-F"]
        avgs = df_results[metric_cols].mean()
        for col in metric_cols:
            print(f"  {col:20s}: {avgs[col]:.4f}")
        print(f"\n  Avg time per file: {df_results['time'].mean():.1f}s")
        print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
