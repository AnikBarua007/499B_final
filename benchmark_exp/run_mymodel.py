#!/usr/bin/env python3
"""Run MyModel across a dataset file list and save per-file anomaly scores."""

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSB_AD.models.MyModel import MyModel


def _resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_on_list(dataset_dir, file_list_csv, out_dir, overwrite=False, model_kwargs=None):
    dataset_dir = _resolve_path(dataset_dir)
    file_list_csv = _resolve_path(file_list_csv)
    out_dir = _resolve_path(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    file_list = pd.read_csv(file_list_csv)["file_name"].values

    for filename in file_list:
        clf = None
        df = None
        data = None
        data_train = None
        scores = None
        out_path = out_dir / f"{Path(filename).stem}.npy"
        if out_path.exists() and not overwrite:
            print("Skipping (exists):", filename)
            continue

        print("Processing:", filename)
        try:
            df = pd.read_csv(dataset_dir / filename).dropna()
        except Exception as exc:
            print(f"Failed reading {filename}: {exc}")
            continue

        data = df.iloc[:, 0:-1].values.astype(float)

        try:
            train_index = int(Path(filename).stem.split("_")[-3])
        except Exception:
            train_index = max(1, int(0.2 * len(data)))

        data_train = data[:train_index, :]

        clf = MyModel(**(model_kwargs or {}))
        t0 = time.time()
        try:
            clf.fit(data_train)
            scores = clf.decision_function(data)

            elapsed = time.time() - t0
            print(f"Time: {elapsed:.2f}s | saving -> {out_path}")
            np.save(out_path, scores)
        except Exception as exc:
            print(f"Error processing {filename}: {exc}")
            traceback.print_exc()
            continue
        finally:
            # TensorFlow/Keras can retain graph state across iterations unless we clear it.
            if clf is not None and getattr(clf, "keras", None) is not None:
                clf.keras.backend.clear_session()
            del clf, df, data, data_train, scores
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="Datasets/TSB-AD-M")
    parser.add_argument("--file_list", default="Datasets/File_List/TSB-AD-M-Eva.csv")
    parser.add_argument("--out_dir", default="benchmark_exp/eval/score/mymodel")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--predict_batch_size", type=int, default=32)
    parser.add_argument("--ae_epochs", type=int, default=20)
    args = parser.parse_args()

    model_kw = dict(
        window_size=100,
        stride=5,
        batch_size=args.batch_size,
        predict_batch_size=args.predict_batch_size,
        ae_epochs=args.ae_epochs,
        lr=1e-4,
        dropout=0.1,
        mask_ratio=0.25,
        shuffle_buffer_size=1024,
        prefetch_buffer=1,
    )

    run_on_list(
        args.dataset_dir,
        args.file_list,
        args.out_dir,
        overwrite=args.overwrite,
        model_kwargs=model_kw,
    )
