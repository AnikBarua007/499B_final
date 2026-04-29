#!/usr/bin/env python3
"""Evaluate saved anomaly-score files against TSB-AD labels."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _evaluate_one(filename: str, score_dir: str, dataset_dir: str) -> dict:
    score_path = Path(score_dir) / f"{Path(filename).stem}.npy"
    data_path = Path(dataset_dir) / filename

    df = pd.read_csv(data_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    labels = df["Label"].astype(int).to_numpy()
    scores = np.load(score_path)

    if len(scores) != len(labels):
        raise ValueError(
            f"Length mismatch for {filename}: score={len(scores)} label={len(labels)}"
        )

    sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    metrics = get_metrics(scores, labels, slidingWindow=sliding_window)
    return {"file": filename, **metrics}


def evaluate_scores(
    score_dir: Path,
    dataset_dir: Path,
    file_list_csv: Path,
    output_path: Path | None = None,
    resume: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    file_list = pd.read_csv(file_list_csv)["file_name"].tolist()
    rows = []
    processed = set()

    if resume and output_path is not None and output_path.exists():
        existing_df = pd.read_csv(output_path)
        if not existing_df.empty and "file" in existing_df.columns:
            rows = existing_df.to_dict("records")
            processed = set(existing_df["file"].tolist())
            print(f"Resuming from {len(processed)} existing rows in {output_path}")

    pending = []
    for filename in file_list:
        score_path = score_dir / f"{Path(filename).stem}.npy"
        if score_path.exists() and filename not in processed:
            pending.append(filename)

    if workers <= 1:
        for filename in pending:
            rows.append(_evaluate_one(filename, str(score_dir), str(dataset_dir)))
            processed.add(filename)
            print(f"Evaluated {filename}")
            if output_path is not None:
                checkpoint_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
                checkpoint_df.to_csv(output_path, index=False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_evaluate_one, filename, str(score_dir), str(dataset_dir)): filename
                for filename in pending
            }
            for future in as_completed(future_map):
                filename = future_map[future]
                rows.append(future.result())
                processed.add(filename)
                print(f"Evaluated {filename}")
                if output_path is not None:
                    checkpoint_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
                    checkpoint_df.to_csv(output_path, index=False)

    result_df = pd.DataFrame(rows)
    if not result_df.empty and "file" in result_df.columns:
        result_df = result_df.sort_values("file").reset_index(drop=True)
    return result_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--file_list", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    score_dir = _resolve_path(args.score_dir)
    dataset_dir = _resolve_path(args.dataset_dir)
    file_list_csv = _resolve_path(args.file_list)
    output_path = _resolve_path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = evaluate_scores(
        score_dir,
        dataset_dir,
        file_list_csv,
        output_path=output_path,
        resume=not args.no_resume,
        workers=max(1, args.workers),
    )
    result_df.to_csv(output_path, index=False)

    print(f"Saved metrics to: {output_path}")
    if not result_df.empty:
        print("Average metrics:")
        print(result_df.drop(columns=["file"]).mean(numeric_only=True).to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
