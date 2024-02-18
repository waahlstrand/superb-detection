from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from argparse import ArgumentParser
from data.superb import PatientDataset
from pathlib import Path
import lightning as L
import json
from typing import *
from data.types import Patient

def generate_folds(patients_root: Path, errors_path: Path, config_path: Path, n_folds: int = 5, seed: int = 42, filter: Callable[[Patient], bool] = lambda x: True):

    L.seed_everything(seed)

    errors = pd.read_csv(errors_path)
    error_moid   = errors.moid.values

    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Remove patients with errors

    removed = [
        *cfg["removed"],
         *error_moid]

    ds = PatientDataset(
        patients_root=patients_root,
        removed=removed,
    )

    # Filter the dataset
    patient_dirs = [p.root / p.moid for p in ds.patients if filter(p)]

    # Create an initial train-test split
    train_dirs, test_dirs = train_test_split(patient_dirs, test_size=0.15, random_state=seed)

    # Generate the folds
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits  = kf.split(train_dirs)

    # Create the folds as a dataframe
    # with columns: dir, fold, train, val, holdout
    # where holdout has fold == -1
    # folds = pd.DataFrame(columns=["dir", "fold", "train", "val", "holdout"])

    # Add the test set
    rows = []
    for test_dir in test_dirs:
        rows.append({
            "dir": test_dir,
            "fold": -1,
            "train": 0,
            "val": 0,
            "holdout": 1
        })

    # Add the train-val splits
    for fold_idx, (train_dir_idx, val_dir_idx) in enumerate(splits):

        # Add the train set
        for train_dir_idx in train_dir_idx:
            rows.append({
                "dir": patient_dirs[train_dir_idx],
                "fold": fold_idx,
                "train": 1,
                "val": 0,
                "holdout": 0
            })

        # Add the val set
        for val_dir_idx in val_dir_idx:
            rows.append({
                "dir": patient_dirs[val_dir_idx],
                "fold": fold_idx,
                "train": 0,
                "val": 1,
                "holdout": 0
            })

    # Save the folds
    folds = pd.DataFrame(rows)
    folds.to_csv("folds.csv")

    # Print train and val sizes for each fold
    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx}")
        print(f"Train: {len(folds[(folds.fold == fold_idx) & (folds.train == 1)])}")
        print(f"Val: {len(folds[(folds.fold == fold_idx) & (folds.val == 1)])}")
        print()

    return folds



def main():

    parser = ArgumentParser()

    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--errors", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--n_folds", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    fold = generate_folds(
        patients_root=Path(args.source),
        errors_path=Path(args.errors),
        config_path=Path(args.cfg),
        n_folds=args.n_folds,
        seed=args.seed,
    )

    

if __name__ == "__main__":

    main()