"""
It is important that your first indices correspond to human
conversation. The last indices must correspond to machine conversation
(see results_human_and_machine_allwords.csv)
"""

import numpy as np
import pandas as pd


def load_dataset(csv_path, n_human):
    """returns:
    - X as features values processed for RF.
    - y as class values. Labels policy : 0 = human, 1 = machine.
    """
    df = pd.read_csv(csv_path, sep=r",\s*", engine="python")
    df.columns = df.columns.str.strip()

    X = df.drop(columns=["filename"])
    X = X.loc[:, ~X.columns.str.startswith("Unnamed")]

    # SEANCE may write Error or NaN. Handle by setting 0
    X = X.apply(pd.to_numeric, errors="coerce")
    n_missing = int(X.isna().sum().sum())
    if n_missing:
        print(f"NaN values replaced by 0  : {n_missing}")
    X = X.fillna(0.0)

    n_total = len(X)

    # Labels Y : 0 = humain, 1 = machine
    y = pd.Series(np.zeros(n_total, dtype=int), name="label")
    y.iloc[n_human:] = 1

    print(f"X : {X.shape[0]} dialogues × {X.shape[1]} features")
    print(f"Labels :\n{y.value_counts().to_string()}")
    return X, y
