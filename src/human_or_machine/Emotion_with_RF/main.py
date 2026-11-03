"""Dialogue Classification Humain / Machine from sentiment analysis scored by SEANCE.

Usage :
    python main.py                                        # config.py values by default
    python main.py mon_fichier.csv                        # select other results from SEANCE
    python main.py mon_fichier.csv --n-human 17780        # specify the last human dialogue index
    python main.py mon_fichier.csv --no-show              # save the .png file without display
"""

import argparse

import config
from load_data import load_dataset
from plots import plot_results
from train_rf import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest Classification Human / Machine (Sentiment Analysis from SEANCE)"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=config.SEANCE_CSV,
        help="path to SEANCE results (.csv) " "(défaut : %(default)s)",
    )
    parser.add_argument(
        "--n-human",
        type=int,
        default=config.N_HUMAN,
        help="last human dialogue index in dataset" "(défaut : %(default)s)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save figure in .png file without displaying it ",
    )
    args = parser.parse_args()

    X, y = load_dataset(args.csv, args.n_human)
    result = train_and_evaluate(
        X,
        y,
        test_size=config.TEST_SIZE,
        cv_folds=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
    )
    plot_results(
        result["cm"],
        result["importances"],
        top=config.TOP_FEATURES,
        out_path=config.OUT_FIG,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
