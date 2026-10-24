"""Pipeline Parameters"""

# path to SEANCE results. By default from SEANCE/data/processed/
SEANCE_CSV = "results_human_and_machine_allwords.csv"

# Number of human conversations
N_HUMAN = 17_780

# Ratio training / testing
TEST_SIZE = 0.20  # 80 % train / 20 % test

# number of k-folds
CV_FOLDS = 5

# Seed
RANDOM_STATE = 42

# number of features to print
TOP_FEATURES = 20

# figures save files path
OUT_FIG = "confusion_matrice_et_importances.png"
