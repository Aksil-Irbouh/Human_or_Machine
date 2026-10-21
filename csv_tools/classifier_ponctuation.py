import re
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# params
df = pd.read_csv("../data/processed/paired_human_llm_dialogues.csv")
N_WORDS = 80
rng = np.random.default_rng(42)

# garder les paires de conv (humain et llm) avec au moins N_WORDS mots de chaque cote
df = df[
    (df["human_dialogue"].str.split().str.len() >= N_WORDS) &
    (df["llm_dialogue"].str.split().str.len() >= N_WORDS)
].reset_index(drop=True)

# garder N_WORDS mots consecutifs a une position random
def sample_text(text):
    words = text.split()
    start = rng.integers(0, len(words) - N_WORDS + 1)
    return " ".join(words[start:start + N_WORDS])

# features
symbols = ["!", "?", ".", ",", ";", ":", "'", '"']

# extract features et normalisation /100 mots
def extract_features(text):
    n = len(text.split())
    counts = [text.count(s) for s in symbols]
    return np.array(counts) * 100 / n

# build le dataset
texts = df["human_dialogue"].tolist() + df["llm_dialogue"].tolist()
X = np.array([extract_features(sample_text(t)) for t in texts])
y = np.array([0] * len(df) + [1] * len(df))  # 0 = humain, 1 = LLM
groups = df["prompt"].tolist() * 2

# split train/test par prompt: 80/20
split = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups))

# regression logistique
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X[train_idx], y[train_idx])

# evaluation
pred = model.predict(X[test_idx])

print("Accuracy:", accuracy_score(y[test_idx], pred))
print(classification_report(y[test_idx], pred, target_names=["Humain", "LLM"]))
print("Matrice de confusion:")
print(confusion_matrix(y[test_idx], pred))

# impact des features (positif = LLM, negatif = humain)
coef = model.named_steps["logisticregression"].coef_[0]
print("\nCoefficients:")
for s, c in sorted(zip(symbols, coef), key=lambda x: x[1], reverse=True):
    classe = "LLM" if c > 0 else "Humain" if c < 0 else "Neutre"
    print(f"{s:>4} : {c:+.3f} -> {classe}")