import re
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# params
df = pd.read_csv("../data/processed/paired_human_llm_dialogues.csv")

N_WORDS = 60
rng = np.random.default_rng(42)

# garder les paires de conv (humain et llm) avec au moins N_WORDS mots de chaque cote
df = df.dropna(subset=["prompt", "human_dialogue", "llm_dialogue"])
df = df[
    (df["human_dialogue"].str.split().str.len() >= N_WORDS) &
    (df["llm_dialogue"].str.split().str.len() >= N_WORDS)
].reset_index(drop=True)

# garder N_WORDS mots consecutifs a une position random
def sample_text(text):
    words = text.split()
    start = rng.integers(0, len(words) - N_WORDS + 1)
    return " ".join(words[start:start + N_WORDS])

# build le dataset
texts = df["human_dialogue"].tolist() + df["llm_dialogue"].tolist()
texts = [sample_text(t) for t in texts]

y = np.array([0] * len(df) + [1] * len(df))
groups = df["prompt"].tolist() * 2

# split train/test par prompt: 80/20
split = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(texts, y, groups))

# features stylometriques
symbols = [".", ",", "!", "?", ";", ":", "'", '"', "-", "—", "(", ")"]

feature_names = [f"Ponctuation '{s}'" for s in symbols] + [
    "Longueur moyenne des mots",
    "Ecart-type longueur des mots",
    "Mediane longueur des mots",
    "Proportion de mots courts",
    "Proportion de mots moyens",
    "Proportion de mots longs",
    "Proportion de mots tres longs",
    "Diversite lexicale",
    "Proportion de mots uniques",
    "Longueur moyenne des segments",
    "Ecart-type longueur des segments",
    "Fins de phrase",
    "Points de suspension",
    "Ponctuation repetee (non '.')",
    "Lettres repetees",
    "Proportion de majuscules",
    "Proportion de chiffres",
    "Densite de ponctuation",
    "Diversite de ponctuation"
]

def extract_features(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())
    lengths = np.array([len(w.replace("'", "")) for w in words])
    counts = Counter(words)

    n = max(len(words), 1)
    n_chars = max(len(text), 1)
    n_letters = max(sum(c.isalpha() for c in text), 1)

    segments = re.split(r"[.!?]+", text.lower())
    segment_lengths = [
        len(re.findall(r"[a-z]+(?:'[a-z]+)*", s))
        for s in segments
        if re.search(r"[a-z]", s)
    ]

    features = [text.count(s) * 100 / n for s in symbols]

    features += [
        np.mean(lengths) if len(lengths) else 0,
        np.std(lengths) if len(lengths) else 0,
        np.median(lengths) if len(lengths) else 0,
        np.mean(lengths <= 3) if len(lengths) else 0,
        np.mean((lengths >= 4) & (lengths <= 6)) if len(lengths) else 0,
        np.mean((lengths >= 7) & (lengths <= 9)) if len(lengths) else 0,
        np.mean(lengths >= 10) if len(lengths) else 0,
        len(counts) / n,
        sum(c == 1 for c in counts.values()) / n,
        np.mean(segment_lengths) if segment_lengths else 0,
        np.std(segment_lengths) if segment_lengths else 0,
        len(re.findall(r"[.!?]+", text)) * 100 / n,
        len(re.findall(r"\.{3,}", text)) * 100 / n,
        len(re.findall(r"([!?,])\1+", text)) * 100 / n,
        len(re.findall(r"([a-z])\1{2,}", text.lower())) * 100 / n,
        sum(c.isupper() for c in text) / n_letters,
        sum(c.isdigit() for c in text) / n_chars,
        sum(text.count(s) for s in symbols) / n_chars,
        sum(s in text for s in symbols)
    ]

    return np.array(features)

X = np.array([extract_features(t) for t in texts])

# regression logistique
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X[train_idx], y[train_idx])

# evaluation
pred = model.predict(X[test_idx])

print("\nAccuracy :", accuracy_score(y[test_idx], pred))
print(classification_report(y[test_idx], pred, target_names=["Humain", "LLM"]))
print("Matrice de confusion :")
print(confusion_matrix(y[test_idx], pred))

# impact des features (positif = LLM, negatif = humain)
coef = model.named_steps["logisticregression"].coef_[0]

print("\nCoefficients:")
for name, c in sorted(zip(feature_names, coef), key=lambda x: x[1], reverse=True):
    classe = "LLM" if c > 0 else "Humain" if c < 0 else "Neutre"
    print(f"{name:<35} : {c:+.3f} -> {classe}")