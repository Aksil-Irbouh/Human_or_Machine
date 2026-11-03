import numpy as np
import pandas as pd
import spacy

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# params
df = pd.read_csv("../data/processed/paired_human_llm_dialogues.csv")

N_WORDS = 60
rng = np.random.default_rng(42)

nlp = spacy.load("en_core_web_sm", disable=["ner"])

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

# features linguistiques spacy
pos_tags = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV",
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART"
]

dep_tags = [
    "nsubj", "dobj", "pobj", "amod", "advmod",
    "aux", "conj", "cc", "ROOT"
]

feature_names = [
    f"POS {p}" for p in pos_tags
] + [
    f"DEP {d}" for d in dep_tags
]

def extract_features(doc):
    words = [t for t in doc if t.is_alpha]
    n = max(len(words), 1)

    features = [
        sum(t.pos_ == p for t in words) / n
        for p in pos_tags
    ]

    features += [
        sum(t.dep_ == d for t in words) / n
        for d in dep_tags
    ]

    return np.array(features)

X = np.array([
    extract_features(doc)
    for doc in nlp.pipe(texts, batch_size=128)
])

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