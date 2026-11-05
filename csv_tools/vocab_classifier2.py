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

N_WORDS = 80
MIN_FREQ = 5  # occurrences pour 10 000 mots
MIN_RATIO = 2
MIN_LENGTH = 1
rng = np.random.default_rng(42)

WORD_REGEX = fr"[a-z]{{{MIN_LENGTH},}}" # garder en features que les mots >= MIN_LENGTH

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

def tokenize(text):
    text = text.lower().replace("'", "").replace("’", "")
    return re.findall(r"[a-z]+|[^\w\s]", text)

# build le dataset
texts = df["human_dialogue"].tolist() + df["llm_dialogue"].tolist()
texts = [sample_text(t) for t in texts]

y = np.array([0] * len(df) + [1] * len(df))
groups = df["prompt"].tolist() * 2

# split train/test par prompt: 80/20
split = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(split.split(texts, y, groups))

# creation du dictionnaire uniquement sur le train
llm = Counter(
    w for i in train_idx if y[i] == 1
    for w in tokenize(texts[i])
)
human = Counter(
    w for i in train_idx if y[i] == 0
    for w in tokenize(texts[i])
)

n_llm = sum(llm.values())
n_human = sum(human.values())

vocab = {
    w for w in set(llm) | set(human)
    if re.fullmatch(WORD_REGEX, w) or re.fullmatch(r"[^\w\s]", w)
}

# ratio de frequences
ratio = {
    w: ((llm[w] + 0.5) / n_llm) /
       ((human[w] + 0.5) / n_human)
    for w in vocab
}

# selection des mots discriminants
llm_words = sorted(
    [w for w in vocab if llm[w] / n_llm * 10000 >= MIN_FREQ and ratio[w] >= MIN_RATIO],
    key=lambda w: ratio[w],
    reverse=True
)

human_words = sorted(
    [w for w in vocab if human[w] / n_human * 10000 >= MIN_FREQ and ratio[w] <= 1 / MIN_RATIO],
    key=lambda w: ratio[w]
)

words = llm_words + human_words

print(f"{'Mot':<20} {'Ratio':>8} {'Classe':>10}")

for w in llm_words:
    print(f"{w:<20} {ratio[w]:>8.2f} {'LLM':>10}")

for w in human_words:
    print(f"{w:<20} {1 / ratio[w]:>8.2f} {'Human':>10}")

if not words:
    raise ValueError("Aucun mot sélectionné : réduire les seuils.")

# extract features et normalisation /100 mots
def extract_features(text):
    tokens = tokenize(text)
    counts = Counter(tokens)
    return np.array([counts[w] * 100 / len(tokens) for w in words])

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
for w, c in sorted(zip(words, coef), key=lambda x: x[1], reverse=True):
    classe = "LLM" if c > 0 else "Humain" if c < 0 else "Neutre"
    print(f"{w:<20} : {c:+.3f} -> {classe}")