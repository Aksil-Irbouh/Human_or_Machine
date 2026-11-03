import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# GPU
# ==========================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU disponible : OUI")
    print("GPU utilise :", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU disponible : NON")
    print("Execution sur CPU")

print("Device :", device)

# ==========================================
# PARAMS
# ==========================================

df = pd.read_csv("../data/processed/paired_human_llm_dialogues.csv")

N_WORDS = 60
MAX_LENGTH = 192
BATCH_SIZE = 32 if device.type == "cuda" else 8
MODEL_NAME = "../models/distilbert-base-uncased"

rng = np.random.default_rng(42)

# ==========================================
# CHARGEMENT LOCAL DISTILBERT
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
)

bert = AutoModel.from_pretrained(
    MODEL_NAME,
    local_files_only=True
).to(device)

bert.eval()

print("DistilBERT charge depuis :", MODEL_NAME)

# ==========================================
# PREPARATION DES DONNEES
# ==========================================

# garder les paires de conv (humain et llm)
# avec au moins N_WORDS mots de chaque cote
df = df.dropna(
    subset=["prompt", "human_dialogue", "llm_dialogue"]
)

df = df[
    (df["human_dialogue"].str.split().str.len() >= N_WORDS) &
    (df["llm_dialogue"].str.split().str.len() >= N_WORDS)
].reset_index(drop=True)

print("Nombre de paires :", len(df))

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

# ==========================================
# SPLIT TRAIN/TEST PAR PROMPT : 80/20
# ==========================================

split = GroupShuffleSplit(
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    split.split(texts, y, groups)
)

print("Train :", len(train_idx))
print("Test :", len(test_idx))

# ==========================================
# EXTRACTION DES EMBEDDINGS CLS
# ==========================================

@torch.inference_mode()
def extract_embeddings(texts):
    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        outputs = bert(**inputs)

        # embedding CLS : batch x 768
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(cls_embeddings.cpu().numpy())

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(
                f"Embeddings : "
                f"{min(i + BATCH_SIZE, len(texts))}/{len(texts)}"
            )

    return np.vstack(embeddings)

# extraction des features
X = extract_embeddings(texts)

print("Shape embeddings :", X.shape)

# ==========================================
# REGRESSION LOGISTIQUE
# ==========================================

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X[train_idx], y[train_idx])

# ==========================================
# EVALUATION
# ==========================================

pred = model.predict(X[test_idx])

print("\nAccuracy :", accuracy_score(y[test_idx], pred))

print(classification_report(
    y[test_idx],
    pred,
    target_names=["Humain", "LLM"]
))

print("Matrice de confusion :")
print(confusion_matrix(y[test_idx], pred))