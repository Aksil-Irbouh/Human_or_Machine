from pathlib import Path
import json
import random
import argparse

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "paired_human_llm_dialogues.csv"
)
MODEL_DIR = PROJECT_ROOT / "models" / "human_or_machine"
REPORT_DIR = PROJECT_ROOT / "reports" / "human_or_machine"

SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = {"human": 0, "llm": 1}
LABEL_NAMES = ["human", "llm"]

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Load and prepare the dataset
# ============================================================

def load_dataset(data_path):
    """
    Convert each paired record into two labeled examples:
        human_dialogue -> 0
        llm_dialogue   -> 1

    conv_id is retained to prevent data leakage between splits.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {data_path}\n"
            "Vérifiez le chemin et le nom du fichier."
        )

    df = pd.read_csv(data_path)

    required_columns = {
        "conv_id",
        "human_dialogue",
        "llm_dialogue",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le dataset : {sorted(missing)}"
        )

    # Keep only the columns required for this task.
    df = df[["conv_id", "human_dialogue", "llm_dialogue"]].copy()

    # Remove records without a conversation identifier.
    df = df.dropna(subset=["conv_id"])

    # Exclude missing and empty dialogue texts.
    for column in ["human_dialogue", "llm_dialogue"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[
        (df["human_dialogue"] != "")
        & (df["llm_dialogue"] != "")
    ].copy()

    if df.empty:
        raise ValueError("Aucune paire de dialogues exploitable.")

    # One row per dialogue, with a binary label.
    human_df = pd.DataFrame({
        "conv_id": df["conv_id"].astype(str),
        "text": df["human_dialogue"],
        "label": LABELS["human"],
    })

    llm_df = pd.DataFrame({
        "conv_id": df["conv_id"].astype(str),
        "text": df["llm_dialogue"],
        "label": LABELS["llm"],
    })

    examples = pd.concat(
        [human_df, llm_df],
        ignore_index=True,
    )

    examples = examples.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    print(f"Nombre de paires : {len(df)}")
    print(f"Nombre total de dialogues : {len(examples)}")
    print("\nRépartition des classes :")
    print(examples["label"].map({0: "human", 1: "llm"}).value_counts())

    return examples


# ============================================================
# 4. Split without conversation leakage
# ============================================================

def split_dataset(examples):
    """
    Split into approximately:
        70% training
        15% validation
        15% test

    All dialogues originating from the same conv_id stay in
    the same subset.
    """
    groups = examples["conv_id"].to_numpy()
    indices = np.arange(len(examples))

    # First reserve 15% for testing.
    split_test = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )

    train_val_idx, test_idx = next(
        split_test.split(indices, groups=groups)
    )

    # Reserve approximately 15% of the total for validation.
    # 0.15 / 0.85 = 0.17647 of the remaining data.
    split_val = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,
        random_state=SEED,
    )

    train_rel_idx, val_rel_idx = next(
        split_val.split(
            train_val_idx,
            groups=groups[train_val_idx],
        )
    )

    train_idx = train_val_idx[train_rel_idx]
    val_idx = train_val_idx[val_rel_idx]

    train_df = examples.iloc[train_idx].reset_index(drop=True)
    val_df = examples.iloc[val_idx].reset_index(drop=True)
    test_df = examples.iloc[test_idx].reset_index(drop=True)

    # Verify that no conversation is shared across subsets.
    train_groups = set(train_df["conv_id"])
    val_groups = set(val_df["conv_id"])
    test_groups = set(test_df["conv_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    for name, subset in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(subset)} dialogues, "
            f"{subset['conv_id'].nunique()} conversations"
        )

        if subset["label"].nunique() < 2:
            raise ValueError(
                f"Le sous-ensemble {name} ne contient pas les deux classes. "
                "Utilisez davantage de conversations ou ajustez le split."
            )

    return train_df, val_df, test_df


# ============================================================
# 5. Sentence-BERT embeddings
# ============================================================

def encode_dialogues(encoder, dialogues, batch_size=BATCH_SIZE):
    """
    Encode texts into normalized Sentence-BERT embeddings.
    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    """
    embeddings = encoder.encode(
        dialogues.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


# ============================================================
# 6. Neural network classifier
# ============================================================

class DialogueClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# 7. Training and validation
# ============================================================

def train_classifier(
    model,
    train_embeddings,
    train_labels,
    val_embeddings,
    val_labels,
    device,
):
    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_x = torch.tensor(
        val_embeddings, dtype=torch.float32
    ).to(device)

    val_y = torch.tensor(
        val_labels, dtype=torch.long
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(train_dataset)

        model.eval()

        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()

            val_predictions = val_logits.argmax(dim=1).cpu().numpy()

        val_f1 = f1_score(
            val_labels,
            val_predictions,
            average="macro",
            zero_division=0,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val macro-F1: {val_f1:.4f}"
        )

        # Early stopping based on validation macro-F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping déclenché.")
            break

    if best_state is None:
        raise RuntimeError("Aucun état du modèle n'a été sauvegardé.")

    model.load_state_dict(best_state)
    model.to(device)

    return model, pd.DataFrame(history)


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate_classifier(model, embeddings, labels, device):
    model.eval()

    x = torch.tensor(
        embeddings, dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    predictions = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    return metrics, predictions, probabilities


# ============================================================
# 9. Save reports and artifacts
# ============================================================

def save_artifacts(
    model,
    encoder,
    history,
    test_df,
    metrics,
    predictions,
    probabilities,
    device,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the Sentence-BERT encoder and classifier weights.
    encoder.save(str(MODEL_DIR / "sentence_bert"))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 384,
            "hidden_dim": 256,
            "dropout": 0.3,
            "labels": LABELS,
        },
        MODEL_DIR / "classifier.pt",
    )

    joblib.dump(
        {
            "sentence_model_name": SENTENCE_MODEL_NAME,
            "labels": LABELS,
            "max_length": encoder.max_seq_length,
        },
        MODEL_DIR / "config.joblib",
    )

    # Save training history.
    history.to_csv(
        REPORT_DIR / "training_history.csv",
        index=False,
    )

    # Save final evaluation metrics.
    with open(
        REPORT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    # Save predictions and class probabilities.
    results = test_df.copy()

    results["true_label"] = results["label"].map({
        0: "human",
        1: "llm",
    })

    results["predicted_label"] = [
        LABEL_NAMES[pred] for pred in predictions
    ]

    results["prob_human"] = probabilities[:, 0]
    results["prob_llm"] = probabilities[:, 1]

    results.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    # Confusion matrix.
    cm = np.array(metrics["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion - Test")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    # Training curves.
    if not history.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(
            history["epoch"], history["train_loss"],
            label="Train loss",
        )
        axes[0].plot(
            history["epoch"], history["val_loss"],
            label="Validation loss",
        )
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(
            history["epoch"],
            history["val_macro_f1"],
            label="Validation macro-F1",
        )
        axes[1].set_title("Validation macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            REPORT_DIR / "training_curves.png",
            dpi=150,
        )
        plt.close()


# ============================================================
# 10. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classification de dialogues humains ou générés "
            "par un LLM avec Sentence-BERT."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Chemin vers le dataset CSV.",
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice utilisé : {device}")
    print(f"Dataset : {args.data}")

    # Load and split data.
    examples = load_dataset(args.data)
    train_df, val_df, test_df = split_dataset(examples)

    # Load pretrained Sentence-BERT.
    print(f"\nChargement de {SENTENCE_MODEL_NAME}...")
    encoder = SentenceTransformer(SENTENCE_MODEL_NAME)

    # Generate embeddings for each split.
    print("\nEncodage des dialogues d'entraînement...")
    train_embeddings = encode_dialogues(
        encoder, train_df["text"]
    )

    print("\nEncodage des dialogues de validation...")
    val_embeddings = encode_dialogues(
        encoder, val_df["text"]
    )

    print("\nEncodage des dialogues de test...")
    test_embeddings = encode_dialogues(
        encoder, test_df["text"]
    )

    train_labels = train_df["label"].to_numpy(dtype=np.int64)
    val_labels = val_df["label"].to_numpy(dtype=np.int64)
    test_labels = test_df["label"].to_numpy(dtype=np.int64)

    # Initialize the classifier.
    model = DialogueClassifier(
        input_dim=train_embeddings.shape[1]
    ).to(device)

    # Train with early stopping.
    print("\nDébut de l'entraînement...")
    model, history = train_classifier(
        model,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        device,
    )

    # Evaluate once on the held-out test set.
    print("\nÉvaluation finale sur le test set...")
    metrics, predictions, probabilities = evaluate_classifier(
        model,
        test_embeddings,
        test_labels,
        device,
    )

    print(f"\nTest accuracy : {metrics['accuracy']:.4f}")
    print(f"Test macro-F1 : {metrics['macro_f1']:.4f}")

    print("\nRapport de classification :")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    # Persist model and reports.
    save_artifacts(
        model,
        encoder,
        history,
        test_df,
        metrics,
        predictions,
        probabilities,
        device,
    )

    print("\nEntraînement terminé.")
    print(f"Modèle sauvegardé dans : {MODEL_DIR}")
    print(f"Rapports sauvegardés dans : {REPORT_DIR}")


if __name__ == "__main__":
    main()
