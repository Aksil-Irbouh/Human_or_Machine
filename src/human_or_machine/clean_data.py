from pathlib import Path
import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / ".." / "data" / "raw" # input
OUTPUT_DIR = BASE_DIR / ".." / ".." / "data" / "processed" # output


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r"_comma_", ",", text, flags=re.I) # changer les "_comma_" en "," et "re.I" ignore la casse
    text = text.lower() # mettre en minuscule

    # remove dialogue labels
    text = re.sub(
        r'^\s*["\']?(?:speaker\d*|listener\d*|assistant\d*|user|chatgpt\d*)\s*:\s*',
        "",
        text,
        flags=re.I | re.M
    )

    # remove spaces before punctuation + normalize whitespace
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def count_gpt_speakers(text): # compte le nombre de speakers (pour etre sur qu'il n'y en a que 2)
    speakers = re.findall(
        r"(?mi)^\s*assistant([12])\s*:",
        str(text)
    )
    return len(set(speakers))

# load data
df_hum = pd.read_csv(
    DATA_DIR / "empathetic_dialogues" / "train.csv",
    on_bad_lines="skip"
)

df_llm = pd.read_csv(
    DATA_DIR / "gpt_empathetic_dialogues" / "2GPTEmpathicDialoguesDataset.csv",
    on_bad_lines="skip"
)


# drop les dialogues avec >2 speakers, car ils y a qq dialogues avec plus que 2
hum_speakers = (
    df_hum.groupby("conv_id")["speaker_idx"]
    .nunique()
)

hum_ids = set(
    hum_speakers[hum_speakers == 2].index
)

df_llm["n_speakers"] = (
    df_llm["gptgen"]
    .apply(count_gpt_speakers)
)

llm_ids = set(
    df_llm.loc[
        df_llm["n_speakers"] == 2,
        "conv_id"
    ]
)

paired_ids = hum_ids & llm_ids

df_hum = df_hum[
    df_hum["conv_id"].isin(paired_ids)
].copy()

df_llm = df_llm[
    df_llm["conv_id"].isin(paired_ids)
].copy()


# mettre les dialogues humains en one line
df_hum["utterance"] = (
    df_hum["utterance"]
    .apply(clean_text)
)

human_dialogues = (
    df_hum
    .sort_values(["conv_id", "utterance_idx"])
    .groupby("conv_id")
    .agg(
        context=("context", "first"),
        prompt=("prompt", "first"),
        human_dialogue=("utterance", " ".join)
    )
    .reset_index()
)


# mettre les dialogues llm en one line
llm_dialogues = (
    df_llm[
        ["conv_id", "context", "prompt", "processed"]
    ]
    .rename(
        columns={"processed": "llm_dialogue"}
    )
    .copy()
)

llm_dialogues["llm_dialogue"] = (
    llm_dialogues["llm_dialogue"]
    .apply(clean_text)
)


# regrouepr les dialogues hummain et llm par paires (par id)
df_pairs = human_dialogues.merge(
    llm_dialogues,
    on=["conv_id", "context", "prompt"],
    how="inner",
    validate="one_to_one"
)


# Clean prompt too
df_pairs["prompt"] = (
    df_pairs["prompt"]
    .apply(clean_text)
)

# save
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR
    / "paired_human_llm_dialogues.csv"
)

df_pairs.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

print("Paired conversations:", len(df_pairs))
print("Saved:", output_path)