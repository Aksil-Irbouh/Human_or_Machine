from pathlib import Path
import sys
import re
import pandas as pd

# UTILS
# sys.stdout.reconfigure(encoding="utf-8")
# pd.set_option('display.max_columns', None)
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.width", None)

def regexor(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"_comma_", ",", text, flags=re.IGNORECASE) # "_comma_" -> ","
    text = text.lower() # lowercase
    text = re.sub(r"^\s*(?:speaker|listener|assistant\d*|user)\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE) # remove lines containing only a role
    text = re.sub(r"^\s*[\"']?[a-z][a-z0-9_' -]{0,24}:\s+", "", text, flags=re.IGNORECASE | re.MULTILINE) # remove dialogue labels at the beginning of each utterance
    text = re.sub(r'^\s*["“](.*?)["”]\s*$', r"\1", text, flags=re.MULTILINE) # remove quotation marks surrounding an entire utterance
    text = re.sub(r"\s+([,.!?;:])", r"\1", text) # remove spaces before punctuation
    text = re.sub(r"\s+", " ", text) # normalize all whitespace, including \n
    return text.strip()

def corrupt_trim(text):
    if not isinstance(text, str):
        return True
    patterns = [
        r"hit:\d+_conv:\d+", # detect "conv_id"
        r"\d\|\d\|\d+_\d\|\d\|\d", # detect "selfeval" (ex: 5|5|5_5|5|5)
    ]
    return any(re.search(p, text) for p in patterns)

# INPUT
BASE_DIR = Path(__file__).resolve().parent
df_hum = pd.read_csv(BASE_DIR / ".." / "data" / "raw" / "empathetic_dialogues" / "train.csv", on_bad_lines="skip")

mask = df_hum["utterance"].apply(corrupt_trim)
corrupted_conv_ids = df_hum.loc[mask, "conv_id"].unique()
df_hum = df_hum[~df_hum["conv_id"].isin(corrupted_conv_ids)]

df_hum = (
    df_hum
    .sort_values(["conv_id", "utterance_idx"])
    .groupby("conv_id")["utterance"]
    .apply(lambda x: "\n".join(x.astype(str)))
    .reset_index(drop=True)
    .to_frame(name="dialogue")
)

df_llm = pd.read_csv(BASE_DIR / ".." / "data" / "raw" / "gpt_empathetic_dialogues" / "2GPTEmpathicDialoguesDataset.csv", on_bad_lines="skip")

df_llm = df_llm[["processed"]]
df_llm.columns = ["dialogue"]

# REGEX
df_hum["dialogue"] = df_hum["dialogue"].apply(regexor)
df_llm["dialogue"] = df_llm["dialogue"].apply(regexor)

# CSV
print("===== HUM INFO =====\n")
df_hum.info()
print("\n===== HUM HEAD =====\n")
print(df_hum.head(5))
print("\n===== LLM INFO =====\n")
df_llm.info()
print("\n===== LLM HEAD =====\n")
print(df_llm.head(5))
# df_hum.to_csv("data/cleaned/hum_dialogues.csv", index=False)
# df_llm.to_csv("data/cleaned/llm_dialogues.csv", index=False)