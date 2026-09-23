import sys
import pandas as pd

# UTILS
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

def spacer():
    print("\n==========\n")

# INPUT
df_hum_train = pd.read_csv("data/raw/empathetic_dialogues/train.csv", on_bad_lines="skip")
df_hum_valid = pd.read_csv("data/raw/empathetic_dialogues/valid.csv", on_bad_lines="skip")
df_hum_test = pd.read_csv("data/raw/empathetic_dialogues/test.csv", on_bad_lines="skip")
df_llm = pd.read_csv("data/raw/gpt_empathetic_dialogues/2GPTEmpathicDialoguesDataset.csv", on_bad_lines="skip")

print("===== HUM =====")

print("\n=== Train ===\n")
print(df_hum_train.head(10))
spacer()
print(df_hum_train.info())
spacer()
print(df_hum_train.describe(include="all"))

print("\n=== Test ===\n")
print(df_hum_test.head(10))
spacer()
print(df_hum_test.info())
spacer()
print(df_hum_test.describe(include="all"))

print("\n=== Valid ===\n")
print(df_hum_valid.head(10))
spacer()
print(df_hum_valid.info())
spacer()
print(df_hum_valid.describe(include="all"))

print("\n===== LLM =====\n")
print(df_llm.head(10))
spacer()
df_llm.info()
spacer()
print(df_llm.describe(include="all"))