"""Conversation-level EDA for the matched human and GPT dialogues.

Run: python scripts/achraf/eda_statistics.py
Reads data/processed/achraf/{human_matched,gpt_matched}.csv.
Writes results/achraf/eda/eda_counts.csv.

Human utterances are verified from the original annotations; GPT line counts
remain a proxy until generated speaker/turn boundaries are inspected. Sentence
counts are simple punctuation-based estimates rather than a linguistic parser.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data/processed/achraf'

def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]

def count_sentences(text: str) -> int:
    total = 0
    for line in nonempty_lines(text):
        parts = re.split('(?<=[.!?])\\s+', line)
        total += sum((bool(part.strip()) for part in parts))
    return total

def describe_source(source: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'conv_id', 'context', 'dialogue_clean'}
    if source == 'Human':
        required.add('n_utterances')
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns in {path.name}: {sorted(missing)}')
    if df.empty or df[['conv_id', 'context', 'dialogue_clean']].isna().any().any() or df['conv_id'].duplicated().any():
        raise ValueError(f'Invalid or duplicate dialogue records in {path}')
    texts = df['dialogue_clean']
    if texts.str.strip().eq('').any():
        raise ValueError(f'Empty dialogue text in {path}')
    word_counts = texts.str.split().str.len()
    sentence_counts = texts.map(count_sentences)
    line_counts = texts.map(lambda text: len(nonempty_lines(text)))
    if source == 'Human':
        utterance_counts = df['n_utterances']
        mismatches = int((utterance_counts != line_counts).sum())
        print(f'Human utterance/line mismatches: {mismatches}')
        if mismatches:
            raise ValueError('Some human utterance counts do not match the dialogue lines')
    else:
        utterance_counts = line_counts
    stats = pd.DataFrame({'pair_id': df['conv_id'], 'source': source, 'context': df['context'], 'word_count': word_counts, 'sentence_count_estimate': sentence_counts, 'utterance_count_or_line_proxy': utterance_counts})
    print(f"\n{'=' * 35}\n{source}\n{'=' * 35}")
    for column in ('word_count', 'sentence_count_estimate', 'utterance_count_or_line_proxy'):
        print(f'\n{column}:\n{stats[column].describe()}')
    return stats

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-file', type=Path, default=None)
    args = parser.parse_args()
    result = pd.concat([describe_source('Human', args.data_dir / 'human_matched.csv'), describe_source('GPT', args.data_dir / 'gpt_matched.csv')], ignore_index=True)
    source_ids = result.groupby('source')['pair_id'].agg(set)
    if source_ids['Human'] != source_ids['GPT']:
        raise ValueError('Human and GPT EDA IDs differ')
    output_file = args.output_file or PROJECT_ROOT / 'results/achraf/eda/eda_counts.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f'\nSaved: {output_file.resolve()}')
if __name__ == '__main__':
    main()
