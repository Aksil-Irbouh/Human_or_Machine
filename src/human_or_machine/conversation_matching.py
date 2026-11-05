"""Match EmpatheticDialogues training conversations with 2GPTEmpathicDialogues.

Install: pip install pandas
Run:     python conversation_matching.py
Optional: python conversation_matching.py --data-dir path/to/csvs --output-dir processed

Required input files (in --data-dir): train.csv, 2GPTEmpathicDialoguesDataset.csv
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd


SPEAKER_PATTERN = r'(?im)^[ \t]*(?:(?:assistant[12]|speaker[12]?|listener[12]?)\s*:\s*)+'
STANDALONE_PATTERN = r'(?im)^[ \t]*(?:assistant[12]|speaker[12]?|listener[12]?)\s*$'
REMAINING_LABEL_PATTERN = (
    r'(?im)(?:^[ \t]*(?:assistant[12]|speaker[12]?|listener[12]?)[ \t]*:'
    r'|^[ \t]*(?:assistant[12]|speaker[12]?|listener[12]?)[ \t]*$)'
)
ARTIFACT_PATTERNS = {
    'Comma placeholders': r'_comma_',
    'Literal newline sequences': r'\\n',
    'Multiple spaces': r' {2,}',
    'Multiple newlines': r'\n[ \t]*\n',
    'Tabs': r'\t',
}


def clean_speaker_labels(text: str) -> str:
    """Replicate the cleaned notebook's GPT speaker-label removal."""
    text = re.sub(SPEAKER_PATTERN, '\n', text)
    text = re.sub(STANDALONE_PATTERN, '', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()


def run(data_dir: Path, output_dir: Path) -> None:
    # 1. Load: treat quotes in human training CSV as literal text.
    train = pd.read_csv(data_dir / 'train.csv', quoting=csv.QUOTE_NONE)
    gpt = pd.read_csv(data_dir / '2GPTEmpathicDialoguesDataset.csv')
    required_human = {'conv_id', 'utterance_idx', 'utterance', 'prompt', 'context'}
    required_gpt = {'conv_id', 'gptgen', 'prompt', 'context'}
    if missing := required_human.difference(train.columns):
        raise ValueError(f'Missing human columns: {sorted(missing)}')
    if missing := required_gpt.difference(gpt.columns):
        raise ValueError(f'Missing GPT columns: {sorted(missing)}')

    metadata_variations = train.groupby('conv_id')[['prompt', 'context']].nunique(dropna=False).gt(1).sum()
    if metadata_variations.any():
        print('WARNING: Some human conversations vary in prompt/context:', metadata_variations.to_dict())

    train['split'] = 'train'
    for col in ('prompt', 'utterance'):
        train[col] = train[col].str.replace('_comma_', ',', regex=False)

    train = train.sort_values(['split', 'conv_id', 'utterance_idx'])
    human_conversations = (
        train.groupby(['split', 'conv_id'], as_index=False)
        .agg(context=('context', 'first'),
             prompt=('prompt', 'first'),
             dialogue=('utterance', '\n'.join),
             n_turns=('utterance', 'size'))
    )

    # 2. Align by ID, preserving originals and sorting identically.
    common_ids = set(human_conversations['conv_id']) & set(gpt['conv_id'])
    human_matched = (human_conversations[human_conversations['conv_id'].isin(common_ids)]
                     .sort_values('conv_id').reset_index(drop=True).copy())
    gpt_matched = (gpt[gpt['conv_id'].isin(common_ids)]
                   .sort_values('conv_id').reset_index(drop=True).copy())

    # 3. Text cleaning matches the notebook (raw columns remain available).
    gpt_matched['dialogue_clean'] = gpt_matched['gptgen'].apply(clean_speaker_labels)
    human_matched['dialogue_clean'] = human_matched['dialogue'].str.replace(r' {2,}', ' ', regex=True)
    gpt_matched['dialogue_final'] = gpt_matched['dialogue_clean'].str.replace('_comma_', ',', regex=True)

    # 4. Audit before last-stage text normalization (same as notebook).
    datasets = {'Human': human_matched['dialogue'], 'GPT': gpt_matched['dialogue_clean']}
    audit = pd.DataFrame({
        name: {artifact: series.str.contains(regex, regex=True, na=False).sum()
               for artifact, regex in ARTIFACT_PATTERNS.items()}
        for name, series in datasets.items()
    })

    # 5. Pair validation; fail rather than silently export unaligned pairs.
    assert not human_matched['conv_id'].duplicated().any(), 'Duplicate human conversation IDs'
    assert not gpt_matched['conv_id'].duplicated().any(), 'Duplicate GPT conversation IDs'
    assert human_matched['conv_id'].equals(gpt_matched['conv_id']), 'Conversation IDs are not aligned'

    comparison = human_matched[['conv_id', 'prompt', 'context']].merge(
        gpt_matched[['conv_id', 'prompt', 'context']],
        on='conv_id', suffixes=('_human', '_gpt'), validate='one_to_one'
    )
    prompt_matches = comparison['prompt_human'].eq(comparison['prompt_gpt'])
    normalized_prompt_matches = comparison['prompt_human'].eq(
        comparison['prompt_gpt'].str.replace('_comma_', ',', regex=False)
    )
    context_matches = comparison['context_human'].eq(comparison['context_gpt'])
    remaining_labels = gpt_matched['dialogue_clean'].str.contains(
        REMAINING_LABEL_PATTERN, regex=True, na=False
    ).sum()

    print(f'Human source: {len(train):,} utterances / {len(human_conversations):,} conversations')
    print(f'GPT source: {len(gpt):,} rows / {gpt.conv_id.nunique():,} IDs')
    print(f'Matched: {len(human_matched):,} human / {len(gpt_matched):,} GPT')
    print(f'Prompts matching exactly: {prompt_matches.sum():,}')
    print(f'Prompts matching after comma normalization: {normalized_prompt_matches.sum():,}')
    print(f'Emotional contexts matching: {context_matches.sum():,}')
    print(f'GPT conversations with remaining targeted speaker labels: {remaining_labels:,}')
    print('\nFormatting audit (number of conversations):\n', audit.to_string())

    if not normalized_prompt_matches.all() or not context_matches.all():
        print('\nWARNING: Some paired metadata differ. Inspect before comparative analysis.')

    # 6. Output. Create output folder if it does not yet exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    human_matched.to_csv(output_dir / 'human_matched.csv', index=False)
    gpt_matched.to_csv(output_dir / 'gpt_matched.csv', index=False)
    human_conversations.to_csv(output_dir / 'human_conversations.csv', index=False)
    audit.to_csv(output_dir / 'formatting_audit.csv')
    print(f'\nFiles saved in: {output_dir.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent,
                        help='Folder containing train.csv and 2GPTEmpathicDialoguesDataset.csv')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output folder (default: <data-dir>/processed)')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir or args.data_dir / 'processed')


if __name__ == '__main__':
    main()
