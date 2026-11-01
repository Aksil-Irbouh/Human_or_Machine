"""Reconstruct the 19,533 descriptive pairs, preserving case and line breaks.
Use QUOTE_NONE for human CSVs and historical minimal_v1 label cleaning.
GPT lines are a proxy outside the reviewed sample.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import pandas as pd
SPEAKER_PATTERN = '(?im)^[ \\t]*(?:(?:assistant[12]|speaker[12]?|listener[12]?)\\s*:\\s*)+'
STANDALONE_PATTERN = '(?im)^[ \\t]*(?:assistant[12]|speaker[12]?|listener[12]?)\\s*$'
REMAINING_LABEL_PATTERN = '(?im)(?:^[ \\t]*(?:assistant[12]|speaker[12]?|listener[12]?)[ \\t]*:|^[ \\t]*(?:assistant[12]|speaker[12]?|listener[12]?)[ \\t]*$)'
ARTIFACT_PATTERNS = {'Comma placeholders': '_comma_', 'Literal newline sequences': '\\\\n', 'Multiple spaces': ' {2,}', 'Multiple newlines': '\\n[ \\t]*\\n', 'Tabs': '\\t'}

def clean_speaker_labels(text: str) -> str:
    """Supprime les étiquettes GPT sans effacer les séparations de répliques."""
    text = re.sub(SPEAKER_PATTERN, '\n', text)
    text = re.sub(STANDALONE_PATTERN, '', text)
    text = re.sub('\\n\\s*\\n+', '\n', text)
    return text.strip()

def integrity_report(train: pd.DataFrame, human: pd.DataFrame, gpt: pd.DataFrame) -> dict:
    """Vérifie la structure sans assimiler chaque saut de ligne GPT à un tour."""
    indices = human['utterance_indices']
    speaker_lists = human['speaker_ids']
    reconstructed_raw = train.groupby('conv_id', sort=False)['utterance_raw'].agg('\n'.join)

    def expected_indices(values):
        return values == list(range(1, len(values) + 1))

    def consecutive_same_speaker(values):
        return any((left == right for left, right in zip(values, values[1:])))
    checks = {'human_nonconsecutive_indices': int((~indices.map(expected_indices)).sum()), 'human_index_count_mismatches': int((human['n_utterances'] != indices.map(len)).sum()), 'human_speaker_count_mismatches': int((human['n_utterances'] != speaker_lists.map(len)).sum()), 'human_reconstructed_raw_mismatches': int((~human['dialogue_raw'].eq(human['conv_id'].map(reconstructed_raw))).sum()), 'human_repeated_consecutive_speaker': int(speaker_lists.map(consecutive_same_speaker).sum()), 'human_more_than_two_speakers': int(speaker_lists.map(lambda x: len(set(x)) > 2).sum()), 'human_empty_clean_dialogues': int(human['dialogue_clean'].str.strip().eq('').sum()), 'gpt_empty_clean_dialogues': int(gpt['dialogue_clean'].str.strip().eq('').sum()), 'gpt_remaining_targeted_labels': int(gpt['dialogue_clean'].str.contains(REMAINING_LABEL_PATTERN, regex=True, na=False).sum())}
    return checks

def run(human_file: Path, gpt_file: Path, output_dir: Path) -> None:
    train = pd.read_csv(human_file, quoting=csv.QUOTE_NONE)
    gpt = pd.read_csv(gpt_file)
    required_human = {'conv_id', 'utterance_idx', 'utterance', 'prompt', 'context', 'speaker_idx'}
    required_gpt = {'conv_id', 'gptgen', 'prompt', 'context'}
    if (missing := required_human.difference(train.columns)):
        raise ValueError(f'Missing human columns: {sorted(missing)}')
    if (missing := required_gpt.difference(gpt.columns)):
        raise ValueError(f'Missing GPT columns: {sorted(missing)}')
    metadata_variations = train.groupby('conv_id')[['prompt', 'context']].nunique(dropna=False).gt(1).sum()
    if metadata_variations.any():
        raise ValueError(f'Inconsistent within-conversation metadata: {metadata_variations.to_dict()}')
    if train.empty or train[['conv_id', 'utterance_idx', 'speaker_idx', 'utterance', 'prompt', 'context']].isna().any().any():
        raise ValueError('Missing human IDs, speaker IDs, indices or utterance text')
    if gpt.empty or gpt[['conv_id', 'gptgen', 'prompt', 'context']].isna().any().any():
        raise ValueError('Missing GPT conversation IDs or text')
    if train.duplicated(['conv_id', 'utterance_idx']).any() or gpt['conv_id'].duplicated().any():
        raise ValueError('Duplicate human utterance keys or GPT conversation IDs')
    train['split'] = 'train'
    train['prompt_raw'] = train['prompt']
    train['utterance_raw'] = train['utterance']
    train['prompt'] = train['prompt_raw'].str.replace('_comma_', ',', regex=False)
    train['utterance_clean'] = train['utterance_raw'].str.replace('_comma_', ',', regex=False).str.replace(' {2,}', ' ', regex=True)
    train = train.sort_values(['split', 'conv_id', 'utterance_idx'], kind='stable')
    human_conversations = train.groupby(['split', 'conv_id'], as_index=False).agg(context=('context', 'first'), prompt=('prompt', 'first'), prompt_raw=('prompt_raw', 'first'), dialogue_raw=('utterance_raw', '\n'.join), dialogue_clean=('utterance_clean', '\n'.join), utterance_indices=('utterance_idx', list), speaker_ids=('speaker_idx', list), n_utterances=('utterance', 'size'))
    common_ids = set(human_conversations['conv_id']) & set(gpt['conv_id'])
    human_matched = human_conversations[human_conversations['conv_id'].isin(common_ids)].sort_values('conv_id').reset_index(drop=True).copy()
    gpt_matched = gpt[gpt['conv_id'].isin(common_ids)].sort_values('conv_id').reset_index(drop=True).copy()
    if not common_ids:
        raise ValueError('No matched conversation IDs')
    gpt_matched['dialogue_raw'] = gpt_matched['gptgen']
    gpt_matched['prompt_raw'] = gpt_matched['prompt']
    gpt_matched['dialogue_clean'] = gpt_matched['dialogue_raw'].apply(clean_speaker_labels).str.replace('_comma_', ',', regex=False)
    datasets = {'Human': human_matched['dialogue_clean'], 'GPT': gpt_matched['dialogue_clean']}
    audit = pd.DataFrame({name: {artifact: series.str.contains(regex, regex=True, na=False).sum() for artifact, regex in ARTIFACT_PATTERNS.items()} for name, series in datasets.items()})
    if human_matched['conv_id'].duplicated().any() or gpt_matched['conv_id'].duplicated().any():
        raise ValueError('Duplicated conversation IDs in matched data')
    if not human_matched['conv_id'].equals(gpt_matched['conv_id']):
        raise ValueError('Human and GPT conversation IDs are not aligned')
    comparison = human_matched[['conv_id', 'prompt', 'context']].merge(gpt_matched[['conv_id', 'prompt', 'context']], on='conv_id', suffixes=('_human', '_gpt'), validate='one_to_one')
    prompt_matches = comparison['prompt_human'].eq(comparison['prompt_gpt'])
    normalized_prompt_matches = comparison['prompt_human'].eq(comparison['prompt_gpt'].str.replace('_comma_', ',', regex=False))
    context_matches = comparison['context_human'].eq(comparison['context_gpt'])
    checks = integrity_report(train, human_matched, gpt_matched)
    print(f'Human source: {len(train):,} utterances / {len(human_conversations):,} conversations')
    print(f'GPT source: {len(gpt):,} rows / {gpt.conv_id.nunique():,} IDs')
    print(f'Matched: {len(human_matched):,} human / {len(gpt_matched):,} GPT')
    print(f'Prompts matching exactly: {prompt_matches.sum():,}')
    print(f'Prompts matching after comma normalization: {normalized_prompt_matches.sum():,}')
    print(f'Emotional contexts matching: {context_matches.sum():,}')
    print('Integrity audit:', checks)
    print('\nFormatting audit (number of conversations):\n', audit.to_string())
    if not normalized_prompt_matches.all() or not context_matches.all():
        raise ValueError('Paired prompts or emotions differ after comma normalization')
    critical = [name for name in checks if name in {'human_nonconsecutive_indices', 'human_index_count_mismatches', 'human_speaker_count_mismatches', 'human_reconstructed_raw_mismatches', 'human_empty_clean_dialogues', 'gpt_empty_clean_dialogues', 'gpt_remaining_targeted_labels'}]
    if any((checks[name] for name in critical)):
        raise ValueError('Integrity checks failed; no outputs exported. See report above.')
    paired_inputs = pd.concat([human_matched[['conv_id', 'context', 'prompt', 'dialogue_clean']].assign(source='human', generator='human'), gpt_matched[['conv_id', 'context', 'prompt', 'dialogue_clean']].assign(source='gpt', generator='original_gpt_dataset')], ignore_index=True).rename(columns={'conv_id': 'pair_id', 'dialogue_clean': 'text'})
    paired_inputs['cleaning_profile'] = 'minimal_v1'
    output_dir.mkdir(parents=True, exist_ok=True)

    def serialize_lists(frame):
        copy = frame.copy()
        for column in ('utterance_indices', 'speaker_ids'):
            copy[column] = copy[column].map(json.dumps)
        return copy
    serialize_lists(human_matched).to_csv(output_dir / 'human_matched.csv', index=False)
    gpt_export = gpt_matched.drop(columns=['gptgen', 'processed', 'Unnamed: 0'], errors='ignore')
    gpt_export.to_csv(output_dir / 'gpt_matched.csv', index=False)
    serialize_lists(human_conversations).to_csv(output_dir / 'human_conversations.csv', index=False)
    paired_inputs.to_csv(output_dir / 'model_inputs.csv', index=False)
    audit.to_csv(output_dir / 'formatting_audit.csv')
    pd.Series(checks, name='affected_conversations').rename_axis('check').to_csv(output_dir / 'integrity_audit.csv')
    for name, expected_rows in [('human_matched.csv', len(human_matched)), ('gpt_matched.csv', len(gpt_matched)), ('model_inputs.csv', len(paired_inputs))]:
        actual_rows = len(pd.read_csv(output_dir / name, usecols=[0]))
        if actual_rows != expected_rows:
            raise RuntimeError(f'Export incomplete: {name} has {actual_rows} rows, expected {expected_rows}')
    print(f'\nFiles saved in: {output_dir.resolve()}')

def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description='Reproduce the minimally cleaned 19,533-pair analysis dataset.')
    parser.add_argument('--human-file', type=Path, default=root / 'data/raw/empathetic_dialogues/train.csv')
    parser.add_argument('--gpt-file', type=Path, default=root / 'data/raw/gpt_empathetic_dialogues/2GPTEmpathicDialoguesDataset.csv')
    parser.add_argument('--output-dir', type=Path, default=root / 'data/processed/achraf')
    args = parser.parse_args()
    run(args.human_file, args.gpt_file, args.output_dir)
if __name__ == '__main__':
    main()
