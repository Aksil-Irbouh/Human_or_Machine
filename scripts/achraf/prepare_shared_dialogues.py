"""Adapt the shared paired CSV for Day 2 without changing text or excluding pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from eda_statistics import count_sentences

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', type=Path, default=ROOT / 'data/processed/paired_human_llm_dialogues.csv')
    parser.add_argument('--output-file', type=Path, default=ROOT / 'data/processed/achraf/model_inputs_shared_17780.csv')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/achraf/day2_shared_17780')
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.input_file.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if source_hash != '563eb5f5a3803e71a748303f9e18cc47733f2b2fe0f7fcd34990ee87cd6694b9':
        raise ValueError('Shared export differs from the audited version; inspect it before reusing the audit counts')
    pairs = pd.read_csv(args.input_file)
    required = ['conv_id', 'context', 'prompt', 'human_dialogue', 'llm_dialogue']
    if not set(required).issubset(pairs.columns):
        raise ValueError(f'Missing columns: {sorted(set(required) - set(pairs.columns))}')
    if len(pairs) != 17780 or pairs['conv_id'].duplicated().any():
        raise ValueError('Expected 17,780 unique matched pairs; inspect any changed dataset first')
    if pairs[required].isna().any().any() or pairs[required].apply(lambda col: col.str.strip().eq('')).any().any():
        raise ValueError('Missing or empty identifiers, metadata or text')
    frames = []
    for source, column in [('human', 'human_dialogue'), ('gpt', 'llm_dialogue')]:
        frame = pairs[['conv_id', 'context', 'prompt', column]].rename(columns={'conv_id': 'pair_id', column: 'text'}).copy()
        frame['source'] = source
        frame['cleaning_profile'] = 'shared_export_17780_unchanged'
        frames.append(frame)
    model_inputs = pd.concat(frames, ignore_index=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    model_inputs.to_csv(args.output_file, index=False)
    reloaded = pd.read_csv(args.output_file)
    for source, column in [('human', 'human_dialogue'), ('gpt', 'llm_dialogue')]:
        restored = reloaded.loc[reloaded['source'].eq(source)].set_index('pair_id')['text'].reindex(pairs['conv_id']).reset_index(drop=True)
        if not restored.equals(pairs[column].reset_index(drop=True)):
            raise ValueError(f'Export changed {source} text')
    counts = model_inputs[['pair_id', 'source', 'context']].copy()
    counts['word_count'] = model_inputs['text'].str.split().str.len()
    counts['sentence_count_estimate'] = model_inputs['text'].map(count_sentences)
    counts.to_csv(args.results_dir / 'eda_counts.csv', index=False)
    metadata = {'source_file': args.input_file.name, 'source_sha256': hashlib.sha256(args.input_file.read_bytes()).hexdigest(), 'model_inputs_sha256': hashlib.sha256(args.output_file.read_bytes()).hexdigest(), 'n_pairs': len(pairs), 'n_dialogues': len(model_inputs), 'n_emotions': int(pairs.context.nunique()), 'text_preserved_exactly': True, 'excluded_pairs': 0, 'known_incomplete_human_pairs_retained': 31, 'known_missing_human_utterances': 50, 'preprocessing': 'Shared export: lowercase and collapsed whitespace; no further cleaning.', 'utterance_counts': 'Unavailable from flattened text; not inferred from line counts.'}
    (args.results_dir / 'shared_input_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Adapted {len(pairs):,} pairs; all exported texts match their source exactly.')
    print(counts.groupby('source')['word_count'].agg(['mean', 'median', 'min', 'max']).round(3))
    print('Retained the 31 known incomplete human dialogues. No utterance counts inferred.')


if __name__ == '__main__':
    main()
