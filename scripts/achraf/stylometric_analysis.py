"""Day 2: stylometric features and paired bigram repetition analysis.

Run from the repository: python scripts/achraf/stylometric_analysis.py
Input: data/processed/achraf/model_inputs.csv. Outputs: results/achraf/day2.
Based on Flo's 31 measurements plus repetition and contractions.
Zero-word convention retains the historical question-mark result, not a general per-word definition.
Window scores are fractions; summary rates are percentages. No classifier is trained.
"""
import argparse
from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'data' / 'processed' / 'achraf' / 'model_inputs.csv'
OUTPUT = ROOT / 'results' / 'achraf' / 'day2'
SEED = 42
N_SAMPLES = 10
FIXED_N = 30
symbols = ['.', ',', '!', '?', ';', ':', "'", '"', '-', '—', '(', ')']
feature_names = [f"Ponctuation '{s}'" for s in symbols] + ['Longueur moyenne des mots', 'Ecart-type longueur des mots', 'Mediane longueur des mots', 'Proportion de mots courts', 'Proportion de mots moyens', 'Proportion de mots longs', 'Proportion de mots tres longs', 'Diversite lexicale', 'Proportion de mots uniques', 'Longueur moyenne des segments', 'Ecart-type longueur des segments', 'Fins de phrase', 'Points de suspension', "Ponctuation repetee (non '.')", 'Lettres repetees', 'Proportion de majuscules', 'Proportion de chiffres', 'Densite de ponctuation', 'Diversite de ponctuation']

def extract_features(text):
    text = text.replace('’', "'").replace('“', '"').replace('”', '"')
    words = re.findall("[a-z]+(?:'[a-z]+)*", text.lower())
    lengths = np.array([len(w.replace("'", '')) for w in words])
    counts = Counter(words)
    n = max(len(words), 1)
    n_chars = max(len(text), 1)
    n_letters = max(sum((c.isalpha() for c in text)), 1)
    segments = re.split('[.!?]+', text.lower())
    segment_lengths = [len(re.findall("[a-z]+(?:'[a-z]+)*", s)) for s in segments if re.search('[a-z]', s)]
    features = [text.count(s) * 100 / n for s in symbols]
    features += [np.mean(lengths) if len(lengths) else 0, np.std(lengths) if len(lengths) else 0, np.median(lengths) if len(lengths) else 0, np.mean(lengths <= 3) if len(lengths) else 0, np.mean((lengths >= 4) & (lengths <= 6)) if len(lengths) else 0, np.mean((lengths >= 7) & (lengths <= 9)) if len(lengths) else 0, np.mean(lengths >= 10) if len(lengths) else 0, len(counts) / n, sum((c == 1 for c in counts.values())) / n, np.mean(segment_lengths) if segment_lengths else 0, np.std(segment_lengths) if segment_lengths else 0, len(re.findall('[.!?]+', text)) * 100 / n, len(re.findall('\\.{3,}', text)) * 100 / n, len(re.findall('([!?,])\\1+', text)) * 100 / n, len(re.findall('([a-z])\\1{2,}', text.lower())) * 100 / n, sum((c.isupper() for c in text)) / n_letters, sum((c.isdigit() for c in text)) / n_chars, sum((text.count(s) for s in symbols)) / n_chars, sum((s in text for s in symbols))]
    return np.array(features)

def extract_bigrams(text):
    text = text.lower()
    bigrams = []
    for segment in re.split('[.!?\\n]+', text):
        words = re.findall("[a-z]+(?:'[a-z]+)*", segment)
        bigrams.extend(zip(words[:-1], words[1:]))
    return bigrams

def repetition_rate(bigrams):
    return (len(bigrams) - len(set(bigrams))) / len(bigrams) if bigrams else np.nan

def contraction_frequency(text):
    text = text.replace('’', "'").lower()
    words = re.findall("[a-z]+(?:'[a-z]+)*", text)
    if not words:
        return np.nan
    suffixes = ("n't", "'m", "'re", "'ve", "'ll", "'d")
    s_contractions = {"he's", "she's", "it's", "that's", "there's", "here's", "what's", "who's", "where's", "how's", "let's"}
    n_contractions = sum((word.endswith(suffixes) or word in s_contractions for word in words))
    return n_contractions * 100 / len(words)

def paired_summary(features_df, feature):
    paired = features_df.pivot(index='pair_id', columns='source', values=feature).dropna(subset=['human', 'gpt']).copy()
    paired['difference'] = paired['gpt'] - paired['human']
    summary = {'feature': feature, 'pairs': len(paired), 'human_mean': paired['human'].mean(), 'gpt_mean': paired['gpt'].mean(), 'mean_difference': paired['difference'].mean(), 'median_difference': paired['difference'].median(), 'std_difference': paired['difference'].std(), 'pct_gpt_higher': (paired['difference'] > 0).mean() * 100, 'pct_tied': (paired['difference'] == 0).mean() * 100}
    return (paired, summary)

def sampled_rate(bigrams, n, position):
    start = int(position * (len(bigrams) - n + 1))
    return repetition_rate(bigrams[start:start + n])

def repetition_robustness(paired_texts):
    rng = np.random.default_rng(SEED)
    records = []
    for pair_id, row in paired_texts.iterrows():
        human, gpt = (extract_bigrams(row['human']), extract_bigrams(row['gpt']))
        n_min = min(len(human), len(gpt))
        if n_min == 0:
            continue
        h_pos, g_pos = (rng.random(N_SAMPLES), rng.random(N_SAMPLES))
        sizes = [('Pairwise minimum', n_min)]
        if n_min >= FIXED_N:
            sizes.append((f'Fixed {FIXED_N}', FIXED_N))
        for method, n in sizes:
            h_rates = [sampled_rate(human, n, p) for p in h_pos]
            g_rates = [sampled_rate(gpt, n, p) for p in g_pos]
            for aggregation, func in [('Mean', np.mean), ('Median', np.median)]:
                h, g = (float(func(h_rates)), float(func(g_rates)))
                records.append({'pair_id': pair_id, 'sampling': method, 'aggregation': aggregation, 'matched_bigrams': n, 'human': h, 'gpt': g, 'difference': g - h})
    return pd.DataFrame(records)

def robustness_summary(results, eligible_only=False):
    if eligible_only:
        valid_ids = results.loc[results['sampling'] == f'Fixed {FIXED_N}', 'pair_id']
        results = results[results['pair_id'].isin(valid_ids)]
    summary = results.groupby(['sampling', 'aggregation'], sort=False).agg(n_pairs=('pair_id', 'nunique'), human_rate=('human', 'mean'), gpt_rate=('gpt', 'mean'), mean_difference=('difference', 'mean'), pct_gpt_higher=('difference', lambda x: (x > 0).mean() * 100), pct_tied=('difference', lambda x: (x == 0).mean() * 100)).reset_index()
    for col in ['human_rate', 'gpt_rate', 'mean_difference']:
        summary[col] *= 100
    return summary

def main():
    parser = argparse.ArgumentParser(description='Extract stylometric features and length-controlled repetition.')
    parser.add_argument('--input-file', type=Path, default=INPUT)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir
    df = pd.read_csv(args.input_file)
    required = {'pair_id', 'source', 'text'}
    if (missing := (required - set(df.columns))):
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing identifiers, source labels, or dialogue text')
    if df.duplicated(['pair_id', 'source']).any() or set(df['source']) != {'human', 'gpt'}:
        raise ValueError('Expected exactly one human and one GPT row per pair')
    if not df.groupby('pair_id')['source'].nunique().eq(2).all():
        raise ValueError('Incomplete matched pairs')
    if df.empty or df['text'].str.strip().eq('').any():
        raise ValueError('Empty dataset or dialogue text')
    X = np.array([extract_features(t) for t in df['text']])
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df.insert(0, 'pair_id', df['pair_id'].to_numpy())
    features_df.insert(1, 'source', df['source'].to_numpy())
    features_df['Bigram repetition rate'] = df['text'].map(lambda text: repetition_rate(extract_bigrams(text))).to_numpy()
    features_df['Contraction frequency'] = df['text'].map(contraction_frequency).to_numpy()
    no_words = ~df['text'].str.contains('[A-Za-z]', na=False)
    features_df.loc[no_words, 'Longueur moyenne des mots'] = np.nan
    word_pairs, word_summary = paired_summary(features_df, 'Longueur moyenne des mots')
    repetition_pairs, repetition_summary = paired_summary(features_df, 'Bigram repetition rate')
    _, question_summary = paired_summary(features_df, "Ponctuation '?'")
    _, contraction_summary = paired_summary(features_df, 'Contraction frequency')
    summaries = [word_summary, repetition_summary, question_summary, contraction_summary]
    paired_texts = df.pivot(index='pair_id', columns='source', values='text').loc[repetition_pairs.index]
    robustness = repetition_robustness(paired_texts)
    full_summary = robustness_summary(robustness)
    comparable_summary = robustness_summary(robustness, eligible_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_dir / 'stylometric_features.csv', index=False)
    pd.DataFrame(summaries).to_csv(output_dir / 'paired_feature_summary.csv', index=False)
    full_summary.to_csv(output_dir / 'repetition_robustness_all_pairs.csv', index=False)
    comparable_summary.to_csv(output_dir / 'repetition_robustness_comparable.csv', index=False)
    print(f'Extracted {len(feature_names)} Flo features + bigram repetition + contraction frequency for {len(features_df):,} dialogues')
    for summary in summaries:
        print(f"\n{summary['feature']} ({summary['pairs']:,} valid pairs)")
        print(f"Human mean: {summary['human_mean']:.6f} | GPT mean: {summary['gpt_mean']:.6f}")
        print(f"Mean GPT - Human: {summary['mean_difference']:.6f} | GPT higher: {summary['pct_gpt_higher']:.2f}%")
    print('\nLength-controlled repetition (all eligible pairs; rates in %):')
    print(full_summary.round(2).to_string(index=False))
    print(f'\nFour-way robustness comparison (same pairs with >= {FIXED_N} bigrams):')
    print(comparable_summary.round(2).to_string(index=False))
    print(f'\nSaved results to: {output_dir}')
if __name__ == '__main__':
    main()
