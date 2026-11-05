# Day 2 on the shared 17,780-pair dataset

This rerun uses the exact `data/processed/paired_human_llm_dialogues.csv` supplied in the dev and Emotions_with_RF snapshots. Both copies have the same SHA-256. All 17,780 pairs (35,560 dialogues, 32 emotions) remain; no text is cleaned again and no length threshold is applied to the descriptive population.

## Run from the repository root

```bash
python scripts/achraf/prepare_shared_dialogues.py
python scripts/achraf/stylometric_analysis.py --input-file data/processed/achraf/model_inputs_shared_17780.csv --output-dir results/achraf/day2_shared_17780
```

Then open `notebooks/achraf/linguistic_features_shared_17780.ipynb` and run all cells. Its default `RECOMPUTE_SYNTAX = True` performs fresh spaCy inference. Install the analysis requirements and `en_core_web_sm` if necessary, as described in the main README. To reproduce figures from the supplied new syntax cache, set the flag to `False` after running the two commands above. Input and syntax-cache fingerprints are checked in cache mode.

The adapter checks the audited source fingerprint, allowing LF/CRLF conversion, checks complete pair records and verifies that exported text is identical after CSV round-trip. A different source file requires inspection before changing its expected fingerprint or reusing the audit counts.

The generated model input is `data/processed/achraf/model_inputs_shared_17780.csv`. All new counts, features, figures and metadata go to `results/achraf/day2_shared_17780`. The original Day 2 notebook/results and all Day 3 files remain unchanged. The adapter creates EDA word counts and punctuation-based sentence estimates; it does not invent utterance counts from flattened text. Do not run the original EDA script on this new input.

## Preprocessing and limitations

The shared export is lowercased and whitespace is collapsed. GPT text originates from the raw dataset's `processed` column. In contrast, the historical 19,533-pair analysis preserved case and line breaks and used minimally cleaned `gptgen`. Differences between runs therefore reflect both population and preprocessing changes. In particular, uppercase frequency is uninformative here, and removing line boundaries can change the valid bigram sequence used for repetition. The tokenizer and feature formulas themselves are unchanged.

The audit found 31 retained human dialogues missing 50 utterances. Another 18 final human texts differ only in double quotation marks. All remain as supplied, by agreement. The corresponding limitation applies to the shared dataset; these runs do not repair it. Removing the 31 pairs later would leave 17,749 pairs and require recalculating the results.

One numeric-only human dialogue has undefined word length, contraction frequency and bigram repetition: those paired summaries use 17,779 pairs. The historical question-mark denominator convention is retained and that comparison uses all 17,780 pairs. Both repetition window conditions use the same 15,546 eligible pairs, ten windows per dialogue, seed 42. Full raw repetition uses a different population; do not interpret differences as a controlled causal estimate. This diagnostic is not classifier transformation robustness.

The separately reviewed 320-pair alignment sample remains drawn from the original 19,533-pair corpus. Its provenance checks still depend on the original `model_inputs.csv`, not the new shared input. Leave that original input and the reviewed CSV untouched.

## Installation of this add-on

Copy the `scripts`, `notebooks`, and `results` folders into their corresponding repository folders. Copy this note into `docs/achraf`. The shared input CSV already belongs at `data/processed/paired_human_llm_dialogues.csv`; it is not duplicated in this add-on. The two reused helper scripts are identical to the corrected lean contribution. Do not overwrite later local edits to those helpers without reviewing them.

The ZIP contains newly named files and unchanged helper copies; it does not replace the existing 19,533-pair results or Day 3. Review before staging. No commit or push is performed automatically.

## Completed run: 2 November 2026

Fresh inference completed with spaCy 3.8.18 and en_core_web_sm 3.8.0, Python 3.12.14. Every notebook code cell completed; all 35,560 feature keys matched and figures were visually checked. The shared texts matched exactly after conversion and CSV round-trip. No original contribution file was changed.

| Measure | Human | GPT | Paired dz | Pairs |
|---|---:|---:|---:|---:|
| Longueur moyenne des mots | 3.840039 | 4.496318 | 1.924455 | 17,779 |
| Contraction frequency | 2.595347 | 4.723533 | 0.706319 | 17,779 |
| Ponctuation '?' | 1.853364 | 1.133869 | -0.346720 | 17,780 |
| Bigram repetition rate | 2.552211 | 15.134704 | 2.111199 | 17,779 |
| avg_tree_depth | 4.061339 | 5.037404 | 0.929071 | 17,780 |

Bigram repetition in this table is expressed as a percentage. On the same 15,546 eligible pairs, pairwise-minimum windows give 2.670335% human / 2.753498% GPT; fixed 30-bigram windows give 1.499850% / 0.991981%.

Update the report population and preprocessing description, descriptive values and dz, repetition values and eligibility count. Keep the 320-pair alignment section unchanged. Do not reuse the old tree-depth or repetition figures for this rerun.
