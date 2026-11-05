# Achraf: linguistic features and conversational alignment

This contribution covers descriptive linguistic features (Day 2) and conversational alignment (Day 3). It does not train a classifier or replace the group's modelling data, split or generalisation pipeline.

## Analysis Overview

This contribution focuses on two complementary aspects of human–LLM dialogue comparison: **linguistic characteristics** and **conversational alignment**.

### 1. Linguistic and Syntactic Comparison

Compares human-written and GPT-generated dialogues using stylometric features, syntactic measures, and repetition analysis.

Two datasets were analysed:

- **Original dataset (19,533 matched pairs):** The initial analysis, with results stored in `results/achraf/day2/`.
- **Shared dataset (17,780 matched pairs):** An additional analysis using the group's common dataset, with results stored in `results/achraf/day2_shared_17780/`.

The corresponding notebooks are `linguistic_features.ipynb` and `linguistic_features_shared_17780.ipynb`.

Both analyses use comparable methods, allowing results to be examined across the two dataset versions. The shared dataset preserves all 17,780 pairs, including 31 pairs with known incomplete human dialogues.

For the shared-dataset workflow and input provenance, see `docs/achraf/SHARED_17780.md`.

### 2. Conversational Alignment

Examines how linguistic similarity evolves between successive contributions in human–human and human–LLM-generated dialogues.

The analysis uses **320 manually reviewed matched pairs**, balanced across 32 emotions, and includes:

- Lexical overlap using Jaccard similarity.
- Semantic similarity using SentenceTransformer embeddings and cosine similarity.
- Conversational progression and variability.
- Paired comparisons, bootstrap confidence intervals, and sensitivity checks.

The corresponding notebook is `conversational_alignment.ipynb`, with results stored in `results/achraf/day3/`.

These measures describe conversational similarity and alignment patterns; higher similarity alone does not establish stronger adaptation or empathy.

### Directory Naming Convention

The names `day2` and `day3` refer to the original development stages of the project, not separate experimental days.

| Directory | Description |
|---|---|
| `results/achraf/day2/` | Linguistic and syntactic analysis — 19,533 matched pairs |
| `results/achraf/day2_shared_17780/` | Linguistic and syntactic analysis — 17,780 shared matched pairs |
| `results/achraf/day3/` | Conversational alignment — 320 reviewed matched pairs |

The original directory names have been retained to preserve compatibility with existing notebooks, scripts, and file references.


## Included code

| File | Purpose |
|---|---|
| `scripts/achraf/prepare_dialogues.py` | Reconstruct the minimally cleaned 19,533 matched pairs from the repository's raw train and GPT CSVs |
| `scripts/achraf/prepare_shared_dialogues.py` | Validate and adapt the group's 17,780-pair dataset without changing dialogue text |
| `scripts/achraf/eda_statistics.py` | Reproduce dialogue word counts and explicitly labelled sentence/line estimates |
| `scripts/achraf/stylometric_analysis.py` | Flo's 31 feature definitions plus contractions, repeated bigrams and paired window controls |
| `notebooks/achraf/linguistic_features.ipynb` | Roland-based dependency extraction, paired effects, feature merge and Day 2 figures |
| `notebooks/achraf/linguistic_features_shared_17780.ipynb` | Reproduce the linguistic and syntactic comparisons on the shared 17,780-pair dataset |
| `notebooks/achraf/conversational_alignment.ipynb` | Lexical and semantic adjacency, progression, variability, bootstrap and first-three checks |


All function calls, signatures, assignments and comprehensions stay on one physical line. Multi-line Python blocks retain their normal indentation. Learning examples and debugging histories remain in the original personal notebooks, outside this contribution.

## Data and execution order

Run terminal commands from the cloned repository root. Keep the provided `data/processed/achraf/day3_reviewed_320_v2.csv` unchanged. The raw inputs already exist in the supplied repository snapshot and are byte-identical to Ash's copies.

```bash
python scripts/achraf/prepare_dialogues.py
python scripts/achraf/eda_statistics.py
python scripts/achraf/stylometric_analysis.py
```

The first script creates `data/processed/achraf/model_inputs.csv`, `human_matched.csv`, `gpt_matched.csv`, `human_conversations.csv` and audits. It uses `csv.QUOTE_NONE` for the human CSV and does not silently skip malformed rows. It retains text case, punctuation and line breaks, replaces `_comma_`, and removes the targeted historical GPT labels. This is minimal_v1 cleaning, not an exhaustive role parser. It leaves the original raw files untouched.

Open the two notebooks after these commands. Each finds the repository root by `pyproject.toml` or `scripts/achraf`. All results go under `results/achraf`. No notebook requires a Colab upload dialog or automatically downloads a ZIP.

### Inference or fast reproduction

The notebook defaults are `RECOMPUTE_SYNTAX = True` and `RECOMPUTE_SEMANTIC = True`. They rerun the models and require the dependencies and model files. For the supplied matching result caches, set the corresponding flag to `False`. Both notebooks then reproduce the comparisons and figures without model inference. Cache mode is explicitly reported in output metadata. Cache mode checks input and score-file fingerprints, normalizing CRLF to LF. Changed files require fresh inference. Metadata retains the original inference record during cached reruns; unavailable historical versions are labelled explicitly.

For Colab, clone/upload the contribution with its inputs and results, or edit the configuration paths explicitly. A GPU accelerates transformer inference but is not required by the lexical or downstream calculations. The spaCy model must be installed separately:

```bash
python -m pip install -r docs/achraf/requirements-analysis.txt
python -m spacy download en_core_web_sm
```

The dependencies file is a separate analysis list, not a tested replacement for the group's pyproject/lockfile. The supplied repository declares Python >=3.13. No new installation of that environment was validated here. The original spaCy run reported 3.8.16; the exact historical dependency set and Hugging Face revision were not captured. New inference runs record available package/model versions. Exact model revisions remain unrecorded.

## Sample sizes and interpretation

- Full descriptive dataset: 19,533 matched pairs, 39,066 dialogues.
- Shared linguistic-analysis dataset: 17,780 matched pairs, 35,560 dialogues. All pairs retained, including 31 with known incomplete human dialogues.
- Reviewed alignment subset: 320 matched pairs, 10 per each of 32 emotions, 640 dialogues. It is curated, not a random sample.
- Repetition window comparison: the same 16,760 eligible pairs in both conditions. This is not the group's approximately 17k modelling subset.
- The group's exact modelling IDs and common evaluation split remain a separate integration decision. The historical 80/10/10 split script is intentionally excluded because the repository README describes a different modelling split.

### Stylometric conventions retained for reproducibility

Word features and contraction frequency normalize curly apostrophes. Bigram repetition intentionally retains the earlier straight-apostrophe tokenizer **without** curly-apostrophe replacement, because this exactly reproduces the exported results. A newer script had inadvertently changed that convention, changing 22 repetition scores and the fixed-window eligibility count. A methodological harmonisation would require a separately labelled rerun of all affected results.

Human dialogue `hit:3005_conv:6011` contains only numeric contributions (`5`, `4`, `5`, `7`). Mean word length, contractions and repetition are undefined and exclude that pair. The historical question-mark table retains its zero value and all 19,533 pairs. The extractor's denominator fallback is a historical export convention, not a general definition of a rate for texts with zero words. This convention is explicitly retained so the reported results reproduce.

Repetition windows use ten sampled contiguous windows with seed 42. Mean/median aggregation happens within each dialogue before group averaging. The in-memory dialogue-level sampled aggregates are fractions; summary CSV rates use percentages. These controls do not complete the assignment's classifier transformation robustness experiment.

### Syntax

Depth counts the root as level 1. Tree traversal excludes punctuation and spaces. Dependency distance uses positions among retained tokens, excluding self-headed roots. Tree-depth SD uses `ddof=0`; paired Cohen's dz uses the sample SD of differences (`ddof=1`). Parse the complete newline-separated dialogue as in the original analysis. These are whole-dialogue syntactic features, not syntactic accommodation. Maxima and depth may depend on dialogue/sentence length and parser accuracy.

### Alignment

Lowercase word-set Jaccard and pretrained `sentence-transformers/all-MiniLM-L6-v2` cosine compare each contribution with its immediate predecessor, excluding the participant labels. English regex tokenization includes function words and preserves the original straight-apostrophe convention. The notebook validates two alternating participants and contribution counts before scoring. The known `mitig ates` correction applies to one reviewed GPT contribution in memory only. The CSV and full descriptive dataset are not rewritten.

Both original source-text columns must match `model_inputs.csv` after whitespace normalization. Prompt comparison also normalizes the source `_comma_` placeholder. Automated provenance checks do not replace the earlier manual topic review.

Bootstrap resamples 320 whole matched pairs with replacement 10,000 times, seed 42. It is conditional uncertainty for this reviewed sample and does not remove length or selection bias. First-three-transition analysis retains all 320 pairs and controls comparison count/absolute turn position, not word length. Relative progression averages within each dialogue/stage before averaging dialogues. Within-dialogue alignment SD uses `ddof=1`.

Higher lexical and semantic similarity does not independently establish causal partner adaptation, style matching or empathy. Both participants in GPT dialogues are generated. Common topic and repetition can raise similarity. The notebook checks token lengths before fresh inference and stops rather than silently truncating.

## Verified results

| Measure | Human | GPT | Mean gap |
|---|---:|---:|---:|
| Adjacent lexical overlap | 0.060179 | 0.155792 | +0.095613 |
| Adjacent semantic similarity | 0.310020 | 0.594026 | +0.284006 |
| First-three lexical overlap | 0.0606 | 0.1309 | +0.0703 |
| First-three semantic similarity | 0.3100 | 0.5601 | +0.2501 |

95% paired bootstrap intervals: lexical [0.0883, 0.1028], semantic [0.2678, 0.3008]. GPT is higher in 261/320 first-three lexical comparisons and 290/320 semantic comparisons.

## Review and verification limits

Preprocessing, EDA and stylometric extraction ran on all supplied records, and their output tables match the originals within 1e-12 numeric tolerance. Both cleaned notebooks passed every code cell sequentially in new namespaces using explicitly selected saved model outputs. All original syntax/alignment result tables reproduce under that mode. Dependency arithmetic and semantic adjacency indexing also passed controlled input checks.

**Subsequent Colab verification (31 October 2026):** Achraf ran both cleaned notebooks with fresh spaCy and SentenceTransformers inference. Both completed and saved outputs; the displayed means, effect sizes, bootstrap intervals, first-three comparisons and progression values matched the reference results at displayed precision. The syntax run reported spaCy 3.8.18 and en_core_web_sm 3.8.0. This is a comparison of displayed summaries, not a byte-for-byte comparison of all fresh output files. The bundled tables retain the original reference outputs; their metadata describes those outputs. Notebook execution counts/outputs remain cleared.

## Attribution

Flo supplied the original 31 stylometric measurements. Roland supplied the dependency-tree starting code. Achraf developed the paired analyses, added repetition/contraction measures and length sensitivity analyses, and developed the reviewed-sample conversational alignment analyses with implementation assistance. Confirm the final authorship wording with the group.

## Git handoff

Copy the bundle's named folders into your cloned repository while on `achraf/linguistic-alignment`. Review `git status` and stage selected paths explicitly. Do not copy your entire NLP folder or replace the group's root configuration files. The bundle's large result tables are derived outputs; agree with the group whether to version all tables or just summary CSVs/figures. Inputs and result caches under `achraf` have distinct names/paths and do not replace the modelling dataset.

## Minimal exports

This contribution contains 28 files. Runtime metadata JSON files are retained; packaging manifests and historical review documents are omitted. The scripts/notebooks no longer export the combined feature table, duplicate repetition summary, dialogue-by-condition repetition aggregates, dialogue-stage scores, variability table, directional agreement, gap correlations or selection manifest. Their calculations, checks and notebook displays remain unchanged. The stylistic comparison CSV is retained because it includes the paired effect sizes used in the slides.

Model caches are `day2_syntactic_features.csv` and `transition_scores.csv`. `dialogue_scores.csv` is a final result regenerated from transitions. Both the style feature table and main summary CSVs/figures are retained. Run preprocessing first to generate model inputs and matched human/GPT files. Those necessary generated inputs and the preprocessing audits are unchanged.
