This part was done by an LLM and reviewed afterward 

# SEANCE 1.2.0 Patches — adaptation to modern Python
SEANCE 1.2.0 (kristopherkyle/SEANCE) targets Python ≤ 3.10 and spaCy 2.x.
These modifications make it possible to run it with Python 3.11+ and spaCy 3.x,
without changing the computed indices.


## Provenance

- Repository: https://github.com/kristopherkyle/SEANCE
- Reference commit: `89213d9ab7e397a64db1fde91ef7f44494a19e35` (master, 2020-10-28)
- Patch file: `seance_1_2_0_python_moderne.patch`


## Applying the patch

```bash
git clone https://github.com/kristopherkyle/SEANCE.git
cd SEANCE
git apply chemin/vers/seance_1_2_0_python_moderne.patch
```

## Details
Patch 1 — `'rU'` open mode removed (8 occurrences, single and double
quotes): `'rU'` (explicit universal newlines) was removed in Python 3.11;
the `'r'` mode now does this job by default.
No effect on the results.
Patch 2 — spaCy 2 → spaCy 3 loading (in `main`, ~line 1153):
`set_data_path` no longer exists in recent spaCy 3; loading is now done
via `nlp = spacy.load('en_core_web_sm')` instead of the path to the
repository's bundled model. This block only runs if Nouns / Verbs /
Adjectives or Components are checked in the GUI.

Steps requiring no code changes (to do as well):
rename the repository's `en_core_web_sm/` folder (spaCy 2.0.0 format),
e.g. `en_core_web_sm_v2_backup`: spaCy 3 failed to load it
install a modern model: `python -m spacy download en_core_web_sm`.
Consequence: the POS tagger is more recent than the 2018 one, and the
indices are identical in nature.