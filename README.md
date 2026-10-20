# call-search

PII-safe semantic search over synthetic call transcripts.

Pipeline: synthetic calls -> layered PII redaction -> chunk/embed redacted text only -> vector store -> RAG answers with cited call snippets.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm   # already in requirements.txt; run this if it's missing
```

spaCy 3.8.16 ships wheels for Python 3.11 to 3.14.

## Synthetic data

```powershell
python scripts/generate_data.py --n 400 --seed 42
python scripts/split_overlap.py
pytest
```

Calls are split into **train / dev / test** (about 60/15/25). The splits are disjoint in everything a model could memorise:

- **MultiWOZ 2.2** dialogues are reshaped into agent/customer calls, with synthetic PII injected into customer turns. Each split draws from the matching official MultiWOZ file (train, dev or test). Any dialogue that already contains phone numbers, emails or long digit runs is dropped, because that text has no labels.
- **Finance templates** cover refunds, billing disputes, card replacement, identity verification, loan/account inquiries and contact changes. There are 12 train templates, 5 dev and 5 test, and none is shared between splits.
- **Injection phrases** for MultiWOZ calls (greetings, name / phone / email / card sentences, distractors) come from a separate pool for each split.
- **Names** vary in form: first only, last only, full name, title + surname, hyphenated, apostrophes, and names that are also common words. Dev and test use name lists disjoint from train's. They also include patterns train never sees: common-word first names (Grace, Will), Dr/Mrs titles, and particle surnames (van der Berg).
- **Transcript noise** hits about 25% of calls: lowercase, punctuation dropped, "um"/"uh" fillers added. Offsets are computed after the transform.

`split_overlap.py` reports how often a dev/test name's 3-word left context also occurs in train, and checks that no template or injection line is shared (it should report 0%).

Each call in `calls.jsonl` has `split`, `source`, `source_id`, `noise`, `turns`, `labels` (PII spans with character offsets into the turn text) and `distractors`. Distractors are number-like non-PII spans: order numbers and reference numbers that fail their checksums, plus amounts, dates and times. A redactor should not flag them.

All personal data is fake, generated with Faker and checksum generators. `data/raw/` is gitignored, and generated calls or downloaded datasets must never be committed.

## Redaction eval

```powershell
python -m callsearch.eval.redaction_eval --detector pipeline --split dev
```

The eval prints per-type precision/recall/F1, exact-match recall, a leak check and distractor false positives. Detectors are `regex`, `ner`, and `pipeline` (regex + ner, merged by type priority). Results are saved to `results/eval_<detector>_<split>.json`. Pass `--json PATH` to save somewhere else, or `--no-save` to only print.

**Tune on `dev`. Run `test` only for reported numbers.**

## NER layer (PERSON)

```powershell
python scripts/train_ner.py --base sm --epochs 30 --dropout 0.2 --seed 42 --out models/ner_spacy
python scripts/show_errors.py --detector pipeline --split dev --type PERSON --n 15
```

`train_ner.py` trains a fresh PERSON-only NER component on the train split, on top of en_core_web_sm's pretrained tok2vec (`--base blank` starts from scratch instead). It picks the best epoch on the dev split and stops early once dev F1 stops improving. The test split is never read. The model goes to `models/` (gitignored) and a summary to `results/train_ner_spacy.json`.

`show_errors.py` prints false positives and negatives with context. It defaults to dev and refuses `--split test` unless `--allow-test` is passed.

## Data and results layout

| Path | Contents | In git |
|---|---|---|
| `data/raw/` | Generated calls with PII, plus downloaded MultiWOZ | No (gitignored) |
| `data/redacted/` | Redacted transcripts only. The index builder reads from here. | Yes |
| `data/index/` | Vector index built from redacted text | Yes |
| `models/` | Trained models | No (gitignored) |
| `results/` | Eval and training metrics only (counts, rates, type names), never transcript text. A test enforces this. | Yes |
| `results/v1/` | Archived v1 baseline. Its dev set was carved out of train and shared templates and phrasing with it (dev F1 1.000 vs test 0.884), so it is not comparable to current results. | Yes |

## Third-party data and licenses

- **MultiWOZ 2.2**, from [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz). MIT License, Copyright (c) 2019 Paweł Budzianowski. It is downloaded on first run to `data/raw/multiwoz/` (`train/`, `dev/` and `test/dialogues_001.json`) and not redistributed here.
