# call-search

PII-safe semantic search over synthetic call transcripts.

Pipeline: synthetic calls -> layered PII redaction -> chunk/embed redacted text only -> vector store -> RAG answers with cited call snippets.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Synthetic data

```powershell
python scripts/generate_data.py --n 200 --seed 42 --out data/raw/calls.jsonl
pytest
```

The generator draws on two sources:

- **MultiWOZ 2.2** dialogues, reshaped into agent/customer calls. Synthetic PII (name, phone, email, card) is injected into customer turns. Any dialogue that already contains phone numbers, emails or long digit runs is dropped, because that text has no labels. Train calls come from MultiWOZ's official train file and test calls from its official test file.
- **Finance templates** (refund, billing dispute, card replacement, identity verification). The test split uses held-out templates and injection phrases.

Each call in `calls.jsonl` has `turns`, `labels` (PII spans with character offsets into the turn text) and `distractors`. Distractors are number-like non-PII spans: order numbers and reference numbers that fail their checksums, plus amounts, dates and times. A redactor should not flag them.

All personal data is fake, generated with Faker and checksum generators. `data/raw/` is gitignored, and generated calls or downloaded datasets must never be committed.

## Third-party data and licenses

- **MultiWOZ 2.2**, from [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz). MIT License, Copyright (c) 2019 Paweł Budzianowski. It is downloaded on first run to `data/raw/multiwoz/` (`train/dialogues_001.json`, `test/dialogues_001.json`) and not redistributed here.
