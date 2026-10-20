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

The eval prints per-type precision/recall/F1, exact-match recall, a leak check and distractor false positives. Detectors are `regex`, `ner`, `pipeline` (regex + ner, merged by type priority), `presidio`, `presidio_casin`, `llm` and `pipeline_llm` (see below). Results are saved to `results/eval_<detector>_<split>.json`. Pass `--json PATH` to save somewhere else, or `--no-save` to only print.

**Tune on `dev`. Run `test` only for reported numbers.**

## NER layer (PERSON)

```powershell
python scripts/train_ner.py --base sm --epochs 30 --dropout 0.2 --seed 42 --out models/ner_spacy
python scripts/show_errors.py --detector pipeline --split dev --type PERSON --n 15
```

`train_ner.py` trains a fresh PERSON-only NER component on the train split, on top of en_core_web_sm's pretrained tok2vec (`--base blank` starts from scratch instead). It picks the best epoch on the dev split and stops early once dev F1 stops improving. The test split is never read. The model goes to `models/` (gitignored) and a summary to `results/train_ner_spacy.json`.

`show_errors.py` prints false positives and negatives with context. It defaults to dev and refuses `--split test` unless `--allow-test` is passed.

## Presidio baseline (off-the-shelf comparison)

```powershell
python -m callsearch.eval.redaction_eval --detector presidio --split dev
python -m callsearch.redact.presidio_baseline --split dev   # distractor hits by any Presidio entity
python scripts/compare_results.py --split dev                # -> results/comparison_dev.md
```

`presidio` runs `presidio-analyzer` 2.2.364 with its **default configuration**: `AnalyzerEngine()` with default recognizers and its default spaCy model, **en_core_web_lg 3.8.0** (Presidio downloads it on first use). There are no custom recognizers and no threshold tuning. Where mapped spans overlap, the higher Presidio score wins, matching how Presidio's anonymizer resolves conflicts.

Entity mapping (Presidio -> ours):

| Presidio entity | Ours | Notes |
|---|---|---|
| `PERSON` | PERSON | spaCy NER via en_core_web_lg |
| `EMAIL_ADDRESS` | EMAIL | |
| `PHONE_NUMBER` | PHONE | |
| `CREDIT_CARD` | CARD | regex + Luhn |
| `CA_SIN` | SIN | Presidio has a built-in Canadian SIN recognizer (`CaSinRecognizer`, regex + Luhn), but it is **disabled in the default config**. Under the default baseline it never fires, so Presidio's SIN recall is 0. |
| `US_BANK_NUMBER` | ACCOUNT | Presidio's only built-in bank-account recognizer (8–17 digits, boosted by words like "account"). It is US-labelled but format-generic, and enabled by default. |

Two Presidio detectors are registered:

- **`presidio`**: Presidio's default configuration, exactly as shipped.
- **`presidio_casin`**: the same configuration with one change. The built-in `CaSinRecognizer` is set to `enabled: true` in Presidio's own `default_recognizers.yaml`. This is config only, with no custom recognizer. It is a fairness check, since the default config can't detect Canadian SINs at all.

**Unmapped** (ignored in scoring): `US_SSN` (a US ID with a different 3-2-4 format; mapping it to SIN would inflate the score), `LOCATION`, `DATE_TIME`, `NRP`, `URL`, `IP_ADDRESS`, `IBAN_CODE`, `CRYPTO`, `MAC_ADDRESS`, `MEDICAL_LICENSE`, `UK_NHS`, `US_ITIN`, `US_DRIVER_LICENSE`, `US_PASSPORT`. The `presidio_baseline` CLI still counts every entity, mapped or not, that lands on a distractor, and lists labels touched only by an unmapped entity. Results go to `results/presidio_extras_<split>.json`.

## LLM layer (Layer 3, add-only)

```powershell
python -m callsearch.eval.redaction_eval --detector pipeline_llm --split dev --dry-run         # estimate only
python -m callsearch.eval.redaction_eval --detector pipeline_llm --split dev --max-requests 700
python -m callsearch.redact.llm --split dev --show-false 15   # correct vs false LLM additions (from cache) -> results/llm_additions_dev.json
```

Current dev result, from **`ollama:qwen2:7b`, prompt v1, PERSON-only post-filter**, re-scored from the cache:

| | PERSON precision | recall | F1 | leak |
|---|---:|---:|---:|---:|
| pipeline | 0.933 | 0.909 | 0.921 | 9.1% |
| pipeline_llm | 0.839 | 0.981 | 0.904 | 1.9% |

Structured types and distractor false positives are identical to `pipeline` (100% and 0). The Claude provider has not been evaluated yet.

- **`pipeline_llm`**: regex + ner first. The LLM then sees each turn with those spans already masked (`[CARD]`, `[PERSON]`, ...) and lists any names still visible. Its spans are only **added**; base spans are never removed. **`llm`** is the LLM alone, for ablation (it sees the turn unmasked).
- **PERSON only.** Regex is exact on the structured types. On dev, every EMAIL/PHONE/CARD/SIN/ACCOUNT span the LLM added was a false positive, including all 23 distractor hits. So only PERSON spans are kept, through a post-filter that applies to every prompt version.
- **Prompt versions:** `v1` asked for all six types; `v2` (the default) asks for PERSON only. Old versions stay in `PROMPTS` so their cached replies can be re-scored with `--llm-prompt v1`. The version is part of the cache key.
- **Parse accounting:** `{}` and `{"spans": []}` count as valid empty answers. Only unparseable replies count as parse failures.
- **Concurrency:** Ollama uses 1 worker, since a local model serves one request at a time, and Anthropic uses 8. Override with `--workers N`. Progress prints every 100 requests.
- **Provenance:** LLM results must name the model that produced them. Every LLM eval JSON stores `"llm": {"provider", "model", "prompt_version"}`, and comparison tables label the detector, e.g. `pipeline_llm [ollama:qwen2:7b prompt v1]`. Files saved before this was recorded show `[model unknown]`.
- **Offsets are ours.** The model returns `{"spans": [{"text", "type"}]}`, and we find every verbatim occurrence of each substring. We discard:
  - substrings that aren't in the turn
  - types outside PERSON / EMAIL / PHONE / CARD / SIN / ACCOUNT
  - matches touching a placeholder
  - matches that start or end mid-word
  
  Unparseable replies give no spans and are counted.
- **Providers:**

  | `CALLSEARCH_LLM_PROVIDER` | Model setting | Key / host |
  |---|---|---|
  | `anthropic` (default) | `CALLSEARCH_LLM_MODEL` (default `claude-haiku-5-5`) | key from `ANTHROPIC_API_KEY`; never hardcode or commit it |
  | `ollama` (local) | `CALLSEARCH_OLLAMA_MODEL` | host from `OLLAMA_HOST` |

  Both run at temperature 0.
- **Cache:** replies are cached in `.cache/llm/` (gitignored), keyed by provider + model + prompt version + masked text. Re-running an eval is free and reproducible.
- **Cost:**
  - `--dry-run` prints requests and estimated tokens without calling the model.
  - `--max-requests N` aborts before any request if more would be needed.
  - A real run prints requests made, cache hits, parse failures, tokens and estimated cost. Prices come from `CALLSEARCH_LLM_PRICE_IN` / `_OUT` (USD per million tokens), defaulting to 1 / 5.
- **Privacy:** only text already masked by regex + ner is sent, and all of it is synthetic. In a real deployment that text would still leave the machine for a hosted API, and anything the first two layers missed would go with it. That is why the provider interface also supports a local Ollama model.

## Data and results layout

| Path | Contents | In git |
|---|---|---|
| `data/raw/` | Generated calls with PII, plus downloaded MultiWOZ | No (gitignored) |
| `data/redacted/` | Redacted transcripts only. The index builder reads from here. | Yes |
| `data/index/` | Vector index built from redacted text | Yes |
| `models/` | Trained models | No (gitignored) |
| `.cache/llm/` | Cached LLM replies | No (gitignored) |
| `results/` | Eval and training metrics only (counts, rates, type names), never transcript text. A test enforces this. | Yes |
| `results/v1/` | Archived v1 baseline. Its dev set was carved out of train and shared templates and phrasing with it (dev F1 1.000 vs test 0.884), so it is not comparable to current results. | Yes |

## Third-party data and licenses

- **MultiWOZ 2.2**, from [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz). MIT License, Copyright (c) 2019 Paweł Budzianowski. It is downloaded on first run to `data/raw/multiwoz/` (`train/`, `dev/` and `test/dialogues_001.json`) and not redistributed here.
