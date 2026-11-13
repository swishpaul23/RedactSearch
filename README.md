# Redact-Search

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

The eval prints per-type precision/recall/F1, exact-match recall, a leak check and distractor false positives. Detectors are `regex`, `ner`, `pipeline` (regex + ner, merged by type priority), `ner_hf`, `pipeline_hf` (the DistilBERT versions), `presidio`, `presidio_casin`, `llm` and `pipeline_llm` (see below). Results are saved to `results/eval_<detector>_<split>.json`. Pass `--json PATH` to save somewhere else, or `--no-save` to only print.

**Tune on `dev`. Run `test` only for reported numbers.**

## NER layer (PERSON)

```powershell
python scripts/train_ner.py --base sm --epochs 30 --dropout 0.2 --seed 42 --out models/ner_spacy
python scripts/show_errors.py --detector pipeline --split dev --type PERSON --n 15
```

`train_ner.py` trains a fresh PERSON-only NER component on the train split, on top of en_core_web_sm's pretrained tok2vec (`--base blank` starts from scratch instead). It picks the best epoch on the dev split and stops early once dev F1 stops improving. The test split is never read. The model goes to `models/` (gitignored) and a summary to `results/train_ner_spacy.json`.

`show_errors.py` prints false positives and negatives with context. It defaults to dev and refuses `--split test` unless `--allow-test` is passed.

### DistilBERT comparison (Hugging Face Transformers)

```powershell
python scripts/train_ner_hf.py --epochs 6 --lr 5e-5 --batch-size 16 --seed 42 --out models/ner_distilbert
python -m callsearch.eval.redaction_eval --detector ner_hf --split dev
python -m callsearch.eval.redaction_eval --detector pipeline_hf --split dev
```

A second NER model from a different family, to check whether the spaCy choice holds up. `train_ner_hf.py` fine-tunes `distilbert/distilbert-base-cased` as a token classifier and follows the same protocol as `train_ner.py`: it trains on train, picks the best epoch on dev by exact-match PERSON F1, stops early, and never reads test. The first run downloads the base model (about 260 MB). It runs on the CPU unless CUDA is available. The model goes to `models/ner_distilbert/` (gitignored) and a summary, including the parameter count, model size and training time, goes to `results/train_ner_distilbert.json`.

- **Labels:** BIO tags (`O`, `B-PERSON`, `I-PERSON`) on every subword. Gold character offsets are aligned to tokens through the fast tokenizer's offset mapping, and predictions are mapped back the same way, then grown to whole words.
- **Long turns:** split into overlapping 256-token windows at inference.
- **Detectors:** `ner_hf` is DistilBERT alone and `pipeline_hf` is regex + DistilBERT, the counterparts of `ner` and `pipeline`.

**Results (PERSON).** Both models were trained with seed 42 on the same train split and selected on dev. Test was run once.

| model | dev F1 | test precision | test recall | test F1 | test leak | size | CPU training |
|---|---:|---:|---:|---:|---:|---:|---:|
| spaCy (`ner`, en_core_web_sm tok2vec) | 0.855 | 0.924 | 0.865 | 0.893 | 14.3% | 7.8 MB | 3 min |
| DistilBERT (`ner_hf`, distilbert-base-cased) | 0.945 | 0.981 | 0.959 | 0.970 | 4.1% | 261 MB | 31 min |
| Presidio default (en_core_web_lg) | none | 0.807 | 0.910 | 0.855 | 9.4% | none | none |

- **Dev F1** is exact span match, as reported by each training script. The test columns come from `redaction_eval`.
- **With regex** (`pipeline_hf` vs `pipeline`) on test: overall leak rate 2.3% vs 7.9%. The structured types stay at 100% recall and 0 distractor hits in both.
- **Trade-off:** DistilBERT leaks 3.5 times fewer names than spaCy, but it is 33 times larger (65.2M parameters) and trains 11 times slower on a laptop CPU.
- **Single run:** one seed per model. The gap is large, but it has not been checked across seeds.

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

**PERSON ablation on dev.** Every LLM row names the model and prompt that produced it. All rows include the PERSON-only post-filter.

| detector | model | prompt | PERSON precision | recall | leak |
|---|---|---|---:|---:|---:|
| pipeline (regex + ner) | none | none | 0.933 | 0.909 | 9.1% |
| pipeline_llm | `ollama:qwen2:7b` | v1 | 0.839 | 0.981 | 1.9% |
| pipeline_llm | `ollama:qwen2:7b` | v2 | 0.640 | 0.994 | 0.6% |
| pipeline_llm | `ollama:qwen2:7b` | v3 | 0.704 | 0.987 | 1.3% |

- **Structured types and distractors:** in every row they are identical to `pipeline` (100% and 0 false positives).
- **v2 vs v1:** v2 finds 2 more names but adds 57 more false positives, mostly places, restaurants and words next to placeholders. That is why Ollama defaults to v1.
- **v3 vs v1:** v3 finds 1 more name but adds 35 more false positives (64 vs 29). Many false hits echo words the prompt tells the model to avoid (`applicant's`), plus pronouns (`I`, `you`), titles (`Mr.`) and restaurant names. On `qwen2:7b`, listing what not to return made the model return it. Ollama keeps v1.
- **Not evaluated yet:** the Claude provider.
- **Results files:** v1 results are archived in `results/llm_prompt_v1/` and v3 results in `results/llm_prompt_v3/`.

- **`pipeline_llm`**: regex + ner first. The LLM then sees each turn with those spans already masked (`[CARD]`, `[PERSON]`, ...) and lists any names still visible. Its spans are only **added**; base spans are never removed. **`llm`** is the LLM alone, for ablation (it sees the turn unmasked).
- **PERSON only.** Regex is exact on the structured types. On dev, every EMAIL/PHONE/CARD/SIN/ACCOUNT span the LLM added was a false positive, including all 23 distractor hits. So only PERSON spans are kept, through a post-filter that applies to every prompt version.
- **Prompt versions:**
  - `v1` asks for all six types.
  - `v2` asks for PERSON only.
  - `v3` asks for PERSON only. It restores v1's rules that businesses and places are not people, says placeholders mark data already masked (never return one or the words beside it), and gives 4 short examples with invented names and places, none taken from the eval data.

  Each provider has its own default: Ollama uses `v1` (v2 measured worse on `qwen2:7b`) and Anthropic uses `v2`. Pick another with `--llm-prompt`. All versions stay in `PROMPTS`, and the version is part of the cache key.
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
  | `anthropic` (default) | `CALLSEARCH_LLM_MODEL` (default `claude-haiku-5-5`) | key from `ANTHROPIC_API_KEY`, in the environment or the repo-root `.env` (gitignored). If the key isn't scoped to a workspace, also set `ANTHROPIC_WORKSPACE_ID` (sent as the `anthropic-workspace-id` header). Only the provider reads `.env`, and only these names and the model name. The key is never printed, logged or committed. |
  | `ollama` (local) | `CALLSEARCH_OLLAMA_MODEL` | host from `OLLAMA_HOST` |

  Ollama runs at temperature 0. The anthropic SDK (1.15) no longer accepts sampling parameters, so Claude runs use the API default. Reruns are reproducible through the reply cache.
- **Cache:** replies are cached in `.cache/llm/` (gitignored), keyed by provider + model + prompt version + masked text. Re-running an eval is free and reproducible.
- **Cost:**
  - `--dry-run` prints requests and estimated tokens without calling the model.
  - `--max-requests N` aborts before any request if more would be needed.
  - A real run prints requests made, cache hits, parse failures, tokens and estimated cost. Prices come from `CALLSEARCH_LLM_PRICE_IN` / `_OUT` (USD per million tokens), defaulting to 1 / 5.
- **Privacy:** only text already masked by regex + ner is sent, and all of it is synthetic. In a real deployment that text would still leave the machine for a hosted API, and anything the first two layers missed would go with it. That is why the provider interface also supports a local Ollama model.

## Search index (Phase 3)

```powershell
python -m callsearch.redact.export --detector pipeline_hf   # raw -> data/redacted/ + manifest
python scripts/build_index.py --model bge-small --context-turns 1
python -m callsearch.rag.guard                              # -> results/index_guard.json
python -m callsearch.rag.store "customer wants a refund" -k 5
```

- **Redaction boundary:** `callsearch.redact.export` runs a detector over every turn and writes `data/redacted/calls.jsonl`. It keeps only the call id, split, source, speaker, start time and redacted text. Labels, distractors and template ids are dropped. `manifest.json` records the detector and the file's sha256.
- **Index builder:** `build_index.py` reads `data/redacted/` only and never opens `data/raw/`. It refuses to run if the manifest is missing, does not mark the file as redacted, or the hash does not match. Before writing anything, it runs the regex layer over every chunk and refuses to build if it finds structured PII.
- **Chunks:** one chunk per speaker turn, with id `<call_id>:<turn_id>` and the speaker, start time, split and source as metadata. The embedded text includes the previous turn (`--context-turns 1`), so a short reply like "yes, that one" keeps what it answers. The stored and displayed text is the turn alone.
- **Embeddings:** sentence-transformers, `BAAI/bge-small-en-v1.5` by default (`--model minilm` for `all-MiniLM-L6-v2`). Vectors are normalised, and queries get BGE's search prefix. The collection records the model, so queries always use the model that built the index.
- **Store:** Chroma, persistent in `data/index/chroma/`, with cosine distance. Rebuilding replaces the collection.
- **Search modes** (`--mode` on `callsearch.rag.store`, `callsearch.rag.answer` and `rag_eval`):
  - `dense`: embeddings only.
  - `bm25`: keyword scoring (Okapi BM25, k1 1.5, b 0.75), built in memory from the stored documents. The tokenizer keeps times, prices and ids whole (`05:00`, `18.88`, `tr7423`) and drops placeholders, so redacted data never becomes a match key.
  - `hybrid`: the top 50 of each list fused with reciprocal rank fusion (`1 / (60 + rank)`, summed over both lists).

  Dense embeddings blur exact tokens such as train ids, times and venue names, and BM25 matches only those tokens, so the two miss different questions.
- **Index guard** (`callsearch.rag.guard`):
  1. A pattern scan of every stored document with the regex layer. On a regex-based redactor this is a check against bugs and hand edits, not an independent detector.
  2. A planted-value scan: every labeled value from `data/raw/` is looked up verbatim in the stored text of its own turn. This is the same leak test `redaction_eval` uses, applied to what is actually in the index. It is evaluation-only and the one place outside the eval scripts that reads raw data.

  Structured leaks always fail. Leaked names fail unless `--allow-person-leaks` is passed, and the count is saved either way. `test_built_index_has_no_structured_pii` re-runs the pattern scan on the built index whenever one exists.

## Answers with citations (Phase 3)

```powershell
$env:CALLSEARCH_LLM_PROVIDER="ollama"; $env:CALLSEARCH_OLLAMA_MODEL="qwen2:7b"   # or anthropic
python -m callsearch.rag.answer "Why did the customer want a refund?" -k 5 [--mode bm25] [--neighbors 1] [--prompt v1] [--split test] [--json]
```

1. The top `k` chunks come from the index. They are already redacted, and nothing else ever goes into the prompt.
2. The model sees them as numbered excerpts (call id, turn, speaker, time, text). It must answer from them only, cite the excerpt numbers it used, treat placeholders as removed data, and reply `NOT_FOUND` when the excerpts do not answer the question.
3. **Server check:** the reply must parse as JSON, cite at least one excerpt, and cite only excerpts that were retrieved. A reply that fails any of these is discarded, returned as `NOT_FOUND`, and the reason is recorded in `rejected`. A model that says `NOT_FOUND` is a valid outcome, not a rejection.
4. **Grounding check** (on by default, `--no-grounding` to skip): every rare content word of the question must appear in the excerpts the answer cites. "Rare" means it occurs in at most 0.1% of stored chunks: venue names, ids, times and unusual nouns such as "sauna" or "ferry", but not "hotel" or "train". A question word that occurs in no chunk at all cannot be supported by any excerpt, so the answer is `NOT_FOUND` without calling the model. This was added because `qwen2:7b` ignored instructions to abstain (prompt v2 made no difference). On dev, 7 of 8 unanswerable questions got invented answers ("Warkworth House offers a sauna") or claims about all calls based on 5 excerpts ("No one asked about a ferry"). In each case the question's key word was missing from the cited excerpt. The 0.1% threshold was chosen on dev.
5. With `--neighbors n`, each excerpt also shows the `n` turns before and after the matched turn, with the matched turn marked. An answer often sits in the reply to the turn that matched. For example, the customer's turn names the restaurant and the agent's next turn gives the price. Citations still point at matched turns.
6. The response gives the answer plus, for each citation, the call id, turn id, speaker, start time and redacted snippet.

The provider is the one the LLM redaction layer uses (`CALLSEARCH_LLM_PROVIDER`), so Claude and a local Ollama model are interchangeable. Replies are cached in `.cache/answers/`, keyed by provider, model, prompt version, question and retrieved chunks, so eval reruns are free. Prompts are versioned in `PROMPTS` like the redaction prompts.

## RAG eval (Phase 3)

```powershell
python -m callsearch.eval.rag_eval --split dev --retrieval-only   # fast, no LLM
python -m callsearch.eval.rag_eval --split dev                    # + cited answers
python -m callsearch.eval.rag_eval --split test                   # reported numbers, run once
```

**Question sets** (`data/questions/rag_<split>.jsonl`, hand-written, in git): dev has 35 answerable + 8 unanswerable questions for tuning, and test has 44 + 10 for reported numbers. The first 15 + 5 dev questions were run before the set was extended and are kept unchanged. Two of them (`dev_a06`, `dev_a13`) put the call's distinguishing detail only in the answer, not in the question, so no retriever can be expected to find them. The later questions carry that detail in the question itself. Each answerable question names its gold call and turns, a list of `evidence` phrases, and `answer_terms` a correct answer should contain. Unanswerable questions list `absent` phrases that no stored turn contains.

**Validation before scoring:** the eval checks every question against the text actually stored in the index. The evidence must be in the gold call, the gold turns must exist, and no other call may contain all the evidence phrases. Without this check, a question about a template call would have dozens of right answers, because the 5 templates repeat their phrasing word for word across calls. Unanswerable questions must have no turn containing their `absent` phrases. If any question fails, the run stops and lists it. Answerable questions are drawn from MultiWOZ calls only, since template calls differ almost only in their (redacted) personal data.

**Metrics.** Retrieval searches the whole index (all splits), as a user would.

| metric | meaning |
|---|---|
| recall@1/3/5/10 | a chunk from the gold call is in the top n |
| turn_hit@k | a gold turn itself is in the top k |
| turn_shown@k | a gold turn is in the top k or within `--neighbors` turns of a top-k hit, so the model sees the answer turn |
| mrr@10 | mean of 1 / rank of the gold call's first chunk |
| answer_found | answerable questions that got an answer (not NOT_FOUND) |
| citation_accuracy | the answer cites a chunk from the gold call |
| answer_term_match | the answer text contains an expected term (a string check; an LLM judge comes later) |
| false_not_found | answerable questions answered NOT_FOUND |
| abstention_rate | unanswerable questions answered NOT_FOUND |

**Dev results** (`qwen2:7b`, prompt v1, k 5, bge-small index with 1 context turn). The first row is the original setup, and the defaults are now `--mode bm25 --neighbors 1`.

| retrieval | neighbors | recall@5 | turn_shown@5 | citation_accuracy | answer_term_match | abstention_rate |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0 | 0.657 | 0.571 | 0.571 | 0.629 | 0.625 |
| dense | 1 | 0.657 | 0.629 | 0.629 | 0.686 | 0.250 |
| **bm25** | **1** | **0.914** | **0.714** | **0.771** | **0.686** | 0.125 |
| hybrid | 1 | 0.771 | 0.657 | 0.686 | 0.657 | 0.125 |
| **bm25 + grounding check** | **1** | **0.914** | **0.714** | **0.714** | **0.629** | **1.000** |

- BM25 finds the right call most often. Neighbouring turns then let the model see the answer, which is usually in the reply to the matched turn.
- With more context, the v1 prompt almost never says NOT_FOUND on unanswerable questions (1 of 8). Prompt v2, which tells the model to check every detail first, changed nothing (1 of 8) and lost 2 correct answers, so v1 stays.
- The grounding check (default) turns all 8 unanswerable questions into NOT_FOUND. It also turns 5 answerable ones into NOT_FOUND: 3 of those answers cited the wrong call anyway, 1 question names a surname the redactor masked, and 1 is a real loss (`dev_a26`: the venue name is two turns before the answer, outside the one-turn window). When the system answers, it is right 25 of 27 times (93%), against 27 of 32 (84%) without the check.
- **Caveats:** the dev set has 35 + 8 questions, so one question moves a rate by 3 or 12.5 points. The questions also name exact details (train ids, reference numbers), which favours keyword search; vaguer real-world questions may favour dense or hybrid.

**Test results** (run once, with the setup chosen on dev: bm25, 1 neighbour, prompt v1, grounding check, `qwen2:7b`, k 5):

| metric | dev | test |
|---|---:|---:|
| recall@5 | 0.914 | 0.818 |
| turn_shown@5 | 0.714 | 0.591 |
| citation_accuracy | 0.714 | 0.636 (28/44) |
| answer_term_match | 0.629 | 0.477 (21/44) |
| false_not_found | 0.229 | 0.227 |
| abstention_rate | 1.000 | 1.000 (10/10) |
| answers citing the right call, of those given | 25/27 (93%) | 28/34 (82%) |

- Test is lower than dev on every answerable metric. That is the usual cost of choosing the setup on dev, with small sets (one test question is 2.3 points).
- Abstention held at 10 of 10, so the grounding check generalises to unseen unanswerable questions.
- `answer_term_match` is a plain string check. 7 answers cite the right call but miss the expected string, and some of those may be correct in other words (an LLM judge is planned). The score should be read as a lower bound.

Results go to `results/rag_eval_<split>.json` (with a `_retrieval` suffix when run with `--retrieval-only`). They hold the index and LLM provenance, the metrics, and per-question ids with outcomes. No text.

## Agent (multi-step questions)

```powershell
python scripts/make_agent_questions.py                 # data/questions/agent_test.jsonl (answers computed from the data)
python -m callsearch.eval.agent_eval --compare-rag      # results/agent_eval_agent_test.json
```

Some questions need more than one search: "In the test split, do agents or customers mention parking more, and what was said?" The agent (`src/callsearch/agent/`) answers these by calling tools in a loop: it picks a tool, reads the result, and repeats until it can answer or reaches the 6-step cap. Choose **Agent** on the Ask page, or call `POST /api/agent`.

| tool | what it returns |
|---|---|
| `search_calls(query, k, split, source)` | matching turns from the index, with turn ids |
| `get_transcript(call_id)` | one whole redacted call |
| `compute_stats(query, split, source)` | calls, turns and words by speaker, agent talk share, call length; with a query, matching calls and turns per speaker |
| `redaction_report()` | saved test-split redaction scores and the index check |

- **Any model:** the loop speaks plain JSON (`{"tool", "args"}` or `{"answer", "citations"}`) through the same provider interface as answering, so it runs on Ollama locally and on Claude unchanged. Replies are cached in `.cache/agent/`.
- **Rules in code, not the prompt:** tools read only through `CallStore` (redacted data behind the manifest check), the index and `results/`; none takes a path, and `Tools` refuses a call store pointed at `data/raw/`. Arguments are validated, a repeated identical call is not re-run, and malformed replies use up a step.
- **Citations are checked:** the answer may cite turn ids, call ids or a statistics result (`r2`) only if they appeared in a tool result during that run. Otherwise it becomes not found, as in the fixed pipeline.
- **Evaluation:** 20 multi-step test-split questions in six types (speaker comparison, count plus example, locate then read the call, report plus statistics, three tools, unanswerable), each with expected tools and answers computed from the data. The eval reports tool-selection accuracy (and in order), task success, average steps and step-cap hits; `--compare-rag` also runs the 54 single-step questions through the agent next to the fixed pipeline's saved scores.
- **PII guard:** `test_agent_never_sees_or_returns_planted_values` redacts calls with planted names, cards and emails, runs every tool while any attempt to open `data/raw` fails the test, and checks that no planted value appears in tool results, prompts, the answer or the step trace.

## Web app

```powershell
# API (reads data/redacted/, the index and results/; serves web/dist when built)
python -m callsearch.api.main                       # http://127.0.0.1:8000, API docs at /api/docs

# Frontend in development (Node 20+), proxied to the API
cd web; npm ci; npm run dev                         # http://localhost:5173

# Production: build once, then the API serves the app at /
cd web; npm run build; cd ..; python -m callsearch.api.main
```

FastAPI backend and a React + TypeScript frontend (Vite, TanStack Query, React Router). Five pages, each built on one standard layout pattern:

| page | pattern | what it is for |
|---|---|---|
| Ask (`/`) | split pane | Ask a question. The answer and its cited turns are on the left; the selected call's transcript is on the right, with the cited turn highlighted. A refusal explains why ("No call mentions 'sauna'"). |
| Calls (`/calls`) | list | Search, filter, sort and page through the 400 redacted calls. The state lives in the URL, so views can be shared. |
| Call (`/calls/:id`) | detail | Full redacted transcript, plus a Redactions tab with counts by type. `?turn=n` highlights a cited turn. |
| Upload (`/upload`) | form | Add your own transcripts, one file or a batch. They are redacted in memory, saved with the redacted data and added to the search index; the page shows progress, links to the new calls and lists any lines that could not be read. |
| Quality (`/quality`) | dashboard | Test-split results read from `results/`: redaction comparison, search and answer scores, index check, name models. |

**API** (typed with Pydantic; the schema is at `/api/openapi.json`):

| route | |
|---|---|
| `GET /api/health` | calls loaded, index status, answer model |
| `POST /api/agent` | `{question}` → answer, cited turns, calls and statistics results, every step the agent took |
| `POST /api/ask` | `{question, k, mode, split}` → cited answer, citations, discard reason, provenance, timing |
| `GET /api/search` | retrieval only, no LLM |
| `GET /api/calls` | `q`, `split`, `source`, `sort`, `order`, `page`, `page_size` |
| `GET /api/calls/{id}` | redacted transcript and redaction counts |
| `GET /api/metrics` | test-split results for the Quality page |
| `POST /api/uploads` | multipart `files` → 202 and an upload job; redaction and indexing run in the background |
| `GET /api/uploads/{id}` | job status (`queued`, `redacting`, `indexing`, `done`, `failed`), imported call ids, redaction counts, per-file issues |
| `GET /api/uploads` | recent jobs since the server started |

- **Redacted data only.** The API reads `data/redacted/` through its manifest check, the index and `results/`. It never opens `data/raw/`.
- **Errors** always come back as `{"error": {"code", "message"}}`, and the message says how to fix the problem, for example "The search index is not built. Run: python scripts/build_index.py". A missing index, missing calls or an unreachable answer model return 503 with a stable code.
- **Operations:** every response carries an `X-Request-ID`. Logs are JSON lines with method, path, status and timing, never question or transcript text (a test checks this). `/api/ask` is rate limited per client (`CALLSEARCH_ASK_PER_MINUTE`, default 20) and returns 429 with `Retry-After`. Security headers include a strict Content-Security-Policy (`'self'` only, no inline scripts), `nosniff`, `no-referrer` and `frame-ancestors 'none'`.
- **Uploads:**
  - **Formats:** `.txt` (one call, `Agent:` / `Customer:` lines, optional `[mm:ss]`), `.jsonl` (one call per line) or `.json` (a call or a list of calls), each turn `{"speaker", "text", "start_sec"?}`. Up to 20 files of 2 MB and 200 calls per upload; `CALLSEARCH_UPLOADS_PER_MINUTE` (default 10) limits uploads per client.
  - **Redaction first:** files are parsed and redacted in memory with the same regex + DistilBERT pipeline as the dataset, then checked again for card, SIN, account, email and phone patterns. The raw text is never written to disk or logged. Problems are reported by location only ("Line 3: Not valid JSON"), never by quoting the file.
  - **Where calls go:** uploaded calls get ids like `up-<job>-001`, split and source `upload`, so evaluation splits are unchanged. They are appended to `data/redacted/` (with a new manifest) and added to the existing index with the index's own embedding model. With no index built yet, they are saved and become searchable on the next `build_index.py` run.
- **Typed end to end:** `scripts/export_openapi.py` writes `web/openapi.json`, and `npm run gen:api` generates `web/src/api/schema.d.ts` from it. `test_openapi_file_matches_code` fails if the API changes without regenerating, so the frontend types cannot drift.
- **Accessibility:**
  - **Navigation and focus:** a skip link; landmarks and labelled regions; focus moves to the page heading on navigation; a visible focus ring everywhere.
  - **Keyboard:** `/` focuses the page's search, `Enter` asks, arrow keys move through sources and tabs.
  - **Screen readers:** a live region announces answers. Redaction marks read as "redacted name", not "[PERSON]". Sortable columns expose `aria-sort`.
  - **Display:** light and dark themes that follow the system or a toggle; reduced motion respected; no horizontal scrolling at 375 px; 44 px minimum touch targets.
  - **Checks:** axe-core runs in the unit tests (`npm test`) and found no violations in a real-browser pass of every page at 375 px and 1280 px.
- **Design:** the standards (aesthetic, banned patterns, WCAG 2.1 AA target, voice, Lucide icons) are recorded in `.ux-profile.md`, and `AGENTS.md` points coding agents to it. Quiet, document-review styling in Atkinson Hyperlegible Next, a typeface designed by the Braille Institute for legibility. Redacted values are solid ink marks, and the cited turn is the only highlighted thing on screen.

## AWS: S3 + Lambda redaction (optional)

```text
data/upload/*.jsonl ──aws s3 cp──► s3://callsearch-raw-<id>/incoming/      (KMS-encrypted, deleted after 7 days)
                                        │ s3:ObjectCreated (*.jsonl)
                                        ▼
                                   Lambda callsearch-redact  (container image: regex + DistilBERT)
                                        │
                                        ▼
                                   s3://callsearch-redacted-<id>/redacted/<batch>.jsonl + .manifest.json
                                        │ scripts/pull_redacted.py (verifies every sha256)
                                        ▼
                                   data/redacted/ ──► scripts/build_index.py
```

- **Infrastructure** (`infra/terraform/`, region `us-west-2`): two private buckets with public access blocked and TLS-only policies. The raw bucket uses KMS encryption (the AWS-managed `aws/s3` key) and expires objects after 7 days; the redacted bucket uses SSE-S3. Also an ECR repository, the Lambda and its IAM role, the S3 trigger, a CloudWatch log group (14 days) and a monthly budget alarm by email.
- **Least privilege:** the Lambda can read `incoming/*` from the raw bucket and write `redacted/*` to the redacted bucket, and nothing else.
- **What leaves your machine:** `scripts/split_calls.py` strips labels and distractors, so only what a real transcript would contain is uploaded. The Docker build context is an allow-list (`.dockerignore`): `pyproject.toml`, `src/` and `models/ner_distilbert/` only, never data or `.env`.
- **Same code as local:** the handler calls the same `redact_calls()` as `callsearch.redact.export`. Logs carry keys and counts, never transcript text.
- **Image:** `public.ecr.aws/lambda/python:3.12` with CPU-only torch and the fine-tuned model baked in (no downloads at run time). It is a container image because torch and the model are far over Lambda's zip limit.

**Deploy** (PowerShell, from the repo root, with the AWS CLI logged in and Docker running):

```powershell
cd infra/terraform
copy terraform.tfvars.example terraform.tfvars        # set alert_email
terraform init; terraform validate
terraform apply                                       # 1st pass: buckets, ECR, IAM, budget (no Lambda yet)
$ECR = terraform output -raw ecr_repository_url
cd ../..
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR.Split("/")[0]
docker build -f Dockerfile.lambda --platform linux/amd64 --provenance=false -t "${ECR}:v1" .
docker push "${ECR}:v1"
cd infra/terraform; terraform apply -var image_tag=v1; cd ../..   # 2nd pass: Lambda + trigger
```

**Run:**

```powershell
python scripts/split_calls.py --per-file 25
aws s3 cp data/upload "s3://$(terraform -chdir=infra/terraform output -raw raw_bucket)/incoming/" --recursive
aws logs tail /aws/lambda/callsearch-redact --follow
python scripts/pull_redacted.py --bucket "$(terraform -chdir=infra/terraform output -raw redacted_bucket)"
python scripts/build_index.py; python -m callsearch.rag.guard --allow-person-leaks
```

**Deployed run** (us-west-2, 3,008 MB, 16 batches of 25 calls): all 400 calls were redacted, with the batches running in parallel and finishing within about 30 seconds of each other. One batch (311 turns) took 147 s on a cold start and used 943 MB. The pulled output is byte-identical to the local `callsearch.redact.export --detector pipeline_hf` run (same sha256; 0 of 5,186 turns differ), so the cloud and local pipelines redact the same way.

**Cost:** S3, Lambda and the AWS-managed KMS key cost cents at this scale. The main standing cost is ECR storage for the image (roughly 2 GB). The budget alarm emails at 80% of $5. `terraform destroy` removes everything, including both buckets and their contents.

## Data and results layout

| Path | Contents | In git |
|---|---|---|
| `data/raw/` | Generated calls with PII, plus downloaded MultiWOZ | No (gitignored) |
| `data/redacted/` | Redacted transcripts plus `manifest.json`. The index builder reads from here only. | No (rebuilt by `callsearch.redact.export`) |
| `data/index/` | Chroma index built from redacted text | No (binary; rebuilt by `scripts/build_index.py`) |
| `models/` | Trained models | No (gitignored) |
| `.cache/llm/` | Cached LLM replies | No (gitignored) |
| `results/` | Eval and training metrics only (counts, rates, type names), never transcript text. A test enforces this. | Yes |
| `results/v1/` | Archived v1 baseline. Its dev set was carved out of train and shared templates and phrasing with it (dev F1 1.000 vs test 0.884), so it is not comparable to current results. | Yes |

## Third-party data and licenses

- **MultiWOZ 2.2**, from [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz). MIT License, Copyright (c) 2019 Paweł Budzianowski. It is downloaded on first run to `data/raw/multiwoz/` (`train/`, `dev/` and `test/dialogues_001.json`) and not redistributed here.
