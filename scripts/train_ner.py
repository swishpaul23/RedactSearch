""" Train a PERSON-only spaCy NER model on the TRAIN split and save it to models/.

python scripts/train_ner.py --base sm --epochs 30 --dropout 0.2 --seed 42 --out models/ner_spacy

Trains on the train split and picks the best epoch / early-stops on the dev split, which
uses different templates, phrasing and name lists. The test split is never read here.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, fix_random_seed, minibatch

from callsearch.data.generator import read_jsonl
from callsearch.redact.ner import LABEL, build_examples

SM_DROP = ["tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"]


def build_nlp(base: str):
    """Pipeline with a fresh, untrained NER component that knows only the PERSON label."""
    if base == "blank":
        # Blank English: tokenizer only. The NER gets its own randomly initialised tok2vec.
        nlp = spacy.blank("en")
        nlp.add_pipe("ner")
    elif base == "sm":
        # Keep only en_core_web_sm's shared, pretrained tok2vec (token embeddings + encoder)
        # and drop everything else, including its pretrained NER.
        nlp = spacy.load("en_core_web_sm", exclude=SM_DROP)
        width = nlp.get_pipe("tok2vec").model.get_dim("nO")
        # The new NER "listens" to that tok2vec instead of owning one, so it starts from
        # pretrained token features and fine-tunes them as it trains.
        nlp.add_pipe("ner", config={"model": {
            "@architectures": "spacy.TransitionBasedParser.v2", "state_type": "ner",
            "extra_state_tokens": False, "hidden_width": 64, "maxout_pieces": 2, "use_upper": True, "nO": None,
            "tok2vec": {"@architectures": "spacy.Tok2VecListener.v1", "width": width, "upstream": "tok2vec"},
        }})
    else:
        raise ValueError(f"unknown base {base!r}")
    nlp.get_pipe("ner").add_label(LABEL)
    return nlp


def initialize(nlp, base: str, examples: list[Example]):
    """Initialise weights for the new NER only, and return an optimizer."""
    if base == "blank":
        # Nothing is pretrained, so initialise every component.
        return nlp.initialize(lambda: examples)
    # nlp.initialize() would re-initialise the pretrained tok2vec (and needs lookup tables
    # we don't use), so connect the listener and initialise just the new NER instead.
    nlp._link_components()
    nlp.get_pipe("ner").initialize(lambda: examples, nlp=nlp)
    return nlp.resume_training()


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def train(train_calls: list[dict], dev_calls: list[dict], base: str = "sm", epochs: int = 30,
          dropout: float = 0.2, seed: int = 42, out: Path = Path("models/ner_spacy"), patience: int = 5,
          verbose: bool = True) -> dict:
    if any(c["split"] != "train" for c in train_calls):
        raise ValueError("train_calls must all come from the train split")
    if any(c["split"] != "dev" for c in dev_calls):
        raise ValueError("dev_calls must all come from the dev split")
    fix_random_seed(seed)
    rng = random.Random(seed)

    nlp = build_nlp(base)
    train_ex, n_train_labels, train_failed = build_examples(nlp, train_calls)
    dev_ex, n_dev_labels, dev_failed = build_examples(nlp, dev_calls)
    optimizer = initialize(nlp, base, train_ex)

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # 1. Shuffle so each epoch sees the turns in a different order.
        rng.shuffle(train_ex)
        losses: dict[str, float] = {}
        # 2. Feed small batches that grow from 4 to 32 examples as training settles.
        for batch in minibatch(train_ex, size=compounding(4.0, 32.0, 1.001)):
            # 3. One gradient step: predict, compare with gold spans, backpropagate.
            #    Dropout randomly zeroes activations so the model doesn't memorise turns.
            nlp.update(batch, drop=dropout, sgd=optimizer, losses=losses)

        # 4. Score on the dev calls (exact span match on PERSON).
        scores = nlp.evaluate(dev_ex)
        p, r, f = (float(scores[k] or 0.0) for k in ("ents_p", "ents_r", "ents_f"))
        history.append({"epoch": epoch, "loss": round(float(losses.get("ner", 0.0)), 3),
                        "dev_precision": p, "dev_recall": r, "dev_f1": f})
        if verbose:
            print(f"epoch {epoch:>2}  loss {losses.get('ner', 0.0):8.2f}  dev P {p:.3f}  R {r:.3f}  F1 {f:.3f}")

        # 5. Keep the best model on disk; stop once dev F1 stops improving.
        if f > best_f1:
            best_f1, best_epoch, stale = f, epoch, 0
            nlp.to_disk(out)
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop: no dev F1 gain for {patience} epochs")
                break
    elapsed = time.perf_counter() - t0

    return {
        "base": base,
        "n_train_examples": len(train_ex), "n_dev_examples": len(dev_ex),
        "n_train_calls": len(train_calls), "n_dev_calls": len(dev_calls),
        "n_person_labels": n_train_labels + n_dev_labels,
        "failed_alignments": train_failed + dev_failed,
        "epochs_run": len(history), "best_epoch": best_epoch, "best_dev_f1": best_f1,
        "train_seconds": round(elapsed, 1), "model_size_mb": round(dir_size_mb(out), 2),
        "history": history,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train a PERSON-only spaCy NER model.")
    p.add_argument("--base", choices=["sm", "blank"], default="sm")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("models/ner_spacy"))
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--results", type=Path, help="default: results/train_ner_spacy.json (_blank suffix for blank)")
    args = p.parse_args()

    calls = read_jsonl(args.data)
    train_calls = [c for c in calls if c["split"] == "train"]
    dev_calls = [c for c in calls if c["split"] == "dev"]
    del calls  # the test split is never used here
    summary = train(train_calls, dev_calls, args.base, args.epochs, args.dropout, args.seed, args.out, args.patience)

    results = args.results or Path("results") / f"train_ner_spacy{'' if args.base == 'sm' else '_' + args.base}.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nbest epoch {summary['best_epoch']} (dev F1 {summary['best_dev_f1']:.3f}), "
          f"{summary['train_seconds']}s, model {summary['model_size_mb']} MB -> {args.out}")
    print(f"failed alignments: {summary['failed_alignments']} of {summary['n_person_labels']} PERSON labels")
    print(f"saved {results}")


if __name__ == "__main__":
    main()
