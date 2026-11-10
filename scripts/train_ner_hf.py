"""Fine-tune a Hugging Face token classifier (DistilBERT by default) for PERSON and save it to models/.

python scripts/train_ner_hf.py --epochs 6 --lr 5e-5 --batch-size 16 --seed 42 --out models/ner_distilbert

Same protocol as scripts/train_ner.py (spaCy), so the two are comparable: train on the
train split, pick the best epoch / early-stop on dev (exact-match PERSON F1), never read test.
The first run downloads the base model (~260 MB) from the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from callsearch.data.generator import read_jsonl
from callsearch.redact.ner_hf import (DEFAULT_BASE_MODEL, IGNORE, LABEL, MAX_LENGTH, TAGS, align_labels,
                                      predict_spans)


def turns_with_spans(calls: list[dict]) -> list[tuple[str, list[tuple[int, int]]]]:
    """(turn text, gold PERSON char spans) for every turn, including turns with no names."""
    out = []
    for call in calls:
        for turn in call["turns"]:
            spans = [(l["start"], l["end"]) for l in call["labels"]
                     if l["turn_id"] == turn["turn_id"] and l["type"] == LABEL]
            out.append((turn["text"], spans))
    return out


def encode(tokenizer, examples, max_length: int = MAX_LENGTH):
    """Tokenize and align BIO labels. Returns (features, n_labels, n_failed, n_truncated_turns)."""
    feats, n_labels, n_failed, n_trunc = [], 0, 0, 0
    for text, spans in examples:
        enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")]
        tags, missed = align_labels(offsets, spans)
        n_labels += len(spans)
        n_failed += missed  # a span cut off by truncation also counts here
        n_trunc += len(enc["input_ids"]) == max_length
        feats.append({**{k: v for k, v in enc.items() if k in tokenizer.model_input_names}, "labels": tags})
    return feats, n_labels, n_failed, n_trunc


def exact_prf(pred: list[list[tuple[int, int]]], gold: list[list[tuple[int, int]]]) -> tuple[float, float, float]:
    """Exact span match, the same scoring spaCy's nlp.evaluate() uses for ents_p/r/f."""
    tp = sum(len(set(p) & set(g)) for p, g in zip(pred, gold))
    n_pred, n_gold = sum(map(len, pred)), sum(map(len, gold))
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def train(train_calls: list[dict], dev_calls: list[dict], base_model: str = DEFAULT_BASE_MODEL,
          epochs: int = 6, lr: float = 5e-5, batch_size: int = 16, weight_decay: float = 0.01,
          warmup: float = 0.1, seed: int = 42, out: Path = Path("models/ner_distilbert"),
          patience: int = 2, max_length: int = MAX_LENGTH, device: str | None = None,
          verbose: bool = True) -> dict:
    if any(c["split"] != "train" for c in train_calls):
        raise ValueError("train_calls must all come from the train split")
    if any(c["split"] != "dev" for c in dev_calls):
        raise ValueError("dev_calls must all come from the dev split")

    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              get_linear_schedule_with_warmup, set_seed)

    set_seed(seed)
    rng = random.Random(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if not tokenizer.is_fast:
        raise ValueError(f"{base_model} has no fast tokenizer; offset mapping needs one")
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(TAGS), id2label=dict(enumerate(TAGS)),
        label2id={t: i for i, t in enumerate(TAGS)}).to(device)

    train_ex, dev_ex = turns_with_spans(train_calls), turns_with_spans(dev_calls)
    train_feats, n_train_labels, train_failed, train_trunc = encode(tokenizer, train_ex, max_length)
    _, n_dev_labels, dev_failed, dev_trunc = encode(tokenizer, dev_ex, max_length)
    dev_texts, dev_gold = [t for t, _ in dev_ex], [sorted(s) for _, s in dev_ex]

    collate = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE)
    steps_per_epoch = (len(train_feats) + batch_size - 1) // batch_size
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup * total), total)

    history, best_f1, best_epoch, stale = [], -1.0, 0, 0
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # 1. Shuffle, then batch with padding to the longest turn in each batch.
        rng.shuffle(train_feats)
        model.train()
        loss_sum = 0.0
        for batch in DataLoader(train_feats, batch_size=batch_size, collate_fn=collate):
            # 2. One gradient step: cross-entropy on every labeled subword (IGNORE skipped).
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_sum += loss.item()

        # 3. Score on dev: exact PERSON span match, as in the spaCy script.
        pred = predict_spans(model, tokenizer, dev_texts, batch_size=64, max_length=max_length)
        p, r, f = exact_prf(pred, dev_gold)
        avg_loss = loss_sum / steps_per_epoch
        history.append({"epoch": epoch, "loss": round(avg_loss, 4),
                        "dev_precision": p, "dev_recall": r, "dev_f1": f})
        if verbose:
            print(f"epoch {epoch:>2}  loss {avg_loss:7.4f}  dev P {p:.3f}  R {r:.3f}  F1 {f:.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

        # 4. Keep the best model on disk; stop once dev F1 stops improving.
        if f > best_f1:
            best_f1, best_epoch, stale = f, epoch, 0
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stop: no dev F1 gain for {patience} epochs")
                break
    elapsed = time.perf_counter() - t0

    return {
        "base_model": base_model, "device": device,
        "hyperparams": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "weight_decay": weight_decay,
                        "warmup": warmup, "seed": seed, "patience": patience, "max_length": max_length},
        "n_train_examples": len(train_ex), "n_dev_examples": len(dev_ex),
        "n_train_calls": len(train_calls), "n_dev_calls": len(dev_calls),
        "n_person_labels": n_train_labels + n_dev_labels,
        "failed_alignments": train_failed + dev_failed,
        "truncated_turns": train_trunc + dev_trunc,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history), "best_epoch": best_epoch, "best_dev_f1": best_f1,
        "train_seconds": round(elapsed, 1), "model_size_mb": round(dir_size_mb(out), 2),
        "history": history,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a Hugging Face token classifier for PERSON.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Hub id or local directory")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps with rising learning rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--device", help="default: cuda if available, else cpu")
    p.add_argument("--out", type=Path, default=Path("models/ner_distilbert"))
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--results", type=Path, default=Path("results/train_ner_distilbert.json"))
    args = p.parse_args()

    calls = read_jsonl(args.data)
    train_calls = [c for c in calls if c["split"] == "train"]
    dev_calls = [c for c in calls if c["split"] == "dev"]
    del calls  # the test split is never used here
    summary = train(train_calls, dev_calls, args.base_model, args.epochs, args.lr, args.batch_size,
                    args.weight_decay, args.warmup, args.seed, args.out, args.patience, args.max_length, args.device)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nbest epoch {summary['best_epoch']} (dev F1 {summary['best_dev_f1']:.3f}), "
          f"{summary['train_seconds']}s on {summary['device']}, model {summary['model_size_mb']} MB, "
          f"{summary['n_parameters'] / 1e6:.1f}M params -> {args.out}")
    print(f"failed alignments: {summary['failed_alignments']} of {summary['n_person_labels']} PERSON labels; "
          f"truncated turns: {summary['truncated_turns']}")
    print(f"saved {args.results}")


if __name__ == "__main__":
    main()
