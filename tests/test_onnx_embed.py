"""The ONNX query embedder the hosted API uses instead of sentence-transformers (no torch)."""

import json
import sys
from pathlib import Path

import pytest

from callsearch.rag.embed import OnnxEmbedder, SentenceTransformerEmbedder, default_embedder

ROOT = Path(__file__).resolve().parents[1]


def onnx_dir(tmp_path, name="bge-small"):
    d = tmp_path / "onnx"
    d.mkdir()
    (d / "embedder.json").write_text(json.dumps({"name": name, "query_prefix": "q: ", "inputs": ["input_ids"],
                                                 "max_length": 8}))
    return d


def test_default_embedder_uses_onnx_export_only_for_its_own_model(tmp_path, monkeypatch):
    d = onnx_dir(tmp_path)
    monkeypatch.setenv("CALLSEARCH_ONNX_DIR", str(d))
    assert isinstance(default_embedder("bge-small"), OnnxEmbedder)
    assert isinstance(default_embedder("minilm"), SentenceTransformerEmbedder)  # an export of another model
    monkeypatch.setenv("CALLSEARCH_ONNX_DIR", str(tmp_path / "missing"))
    assert isinstance(default_embedder("bge-small"), SentenceTransformerEmbedder)
    monkeypatch.delenv("CALLSEARCH_ONNX_DIR")
    assert isinstance(default_embedder("bge-small"), SentenceTransformerEmbedder)


@pytest.mark.slow
def test_export_matches_sentence_transformers(tmp_path):
    """Tiny random BERT: export, then compare queries of several lengths (truncation included)."""
    pytest.importorskip("onnxscript")
    np = pytest.importorskip("numpy")
    from sentence_transformers import SentenceTransformer, models as stm
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = "i am looking for a hotel in cambridge when does the train to london leave ok refund".split()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {t: i for i, t in enumerate(specials + sorted(set(words)) + [":", "represent", "this", "sentence",
                                                                        "searching", "relevant", "passages",
                                                                        "double", "charge"])}
    tk = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    hf = tmp_path / "hf"
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64)).save_pretrained(hf)
    BertTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                      sep_token="[SEP]", mask_token="[MASK]").save_pretrained(hf)
    st = SentenceTransformer(modules=[stm.Transformer(str(hf), max_seq_length=16),
                                      stm.Pooling(32, pooling_mode="cls"), stm.Normalize()])
    st.save(str(tmp_path / "st"))

    sys.path.insert(0, str(ROOT / "scripts"))
    from export_onnx import export

    out = tmp_path / "onnx"
    assert export("bge-small", out, source=str(tmp_path / "st")) < 1e-4
    emb = OnnxEmbedder(out)
    for text in ["ok", "hotel " * 30, "when does the train to london leave"]:
        want = st.encode([emb.query_prefix + text], normalize_embeddings=True)[0]
        assert np.abs(want - np.array(emb.embed_query(text))).max() < 1e-4
