"""
prepare_data.py  --  TaobaoAd_x1  ->  DIN-ready encoded tensors + a shared feature spec.

Pipeline:
    raw (parquet/csv)
      -> parse caret-separated behavior sequences
      -> build vocabs on the TRAIN split (frequency-thresholded, shared where columns
         occupy the same id space, e.g. candidate cate_id and historical cate_his)
      -> encode ids, pad sequences to a fixed length
      -> write train.npz / valid.npz + feature_spec.json + vocabs/<name>.json

feature_spec.json is the single source of truth. The PyTorch trainer reads it to size
embedding tables; the Go server reads it (plus vocabs/) to encode requests identically.
If the encoding contract lived in two places it would drift -- so it lives here only.

Get the dataset first (one of):
    pip install huggingface_hub
    hf download reczoo/TaobaoAd_x1 --repo-type dataset --local-dir ./raw
        # then point --raw at the downloaded parquet, e.g. ./raw/<file>.parquet

Usage:
    python prepare_data.py --raw ./raw/train.parquet --out ./build --max-rows 2000000
    # drop --max-rows for the full 25M (see memory note at bottom)

Deps: pandas, numpy, pyarrow
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Reserved indices present in every vocab.
PAD, OOV = 0, 1

# --- Feature contract -------------------------------------------------------
# Single-value categorical columns, grouped by intent. Cardinalities (from the
# dataset card) noted so the freq cutoffs below make sense.
USER_SPARSE = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
]  # all small (<=96), kept whole
AD_SPARSE = ["adgroup_id", "cate_id", "campaign_id", "customer", "brand"]  # large
CTX_SPARSE = ["pid", "btag"]  # tiny
DENSE = ["price"]             # already ~[0,1] in the source
LABEL = "clk"

# column -> vocab name. Columns mapped to the SAME vocab share one embedding table.
# The candidate ad's cate_id lives in the same space as the historical cate_his,
# so both map to "cate"; likewise brand. That sharing is what lets DIN compute
# attention between the target item and each behavior.
VOCAB_OF = {c: c for c in USER_SPARSE + CTX_SPARSE}          # self-named vocabs
VOCAB_OF.update({
    "adgroup_id": "adgroup", "campaign_id": "campaign", "customer": "customer",
    "cate_id": "cate", "brand": "brand",
})

# behavior sequence column -> (shared vocab name, the single-value column it attends to)
SEQ = {
    "cate_his":  ("cate",  "cate_id"),
    "brand_his": ("brand", "brand"),
    "btag_his":  ("btag",  None),   # behavior type (click/cart/fav/buy); no target item
}
VOCAB_OF["btag"] = "btag"  # btag appears both as context and as a sequence -> shared

# High-cardinality vocabs get a frequency floor so embedding tables stay tractable.
# Anything below the floor (or unseen at serving) collapses to OOV.
MIN_FREQ = {"adgroup": 5, "campaign": 5, "customer": 5, "brand": 5, "cate": 1}
DEFAULT_MIN_FREQ = 1

RAW_COLS = USER_SPARSE + AD_SPARSE + CTX_SPARSE + DENSE + list(SEQ) + [LABEL]


def parse_seq(s):
    """Caret-separated id string -> list[str]. Nulls/empties -> []."""
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)) or s == "null":
        return []
    return s.split("^")


def build_vocab(counts: Counter, min_freq: int) -> dict:
    """Frequency-thresholded vocab. PAD=0, OOV=1, then tokens by descending freq."""
    items = [(t, c) for t, c in counts.items() if c >= min_freq and t != ""]
    items.sort(key=lambda x: (-x[1], x[0]))
    vocab = {"<pad>": PAD, "<oov>": OOV}
    for i, (tok, _) in enumerate(items, start=2):
        vocab[tok] = i
    return vocab


def gather_counts(train: pd.DataFrame) -> dict:
    """Token counts per vocab name, pulling from every column that shares it."""
    counts = defaultdict(Counter)
    for col, vname in VOCAB_OF.items():
        counts[vname].update(train[col].astype(str).tolist())
    for scol, (vname, _) in SEQ.items():
        for lst in train[scol].map(parse_seq):
            counts[vname].update(lst)
    return counts


def encode_split(df: pd.DataFrame, vocabs: dict, max_len: int) -> dict:
    """DataFrame -> dict of numpy arrays ready for np.savez."""
    n = len(df)
    out = {}

    # single-value sparse features
    for col, vname in VOCAB_OF.items():
        v = vocabs[vname]
        out[f"sparse__{col}"] = (
            df[col].astype(str).map(lambda t: v.get(t, OOV)).to_numpy(np.int32)
        )

    # dense features
    out["dense"] = df[DENSE].fillna(0.0).to_numpy(np.float32).clip(0.0, 1.0)

    # behavior sequences: keep the most-recent max_len, right-pad with PAD
    for scol, (vname, _) in SEQ.items():
        v = vocabs[vname]
        ids = np.zeros((n, max_len), dtype=np.int32)
        lens = np.zeros(n, dtype=np.int32)
        for i, raw in enumerate(df[scol].map(parse_seq)):
            seq = [v.get(t, OOV) for t in raw][-max_len:]
            ids[i, : len(seq)] = seq
            lens[i] = len(seq)
        out[f"seq__{scol}"] = ids
        out[f"seqlen__{scol}"] = lens

    out["label"] = df[LABEL].to_numpy(np.float32)
    return out


def make_spec(vocabs: dict, max_len: int) -> dict:
    """The contract Python and Go both consume."""
    return {
        "max_seq_len": max_len,
        "pad_index": PAD,
        "oov_index": OOV,
        "label": LABEL,
        "dense": DENSE,
        "vocab_sizes": {name: len(v) for name, v in vocabs.items()},
        "sparse_features": [
            {"column": c, "vocab": VOCAB_OF[c]}
            for c in USER_SPARSE + AD_SPARSE + CTX_SPARSE
        ],
        "sequence_features": [
            {"column": c, "vocab": v, "target_column": tgt}
            for c, (v, tgt) in SEQ.items()
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="path to .parquet or .csv")
    ap.add_argument("--out", default="./build")
    ap.add_argument("--max-seq-len", type=int, default=50)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "vocabs"), exist_ok=True)

    print(f"[load] {args.raw}")
    if args.raw.endswith(".csv"):
        df = pd.read_csv(args.raw, usecols=RAW_COLS)
    else:
        df = pd.read_parquet(args.raw, columns=RAW_COLS)

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed)
    df = df.reset_index(drop=True)
    print(f"[load] {len(df):,} rows  |  CTR={df[LABEL].mean():.4f}")

    # split
    rng = np.random.default_rng(args.seed)
    is_valid = rng.random(len(df)) < args.valid_frac
    train, valid = df[~is_valid], df[is_valid]
    print(f"[split] train={len(train):,}  valid={len(valid):,}")

    # vocabs (train only -> no leakage)
    print("[vocab] building...")
    counts = gather_counts(train)
    vocabs = {
        name: build_vocab(cnt, MIN_FREQ.get(name, DEFAULT_MIN_FREQ))
        for name, cnt in counts.items()
    }
    for name, v in sorted(vocabs.items()):
        print(f"        {name:10s} -> {len(v):>8,} tokens")

    # encode + write
    print("[encode] train")
    np.savez(os.path.join(args.out, "train.npz"),
             **encode_split(train, vocabs, args.max_seq_len))
    print("[encode] valid")
    np.savez(os.path.join(args.out, "valid.npz"),
             **encode_split(valid, vocabs, args.max_seq_len))

    for name, v in vocabs.items():
        with open(os.path.join(args.out, "vocabs", f"{name}.json"), "w") as f:
            json.dump(v, f)

    spec = make_spec(vocabs, args.max_seq_len)
    with open(os.path.join(args.out, "feature_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    print(f"[done] wrote train.npz, valid.npz, feature_spec.json, vocabs/ -> {args.out}")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# Memory note: encoded sequences are [N, max_seq_len] int32 -- ~0.2 GB per 1M
# rows per sequence (x3 sequences). The full 25M set in one np.savez is large;
# start with --max-rows 2000000 to get the loop working end-to-end, then scale.
# Going full-set later means switching savez -> sharded parquet / a streaming
# Dataset, but the feature contract above does not change.
# ----------------------------------------------------------------------------