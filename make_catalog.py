"""
make_catalog.py  --  sample real users & ads from TaobaoAd_x1 -> catalog.json

The Go server merges a User and an Ad into one model request, so the column split
here MUST match prepare_data.py:
    user side : userid + USER_SPARSE + the three behavior sequences
    ad side   : adgroup_id + AD_SPARSE(rest) + CTX_SPARSE + price(dense)

We sample DISTINCT users and DISTINCT ads (not raw impression rows) so the demo
shows variety. Behavior sequences are user-level in this dataset, so we take them
from any one impression row for that user. Tokens are kept as raw strings -- the
server looks them up in the same vocabs the model trained on, so an id that fell
below the frequency floor simply resolves to OOV at serve time (realistic cold start).

There are no human-readable product names in the source, so ad "title" is a synthetic
label built from category + brand + price. It's only a UI affordance, never a feature.

Usage:
    python make_catalog.py --raw ./raw/train.parquet --out ./build \
        --n-users 40 --n-ads 200 --seed 42
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

# Column groups -- mirror prepare_data.py exactly.
USER_SPARSE = [
    "cms_segid", "cms_group_id", "final_gender_code", "age_level",
    "pvalue_level", "shopping_level", "occupation", "new_user_class_level",
]
AD_SPARSE = ["adgroup_id", "cate_id", "campaign_id", "customer", "brand"]
CTX_SPARSE = ["pid", "btag"]
DENSE = ["price"]
SEQ_COLS = ["cate_his", "brand_his", "btag_his"]
USER_KEY, AD_KEY = "userid", "adgroup_id"

RAW_COLS = (
    [USER_KEY] + USER_SPARSE + AD_SPARSE + CTX_SPARSE + DENSE + SEQ_COLS
)

# Context defaults when a sampled ad needs pid/btag and we want a stable display.
CTX_DEFAULT = {"pid": "430548_1007", "btag": "0"}


def parse_seq(s):
    if s is None or s == "" or s == "null" or (isinstance(s, float) and np.isnan(s)):
        return []
    return s.split("^")


def sample_users(df, n, rng):
    """One catalog User per distinct userid, with that user's history + profile."""
    # first impression row per user is enough: profile + history are user-level
    first = df.drop_duplicates(subset=[USER_KEY], keep="first")
    if len(first) > n:
        first = first.sample(n=n, random_state=int(rng.integers(1 << 31)))

    users = []
    for _, row in first.iterrows():
        seqs = {c: parse_seq(row[c]) for c in SEQ_COLS}
        # skip users with no history at all -- nothing for attention to chew on
        if not any(seqs.values()):
            continue
        users.append({
            "id": str(row[USER_KEY]),
            "sparse": {c: str(row[c]) for c in USER_SPARSE},
            "seqs": seqs,
        })
    return users


def synth_title(row):
    cate, brand = row.get("cate_id", "?"), row.get("brand", "?")
    price = row.get("price", 0.0)
    return f"Cat {cate} · Brand {brand} · ${float(price):.2f}"


def sample_ads(df, n, rng):
    """One catalog Ad per distinct adgroup_id."""
    first = df.drop_duplicates(subset=[AD_KEY], keep="first")
    if len(first) > n:
        first = first.sample(n=n, random_state=int(rng.integers(1 << 31)))

    ads = []
    for _, row in first.iterrows():
        sparse = {c: str(row[c]) for c in AD_SPARSE}
        # fold context fields into the ad side (server merges user∪ad for all sparse)
        for c in CTX_SPARSE:
            val = row[c]
            sparse[c] = CTX_DEFAULT[c] if pd.isna(val) else str(val)
        price = float(row["price"]) if not pd.isna(row["price"]) else 0.0
        ads.append({
            "id": str(row[AD_KEY]),
            "title": synth_title(row),
            "sparse": sparse,
            "dense": {"price": max(0.0, min(1.0, price))},
        })
    return ads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="./build")
    ap.add_argument("--n-users", type=int, default=40)
    ap.add_argument("--n-ads", type=int, default=200)
    ap.add_argument("--sample-rows", type=int, default=500000,
                    help="rows to scan when picking distinct users/ads (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[load] {args.raw}")
    if args.raw.endswith(".csv"):
        df = pd.read_csv(args.raw, usecols=RAW_COLS)
    else:
        df = pd.read_parquet(args.raw, columns=RAW_COLS)
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(n=args.sample_rows, random_state=args.seed)
    print(f"[load] scanning {len(df):,} rows")

    users = sample_users(df, args.n_users, rng)
    ads = sample_ads(df, args.n_ads, rng)
    print(f"[sample] users={len(users)}  ads={len(ads)}")

    catalog = {"users": users, "ads": ads}
    full_path = os.path.join(args.out, "catalog.json")
    with open(full_path, "w") as f:
        json.dump(catalog, f, indent=2)

    # tiny committable sample so the repo runs without the full dataset
    sample = {"users": users[: min(5, len(users))], "ads": ads[: min(20, len(ads))]}
    sample_path = os.path.join(args.out, "catalog.sample.json")
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"[done] wrote {full_path} and {sample_path}")


if __name__ == "__main__":
    main()