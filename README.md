# DIN Recsys — Deep Interest Network CTR recommender

An end-to-end click-through-rate recommender on the Taobao display-ad dataset, built
to demonstrate a full ML serving path across languages: a **Deep Interest Network**
trained in PyTorch, exported to **ONNX**, served from a **Go** inference service, and
explored through a **React/TypeScript** console.

```
PyTorch (train) ──► ONNX (export) ──► Go (serve) ──► React (explore)
```

The interesting engineering is the _seam_: rather than serve the model from Python,
the trained graph is exported to ONNX and run from Go via `onnxruntime`. Go owns
feature assembly — vocabulary lookup, OOV fallback, behavior-sequence padding — and
runs the forward pass. A single feature-encoding contract (`feature_spec.json` +
`vocabs/`) is consumed _identically_ by the Python trainer and the Go server, which
is what prevents train/serve skew.

## Why DIN

The dataset's signal lives in user behavior sequences (`cate_his`, `brand_his`,
`btag_his`). DIN's target-attention pooling weights each historical item by its
relevance to the candidate ad, so a user's "interest" is computed _per candidate_
rather than as a flat average. Categories and brands share embedding tables between
the candidate ad and the history, which is what lets the attention dot-products live
in one space.

## Architecture

```
recsys/
  prep/
    prepare_data.py     raw → encoded tensors + feature_spec.json + vocabs/
    make_catalog.py     sample real users/ads → catalog.json
  model/
    din.py              DIN definition + ONNX export
    train.py            training loop → din.pt → din.onnx
  server/               Go service: /score, /rank, /catalog, /health
    encoder.go          request → tensors (mirrors prepare_data.py exactly)
    runner.go           onnxruntime batched inference
    main.go             HTTP layer + server-side catalog
  web/                  React + TypeScript inference console (Vite)
  build/                generated artifacts (gitignored) — the Python↔Go seam
  raw/                  downloaded dataset (gitignored)
```

The `build/` directory is the contract boundary. Everything in it is generated:

| file                      | produced by     | consumed by                 |
| ------------------------- | --------------- | --------------------------- |
| `feature_spec.json`       | prepare_data.py | train.py **and** server     |
| `vocabs/*.json`           | prepare_data.py | server (encode requests)    |
| `train.npz` / `valid.npz` | prepare_data.py | train.py                    |
| `din.onnx`                | train.py        | server                      |
| `din.onnx.json`           | train.py        | server (input tensor order) |
| `catalog.json`            | make_catalog.py | server                      |

## Setup

### 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
hf download reczoo/TaobaoAd_x1 --repo-type dataset --local-dir ./raw
# note the parquet path it writes, e.g. ./raw/train.parquet
```

## Run order

```bash
# (a) prep — parse sequences, build vocabs, encode. Start with a subset.
python prep/prepare_data.py --raw ./raw/train.parquet --out ./build --max-rows 2000000

# (b) train — DIN → best checkpoint → ONNX export
python model/train.py --build ./build --epochs 3 --batch 4096

# (c) catalog — sample real users/ads for the demo
python prep/make_catalog.py --raw ./raw/train.parquet --out ./build --n-users 40 --n-ads 200

# (d) serve — Go inference server (see ONNX Runtime note below first)
cd server
go run . --build ../build --onnx ../build/din.onnx \
         --ort /runtime/libonnxruntime.dylib --catalog ../build/catalog.json

# (e) web — React console
cd web && npm install && npm run dev   # http://localhost:5173, proxies API to :8080
```

## ONNX Runtime shared library (the one non-obvious step)

`onnxruntime_go` does **not** bundle the ONNX Runtime engine — it calls into the
native shared library at runtime, which you download separately from the
[ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases).

1. Download the build for your platform (the `onnxruntime-<os>-<arch>-<ver>` archive).
2. Extract it; the library is under `lib/`:
   - Linux: `libonnxruntime.so`
   - macOS: `libonnxruntime.dylib`
   - Windows: `onnxruntime.dll`
3. Pass its path to the server via `--ort`, or set it in code via
   `ort.SetSharedLibraryPath(...)`.

The Go module dep itself is fetched normally:

```bash
cd server && go mod tidy
```

## Export note (onnxscript / dynamo)

On PyTorch ≥ 2.5 the ONNX exporter may route through the dynamo path, which needs
the `onnxscript` package and rejects the legacy `dynamic_axes` argument. This repo
pins the **legacy TorchScript exporter** by passing `dynamo=False` in
`din.py:export_onnx`, which is deterministic across versions and handles the model's
`masked_fill(-inf)` + empty-sequence guard cleanly. `onnxscript` is still listed in
`requirements.txt` so the dynamo path remains available if you want to experiment.

After export, validate the graph and confirm input names match the sidecar:

```python
import onnx
m = onnx.load("build/din.onnx")
onnx.checker.check_model(m)
print([i.name for i in m.graph.input])   # must match din.onnx.json input_order
```

## Endpoints

| method | path       | body                         | returns                                            |
| ------ | ---------- | ---------------------------- | -------------------------------------------------- |
| POST   | `/score`   | `{user_id, ad_id}`           | `{pctr}` for one pair                              |
| POST   | `/rank`    | `{user_id, ad_ids[], top_n}` | top-N ads by pCTR (empty `ad_ids` = whole catalog) |
| GET    | `/catalog` | —                            | sampled users + ads                                |
| GET    | `/health`  | —                            | `{status:"ok"}`                                    |

`/rank` scores the entire candidate set in a single batched forward pass.

## Notes & honest caveats

- **Metrics.** AUC on a few-million-row subset lands roughly in the 0.62–0.64 range;
  full-set numbers in the DIN literature are higher. Use the offline AUC from
  `train.py` as the figure of record — there is no live traffic, so don't read
  ranking numbers off the UI as performance metrics.
- **Train/valid split** is random, not temporal (the dataset exposes no clean
  timestamp), which slightly inflates offline metrics versus a chronological holdout.
- **`userid` is dropped** — with ~1.14M users it's mostly noise, and in DIN the
  behavior sequence _is_ the user representation.
- **Cold start** falls out for free: ids unseen at training (or below the frequency
  floor) resolve to the out-of-vocabulary index at serve time, same as in training.
- **btag labels** (view/cart/fav/buy) in the UI are a cosmetic guess; the model only
  sees the underlying token.
- **Scaling to the full 25M rows** later means swapping `np.savez` for sharded parquet
  / a streaming dataset — but the feature contract does not change.
