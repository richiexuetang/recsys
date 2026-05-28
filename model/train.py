"""
train.py  --  train DIN on the encoded TaobaoAd_x1 tensors, then export to ONNX.

    python train.py --build ./build --epochs 3 --batch 4096

Reads from --build:  train.npz, valid.npz, feature_spec.json
Writes to  --build:  din.pt (best checkpoint), din.onnx, din.onnx.json (sidecar)

Eval metric is AUC (the metric that matters for ranking) plus logloss. A DIN on a
few-million-row subset of this data lands roughly in the 0.62-0.64 AUC range; the
public DIN paper numbers on the full set are higher. Use this loop to confirm the
pipeline learns, then scale rows / tune before reading too much into the absolute.

Deps: torch, numpy, scikit-learn
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

from din import DIN, export_onnx, load_spec


class NpzDataset(Dataset):
    """Wraps an encoded split. Splits the flat npz back into the dict groups DIN wants."""

    def __init__(self, npz_path, spec):
        d = np.load(npz_path)
        self.spec = spec
        self.sparse = {f["column"]: d[f"sparse__{f['column']}"]
                       for f in spec["sparse_features"]}
        self.dense = d["dense"]
        self.seqs = {s["column"]: d[f"seq__{s['column']}"]
                     for s in spec["sequence_features"]}
        self.seqlens = {s["column"]: d[f"seqlen__{s['column']}"]
                        for s in spec["sequence_features"]}
        self.label = d["label"]
        self.n = len(self.label)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i  # collate pulls rows by index -> fewer python-level tensor ops

    def collate(self, idx):
        idx = np.asarray(idx)
        sparse = {k: torch.from_numpy(v[idx].astype(np.int64))
                  for k, v in self.sparse.items()}
        dense = torch.from_numpy(self.dense[idx])
        seqs = {k: torch.from_numpy(v[idx].astype(np.int64))
                for k, v in self.seqs.items()}
        seqlens = {k: torch.from_numpy(v[idx].astype(np.int64))
                   for k, v in self.seqlens.items()}
        label = torch.from_numpy(self.label[idx])
        return sparse, dense, seqs, seqlens, label


def to_device(sparse, dense, seqs, seqlens, dev):
    return (
        {k: v.to(dev) for k, v in sparse.items()},
        dense.to(dev),
        {k: v.to(dev) for k, v in seqs.items()},
        {k: v.to(dev) for k, v in seqlens.items()},
    )


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ps, ys = [], []
    for sparse, dense, seqs, seqlens, label in loader:
        s, d, sq, sl = to_device(sparse, dense, seqs, seqlens, dev)
        p = torch.sigmoid(model(s, d, sq, sl)).cpu().numpy()
        ps.append(p); ys.append(label.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    return roc_auc_score(y, p), log_loss(y, p, labels=[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="./build")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=16)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = load_spec(os.path.join(args.build, "feature_spec.json"))

    train_ds = NpzDataset(os.path.join(args.build, "train.npz"), spec)
    valid_ds = NpzDataset(os.path.join(args.build, "valid.npz"), spec)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=train_ds.collate, num_workers=2, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch, shuffle=False,
                          collate_fn=valid_ds.collate, num_workers=2)

    model = DIN(spec, emb_dim=args.emb_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print(f"[train] device={dev}  params={sum(p.numel() for p in model.parameters()):,}")

    best_auc, ckpt = 0.0, os.path.join(args.build, "din.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (sparse, dense, seqs, seqlens, label) in enumerate(train_dl, 1):
            s, d, sq, sl = to_device(sparse, dense, seqs, seqlens, dev)
            opt.zero_grad()
            loss = loss_fn(model(s, d, sq, sl), label.to(dev))
            loss.backward()
            opt.step()
            running += loss.item() * len(label); seen += len(label)
            if step % 200 == 0:
                print(f"  e{epoch} step {step}  loss={running/seen:.4f}")
        auc, ll = evaluate(model, valid_dl, dev)
        print(f"[eval] epoch {epoch}  AUC={auc:.4f}  logloss={ll:.4f}")
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), ckpt)
            print(f"       new best -> {ckpt}")

    # export the best checkpoint for Go
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.to("cpu")
    export_onnx(model,
                os.path.join(args.build, "din.onnx"),
                os.path.join(args.build, "din.onnx.json"))
    print(f"[done] best AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()