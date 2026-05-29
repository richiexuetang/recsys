"""
din.py  --  Deep Interest Network for TaobaoAd_x1, sized from feature_spec.json.

This version also EXPOSES ATTENTION WEIGHTS as extra ONNX outputs so the serving
layer can explain a prediction: for each attended behavior sequence (cate_his,
brand_his) we emit the post-softmax weight the model placed on each historical item
relative to the candidate ad. Those weights are exactly what makes DIN interpretable
-- they say "this pCTR is high because the user's history is heavy on the candidate's
category."

Outputs of the exported graph:
    pctr            [B]      predicted click probability
    attn__cate_his  [B, L]   attention over the category history
    attn__brand_his [B, L]   attention over the brand history
"""

import json
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """DIN local-activation unit. Returns (pooled_vector, attention_weights)."""

    def __init__(self, dim, hidden=(64, 32)):
        super().__init__()
        layers, in_dim = [], dim * 4
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, keys, mask):
        L = keys.size(1)
        q = query.unsqueeze(1).expand(-1, L, -1)              # [B, L, D]
        feats = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.mlp(feats).squeeze(-1)                  # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        empty = ~mask.any(dim=1, keepdim=True)                # [B, 1]
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(empty.expand_as(weights),
                              torch.zeros_like(weights), weights)   # [B, L]
        pooled = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)   # [B, D]
        return pooled, weights


class DIN(nn.Module):
    def __init__(self, spec: dict, emb_dim=16, mlp_hidden=(200, 80)):
        super().__init__()
        self.spec = spec
        self.emb_dim = emb_dim
        self.max_len = spec["max_seq_len"]
        self.pad = spec["pad_index"]

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, emb_dim, padding_idx=self.pad)
            for name, size in spec["vocab_sizes"].items()
        })

        self.sparse = spec["sparse_features"]
        self.sequences = spec["sequence_features"]
        self.dense_cols = spec["dense"]

        self.attn = nn.ModuleDict({
            s["column"]: AttentionPool(emb_dim)
            for s in self.sequences if s["target_column"]
        })
        # columns that produce an attention output, in a fixed order
        self.attended = [s["column"] for s in self.sequences if s["target_column"]]

        n_sparse = len(self.sparse)
        n_seq = len(self.sequences)
        in_dim = (n_sparse + n_seq) * emb_dim + len(self.dense_cols)

        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.PReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, sparse, dense, seqs, seqlens, return_attn=False):
        """logits [B]; if return_attn, also a dict {col: weights[B, L]}."""
        parts = []
        attn_out = {}

        for f in self.sparse:
            parts.append(self.embeddings[f["vocab"]](sparse[f["column"]]))

        for s in self.sequences:
            col, vocab, tgt = s["column"], s["vocab"], s["target_column"]
            keys = self.embeddings[vocab](seqs[col])
            pos = torch.arange(self.max_len, device=keys.device).unsqueeze(0)
            mask = pos < seqlens[col].unsqueeze(1)
            if tgt and col in self.attn:
                query = self.embeddings[vocab](sparse[tgt])
                pooled, weights = self.attn[col](query, keys, mask)
                parts.append(pooled)
                attn_out[col] = weights
            else:
                m = mask.unsqueeze(-1).float()
                parts.append((keys * m).sum(1) / m.sum(1).clamp(min=1.0))

        parts.append(dense)
        logits = self.head(torch.cat(parts, dim=-1)).squeeze(-1)
        if return_attn:
            return logits, attn_out
        return logits


# --- ONNX export ------------------------------------------------------------

class ExportWrapper(nn.Module):
    """Flattens dict I/O into positional tensors and emits pctr + attention."""

    def __init__(self, model: DIN):
        super().__init__()
        self.m = model
        self.sparse_cols = [f["column"] for f in model.sparse]
        self.seq_cols = [s["column"] for s in model.sequences]
        self.attended = model.attended

    def input_order(self):
        order = [f"sparse__{c}" for c in self.sparse_cols]
        order.append("dense")
        for c in self.seq_cols:
            order += [f"seq__{c}", f"seqlen__{c}"]
        return order

    def output_order(self):
        return ["pctr"] + [f"attn__{c}" for c in self.attended]

    def forward(self, *tensors):
        it = iter(tensors)
        sparse = {c: next(it) for c in self.sparse_cols}
        dense = next(it)
        seqs, seqlens = {}, {}
        for c in self.seq_cols:
            seqs[c] = next(it)
            seqlens[c] = next(it)
        logits, attn = self.m(sparse, dense, seqs, seqlens, return_attn=True)
        return (torch.sigmoid(logits), *[attn[c] for c in self.attended])


def export_onnx(model: DIN, path: str, sidecar_path: str, batch=4):
    model.eval()
    wrap = ExportWrapper(model)
    L = model.max_len

    dummy, dyn = [], {}
    for f in model.sparse:
        name = f"sparse__{f['column']}"
        dummy.append(torch.zeros(batch, dtype=torch.long)); dyn[name] = {0: "batch"}
    dummy.append(torch.zeros(batch, len(model.dense_cols))); dyn["dense"] = {0: "batch"}
    for s in model.sequences:
        c = s["column"]
        dummy.append(torch.zeros(batch, L, dtype=torch.long)); dyn[f"seq__{c}"] = {0: "batch"}
        dummy.append(torch.zeros(batch, dtype=torch.long));    dyn[f"seqlen__{c}"] = {0: "batch"}

    in_names = wrap.input_order()
    out_names = wrap.output_order()
    for o in out_names:
        dyn[o] = {0: "batch"}

    torch.onnx.export(
        wrap, tuple(dummy), path,
        input_names=in_names, output_names=out_names,
        dynamic_axes=dyn, opset_version=14,
        dynamo=False,   # legacy TorchScript exporter: deterministic, no onnxscript dependency
    )
    with open(sidecar_path, "w") as f:
        json.dump({"input_order": in_names, "outputs": out_names}, f, indent=2)
    print(f"[onnx] {path}  (in={len(in_names)} out={len(out_names)})  +  {sidecar_path}")


def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)