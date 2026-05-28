"""
din.py  --  Deep Interest Network for TaobaoAd_x1, sized from feature_spec.json.

The model has three groups of inputs, all produced by prepare_data.py:
  - sparse single-value features (user/ad/context) -> embedding lookup
  - behavior sequences (cate_his, brand_his, btag_his) -> target-attention pooling
  - dense features (price) -> concatenated raw

Target attention is the idea that makes this "interest" rather than a flat average:
for the candidate ad's category, we weight each historical category by how relevant
it is to that candidate, then sum. A user who clicked lots of shoes gets a shoe-heavy
interest vector *when the candidate is a shoe*, and a different vector for a phone.

Embedding tables are SHARED per the spec: cate_id and cate_his index the same table,
which is what lets the attention dot-products live in one space.

ONNX export at the bottom freezes a forward pass Go can run via onnxruntime-go.
"""

import json
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """DIN local-activation unit: pool a behavior sequence toward a query item.

    query:   [B, D]            embedding of the candidate item (e.g. its cate_id)
    keys:    [B, L, D]         embeddings of the L historical items
    mask:    [B, L] bool       True at real positions, False at PAD
    returns: [B, D]            attention-weighted sum of keys
    """

    def __init__(self, dim, hidden=(64, 32)):
        super().__init__()
        # scores from [query, key, query-key, query*key] -> richer interaction
        layers, in_dim = [], dim * 4
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, query, keys, mask):
        L = keys.size(1)
        q = query.unsqueeze(1).expand(-1, L, -1)          # [B, L, D]
        feats = torch.cat([q, keys, q - keys, q * keys], dim=-1)
        scores = self.mlp(feats).squeeze(-1)              # [B, L]
        # mask PAD positions to -inf before softmax so they contribute nothing
        scores = scores.masked_fill(~mask, float("-inf"))
        # a fully-empty sequence would softmax to NaN; guard it
        empty = ~mask.any(dim=1, keepdim=True)            # [B, 1]
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(empty.expand_as(weights),
                              torch.zeros_like(weights), weights)
        return torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # [B, D]


class DIN(nn.Module):
    def __init__(self, spec: dict, emb_dim=16, mlp_hidden=(200, 80)):
        super().__init__()
        self.spec = spec
        self.emb_dim = emb_dim
        self.max_len = spec["max_seq_len"]
        self.pad = spec["pad_index"]

        # one embedding table per vocab; shared columns reuse the same table
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, emb_dim, padding_idx=self.pad)
            for name, size in spec["vocab_sizes"].items()
        })

        self.sparse = spec["sparse_features"]          # [{column, vocab}, ...]
        self.sequences = spec["sequence_features"]     # [{column, vocab, target_column}]
        self.dense_cols = spec["dense"]

        # one attention unit per sequence that has a target item to attend to
        self.attn = nn.ModuleDict({
            s["column"]: AttentionPool(emb_dim)
            for s in self.sequences if s["target_column"]
        })

        # input width to the head:
        #   sparse fields * D  +  one pooled vector per sequence * D  +  dense
        n_sparse = len(self.sparse)
        n_seq = len(self.sequences)
        in_dim = (n_sparse + n_seq) * emb_dim + len(self.dense_cols)

        layers = []
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.PReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, sparse, dense, seqs, seqlens):
        """
        sparse:  dict[column] -> [B] long
        dense:   [B, n_dense] float
        seqs:    dict[column] -> [B, L] long
        seqlens: dict[column] -> [B] long
        returns: [B] logits  (apply sigmoid for pCTR)
        """
        parts = []

        # single-value sparse
        for f in self.sparse:
            parts.append(self.embeddings[f["vocab"]](sparse[f["column"]]))

        # sequences -> pooled interest vectors
        for s in self.sequences:
            col, vocab, tgt = s["column"], s["vocab"], s["target_column"]
            keys = self.embeddings[vocab](seqs[col])              # [B, L, D]
            pos = torch.arange(self.max_len, device=keys.device).unsqueeze(0)
            mask = pos < seqlens[col].unsqueeze(1)                # [B, L]
            if tgt and col in self.attn:
                query = self.embeddings[vocab](sparse[tgt])       # [B, D]
                parts.append(self.attn[col](query, keys, mask))
            else:
                # no target item (e.g. btag_his) -> masked mean pool
                m = mask.unsqueeze(-1).float()
                summed = (keys * m).sum(1)
                cnt = m.sum(1).clamp(min=1.0)
                parts.append(summed / cnt)

        parts.append(dense)
        x = torch.cat(parts, dim=-1)
        return self.head(x).squeeze(-1)


# --- ONNX export ------------------------------------------------------------
# Go's onnxruntime works best with positional tensor inputs, not dicts. We wrap
# the model in a thin module whose forward takes an explicit, ordered list of
# tensors. input_order() is written into the export sidecar so the Go server
# feeds tensors in exactly this sequence.

class ExportWrapper(nn.Module):
    def __init__(self, model: DIN):
        super().__init__()
        self.m = model
        self.sparse_cols = [f["column"] for f in model.sparse]
        self.seq_cols = [s["column"] for s in model.sequences]

    def input_order(self):
        order = [f"sparse__{c}" for c in self.sparse_cols]
        order.append("dense")
        for c in self.seq_cols:
            order += [f"seq__{c}", f"seqlen__{c}"]
        return order

    def forward(self, *tensors):
        it = iter(tensors)
        sparse = {c: next(it) for c in self.sparse_cols}
        dense = next(it)
        seqs, seqlens = {}, {}
        for c in self.seq_cols:
            seqs[c] = next(it)
            seqlens[c] = next(it)
        logits = self.m(sparse, dense, seqs, seqlens)
        return torch.sigmoid(logits)   # export pCTR directly so Go just reads it


def export_onnx(model: DIN, path: str, sidecar_path: str, batch=4):
    """Trace ExportWrapper to ONNX with a dynamic batch axis; write input order."""
    model.eval()
    wrap = ExportWrapper(model)
    L = model.max_len

    dummy, dyn = [], {}
    for f in model.sparse:
        name = f"sparse__{f['column']}"
        dummy.append(torch.zeros(batch, dtype=torch.long))
        dyn[name] = {0: "batch"}
    dummy.append(torch.zeros(batch, len(model.dense_cols)))
    dyn["dense"] = {0: "batch"}
    for s in model.sequences:
        c = s["column"]
        dummy.append(torch.zeros(batch, L, dtype=torch.long)); dyn[f"seq__{c}"] = {0: "batch"}
        dummy.append(torch.zeros(batch, dtype=torch.long));    dyn[f"seqlen__{c}"] = {0: "batch"}

    names = wrap.input_order()
    dyn["pctr"] = {0: "batch"}
    torch.onnx.export(
        wrap, tuple(dummy), path,
        input_names=names, output_names=["pctr"],
        dynamic_axes=dyn, opset_version=14,
    )
    with open(sidecar_path, "w") as f:
        json.dump({"input_order": names, "output": "pctr"}, f, indent=2)
    print(f"[onnx] {path}  (inputs: {len(names)})  +  {sidecar_path}")


def load_spec(path: str) -> dict:
    with open(path) as f:
        return json.load(f)