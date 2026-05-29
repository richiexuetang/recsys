import type { AttnWeight } from "../types";
import React from "react";

// Highlights which historical items the model leaned on for THIS candidate.
// Weights are post-softmax over the user's behavior sequence; they sum to ~1
// across the sequence, so a few dominant tokens means focused interest.

function Row({
  label,
  tgt,
  weights,
}: {
  label: string;
  tgt: string;
  weights: AttnWeight[];
}) {
  if (!weights || weights.length === 0) return null;
  const max = Math.max(...weights.map((w) => w.weight), 1e-9);
  // chronological order is how the history is stored; show most-attended first
  const sorted = [...weights].sort((a, b) => b.weight - a.weight).slice(0, 12);
  return (
    <div className="attn-block">
      <div className="attn-label">
        {label} <span className="attn-tgt">vs candidate {tgt}</span>
      </div>
      <div className="attn-rows">
        {sorted.map((w, i) => {
          const isTarget = w.token === tgt;
          return (
            <div
              className={"attn-item" + (isTarget ? " attn-hit" : "")}
              key={i}
            >
              <span className="attn-tok">{w.token}</span>
              <span className="attn-bar">
                <span
                  className="attn-fill"
                  style={{ width: `${(w.weight / max) * 100}%` }}
                />
              </span>
              <span className="attn-pct">{(w.weight * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface AttentionViewProps {
  attention: Record<string, AttnWeight[]>;
  candidateCate?: string;
  candidateBrand?: string;
}

export default function AttentionView({
  attention,
  candidateCate,
  candidateBrand,
}: AttentionViewProps) {
  const cate = attention.cate_his;
  const brand = attention.brand_his;
  if (!cate && !brand) return null;
  return (
    <div className="attn">
      <div className="attn-head">attention — what drove this score</div>
      {cate && (
        <Row label="cate_his" tgt={candidateCate ?? "?"} weights={cate} />
      )}
      {brand && (
        <Row label="brand_his" tgt={candidateBrand ?? "?"} weights={brand} />
      )}
    </div>
  );
}
