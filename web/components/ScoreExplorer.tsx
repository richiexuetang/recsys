import { useState } from "react";
import { score } from "../api";
import type { Ad, ScoreResult } from "../types";
import AttentionView from "./AttentionView";
import React from "react";

function Gauge({ value }: { value: number }) {
  // value in [0,1]; CTR is small, so use a sqrt scale for visible movement
  const pct = Math.min(100, Math.sqrt(value) * 100);
  return (
    <div className="gauge">
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="gauge-val">
        <span className="big">{(value * 100).toFixed(3)}</span>
        <span className="unit">% pCTR</span>
      </div>
    </div>
  );
}

interface ScoreExplorerProps {
  userId: string;
  ads: Ad[];
}

export default function ScoreExplorer({ userId, ads }: ScoreExplorerProps) {
  const [adId, setAdId] = useState<string>(ads[0]?.id ?? "");
  const [result, setResult] = useState<ScoreResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    if (!userId || !adId) return;
    setBusy(true);
    setErr(null);
    try {
      setResult(await score(userId, adId));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const selAd = ads.find((a) => a.id === adId);

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>explorer</h2>
        <span className="count">single pair</span>
      </div>

      <select
        className="select"
        value={adId}
        onChange={(e) => setAdId(e.target.value)}
      >
        {ads.map((a) => (
          <option key={a.id} value={a.id}>
            {a.title}
          </option>
        ))}
      </select>

      {selAd && (
        <div className="ad-meta">
          cate {selAd.sparse.cate_id} · brand {selAd.sparse.brand} · price{" "}
          {selAd.dense.price?.toFixed(3) ?? selAd.dense.price}
        </div>
      )}

      <button className="btn" onClick={run} disabled={busy || !userId}>
        {busy ? "scoring…" : "score user × ad"}
      </button>

      {err && <div className="err">{err}</div>}
      {result && !err && <Gauge value={result.pctr} />}
      {result && !err && result.attention && (
        <AttentionView
          attention={result.attention}
          candidateCate={selAd?.sparse.cate_id}
          candidateBrand={selAd?.sparse.brand}
        />
      )}
    </div>
  );
}
