import { useState } from "react";
import { rank } from "../api";
import type { RankedAd } from "../types";
import React from "react";

interface RankedFeedProps {
  userId: string;
}

export default function RankedFeed({ userId }: RankedFeedProps) {
  const [results, setResults] = useState<RankedAd[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [topN, setTopN] = useState(10);

  const run = async () => {
    if (!userId) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await rank(userId, [], topN);
      setResults(r.results ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const maxP = results.length ? results[0].pctr : 1;

  return (
    <div className="panel panel-feed">
      <div className="panel-head">
        <h2>ranked feed</h2>
        <div className="feed-controls">
          <label className="topn">
            top
            <input
              type="number"
              min="1"
              max="50"
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
            />
          </label>
          <button
            className="btn btn-sm"
            onClick={run}
            disabled={busy || !userId}
          >
            {busy ? "ranking…" : "rank catalog"}
          </button>
        </div>
      </div>

      {err && <div className="err">{err}</div>}

      <ol className="feed">
        {results.map((r, i) => (
          <li
            className="feed-item"
            key={r.ad_id}
            style={{ animationDelay: `${i * 35}ms` }}
          >
            <span className="rank-num">{String(i + 1).padStart(2, "0")}</span>
            <div className="feed-body">
              <div className="feed-title">{r.title}</div>
              <div className="feed-bar">
                <div
                  className="feed-bar-fill"
                  style={{ width: `${(r.pctr / maxP) * 100}%` }}
                />
              </div>
            </div>
            <span className="feed-score">{(r.pctr * 100).toFixed(3)}%</span>
          </li>
        ))}
      </ol>

      {!results.length && !busy && (
        <div className="empty">
          rank the catalog to score every ad for this user in one batched pass
        </div>
      )}
    </div>
  );
}
