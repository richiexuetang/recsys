import { useEffect, useState } from "react";
import { getCatalog } from "./api";
import type { Catalog } from "./types";
import UserPicker from "./components/UserPicker";
import ScoreExplorer from "./components/ScoreExplorer";
import RankedFeed from "./components/RankedFeed";
import React from "react";

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [userId, setUserId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getCatalog()
      .then((c) => {
        setCatalog(c);
        setUserId(c.users[0]?.id ?? null);
      })
      .catch((e) => setErr(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">▮▮▯</span>
          <div>
            <div className="brand-name">DIN RECSYS</div>
            <div className="brand-sub">
              deep interest network · ctr inference console
            </div>
          </div>
        </div>
        <div className="stack-badge">
          <span>PyTorch</span>
          <span className="arrow">→</span>
          <span>ONNX</span>
          <span className="arrow">→</span>
          <span>Go</span>
          <span className="arrow">→</span>
          <span>React</span>
        </div>
      </header>

      {err && <div className="banner err">cannot reach server: {err}</div>}

      {catalog && userId && (
        <main className="grid">
          <div className="col col-left">
            <UserPicker
              users={catalog.users}
              selectedId={userId}
              onSelect={setUserId}
            />
            <ScoreExplorer userId={userId} ads={catalog.ads} />
          </div>
          <div className="col col-right">
            <RankedFeed userId={userId} />
          </div>
        </main>
      )}

      {!catalog && !err && <div className="loading">loading catalog…</div>}

      <footer className="foot">
        target-attention pooling over behavior sequences · scores assembled
        server-side from {catalog?.ads.length ?? "—"} ads
      </footer>
    </div>
  );
}
