import type { Catalog, ScoreResult, RankResult } from "./types";

// Thin client for the Go inference server. In dev, Vite proxies these paths
// to :8080 (see vite.config.ts); in prod set VITE_API_BASE.
const BASE: string = import.meta.env.VITE_API_BASE ?? "";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export async function getCatalog(): Promise<Catalog> {
  const res = await fetch(BASE + "/catalog");
  if (!res.ok) throw new Error(`/catalog -> ${res.status}`);
  return res.json() as Promise<Catalog>;
}

// single user x ad -> { pctr }
export const score = (userId: string, adId: string): Promise<ScoreResult> =>
  post<ScoreResult>("/score", { user_id: userId, ad_id: adId });

// rank a candidate set (empty ad_ids => whole catalog)
export const rank = (
  userId: string,
  adIds: string[] = [],
  topN = 0
): Promise<RankResult> =>
  post<RankResult>("/rank", { user_id: userId, ad_ids: adIds, top_n: topN });
