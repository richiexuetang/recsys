// Domain types — mirror the Go server's Catalog/User/Ad structs and endpoint
// payloads. Keeping them here means a shape change on the server surfaces as a
// compile error here rather than a runtime undefined.

export type SeqColumn = "cate_his" | "brand_his" | "btag_his";

export interface User {
  id: string;
  sparse: Record<string, string>; // user-side fields (age_level, ...)
  seqs: Record<SeqColumn, string[]>; // behavior history tokens
}

export interface Ad {
  id: string;
  title: string;
  sparse: Record<string, string>; // cate_id, brand, adgroup_id, ...
  dense: Record<string, number>; // { price: number }
}

export interface Catalog {
  users: User[];
  ads: Ad[];
}

export interface AttnWeight {
  token: string;
  weight: number;
}

export interface ScoreResult {
  user_id: string;
  ad_id: string;
  pctr: number;
  // per attended sequence (cate_his, brand_his): weight on each historical token
  attention?: Record<string, AttnWeight[]>;
}

export interface RankedAd {
  ad_id: string;
  title: string;
  pctr: number;
}

export interface RankResult {
  user_id: string;
  results: RankedAd[];
}
