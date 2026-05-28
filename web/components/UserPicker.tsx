import React from "react";
import type { User, SeqColumn } from "../types";

// btag codes in this dataset: view / cart / fav / buy. Labeled loosely for UI.
const BTAG: Record<string, string> = {
  "0": "view",
  "1": "cart",
  "2": "fav",
  "3": "buy",
};

interface SeqChipsProps {
  label: SeqColumn;
  tokens: string[] | undefined;
  max?: number;
}

function SeqChips({ label, tokens, max = 18 }: SeqChipsProps) {
  if (!tokens || tokens.length === 0) return null;
  const shown = tokens.slice(-max);
  const hidden = tokens.length - shown.length;
  return (
    <div className="seq-row">
      <span className="seq-label">{label}</span>
      <div className="seq-chips">
        {hidden > 0 && <span className="chip chip-more">+{hidden}</span>}
        {shown.map((t, i) => (
          <span className="chip" key={i}>
            {label === "btag_his" ? BTAG[t] ?? t : t}
          </span>
        ))}
      </div>
    </div>
  );
}

interface UserPickerProps {
  users: User[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export default function UserPicker({
  users,
  selectedId,
  onSelect,
}: UserPickerProps) {
  const sel = users.find((u) => u.id === selectedId);
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>user</h2>
        <span className="count">{users.length} sampled</span>
      </div>

      <select
        className="select"
        value={selectedId ?? ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        {users.map((u) => (
          <option key={u.id} value={u.id}>
            uid:{u.id} · age{u.sparse.age_level} · g{u.sparse.final_gender_code}
          </option>
        ))}
      </select>

      {sel && (
        <>
          <div className="profile-grid">
            {Object.entries(sel.sparse).map(([k, v]) => (
              <div className="kv" key={k}>
                <span className="k">{k}</span>
                <span className="v">{v}</span>
              </div>
            ))}
          </div>

          <div className="history">
            <div className="history-head">
              behavior history → DIN attention input
            </div>
            <SeqChips label="cate_his" tokens={sel.seqs.cate_his} />
            <SeqChips label="brand_his" tokens={sel.seqs.brand_his} />
            <SeqChips label="btag_his" tokens={sel.seqs.btag_his} />
          </div>
        </>
      )}
    </div>
  );
}
