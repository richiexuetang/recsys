package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// --- Spec types: mirror feature_spec.json from prepare_data.py ---------------

type SparseFeature struct {
	Column string `json:"column"`
	Vocab  string `json:"vocab"`
}

type SequenceFeature struct {
	Column       string `json:"column"`
	Vocab        string `json:"vocab"`
	TargetColumn string `json:"target_column"`
}

type FeatureSpec struct {
	MaxSeqLen  int               `json:"max_seq_len"`
	PadIndex   int               `json:"pad_index"`
	OOVIndex   int               `json:"oov_index"`
	Label      string            `json:"label"`
	Dense      []string          `json:"dense"`
	VocabSizes map[string]int    `json:"vocab_sizes"`
	Sparse     []SparseFeature   `json:"sparse_features"`
	Sequences  []SequenceFeature `json:"sequence_features"`
}

// Sidecar din.onnx.json: the positional order Go must feed tensors in, and the
// ordered output names (pctr first, then attn__<col> per attended sequence).
type ModelIO struct {
	InputOrder []string `json:"input_order"`
	Outputs    []string `json:"outputs"`
}

// --- Encoder -----------------------------------------------------------------

type Encoder struct {
	Spec   *FeatureSpec
	IO     *ModelIO
	vocabs map[string]map[string]int64 // vocab name -> token -> index
}

func LoadEncoder(buildDir string) (*Encoder, error) {
	e := &Encoder{vocabs: map[string]map[string]int64{}}

	if err := readJSON(filepath.Join(buildDir, "feature_spec.json"), &e.Spec); err != nil {
		return nil, fmt.Errorf("spec: %w", err)
	}
	if err := readJSON(filepath.Join(buildDir, "din.onnx.json"), &e.IO); err != nil {
		return nil, fmt.Errorf("sidecar: %w", err)
	}

	// load every vocab the spec references (dedupe shared ones)
	needed := map[string]bool{}
	for _, f := range e.Spec.Sparse {
		needed[f.Vocab] = true
	}
	for _, s := range e.Spec.Sequences {
		needed[s.Vocab] = true
	}
	for name := range needed {
		var raw map[string]int64
		p := filepath.Join(buildDir, "vocabs", name+".json")
		if err := readJSON(p, &raw); err != nil {
			return nil, fmt.Errorf("vocab %s: %w", name, err)
		}
		e.vocabs[name] = raw
	}
	return e, nil
}

// lookup mirrors prepare_data.py: known token -> its index, anything else -> OOV.
func (e *Encoder) lookup(vocab, token string) int64 {
	if v, ok := e.vocabs[vocab]; ok {
		if idx, ok := v[token]; ok {
			return idx
		}
	}
	return int64(e.Spec.OOVIndex)
}

// Request is one user x one candidate ad. Sequences are the user's history;
// the caller supplies them as already-split string slices (most-recent last,
// same convention as the caret-split in prep).
type Request struct {
	Sparse map[string]string   `json:"sparse"` // column -> raw token, e.g. "cate_id":"6261"
	Dense  map[string]float32  `json:"dense"`  // column -> value, e.g. "price":0.13
	Seqs   map[string][]string `json:"seqs"`   // column -> history tokens
}

// Encoded holds the per-request tensors keyed by ONNX input name. The runner
// stacks these across a batch in InputOrder.
type Encoded struct {
	Sparse  map[string]int64   // "sparse__<col>" -> index
	Dense   []float32          // aligned to Spec.Dense order
	Seqs    map[string][]int64 // "seq__<col>" -> padded indices (len = MaxSeqLen)
	SeqLens map[string]int64   // "seqlen__<col>" -> true length (pre-pad, capped)
}

func (e *Encoder) Encode(r *Request) *Encoded {
	out := &Encoded{
		Sparse:  map[string]int64{},
		Seqs:    map[string][]int64{},
		SeqLens: map[string]int64{},
	}

	for _, f := range e.Spec.Sparse {
		out.Sparse["sparse__"+f.Column] = e.lookup(f.Vocab, r.Sparse[f.Column])
	}

	out.Dense = make([]float32, len(e.Spec.Dense))
	for i, c := range e.Spec.Dense {
		v := r.Dense[c]
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		out.Dense[i] = v
	}

	L := e.Spec.MaxSeqLen
	for _, s := range e.Spec.Sequences {
		hist := r.Seqs[s.Column]
		// keep most-recent L, right-pad with PAD -- identical to prep
		if len(hist) > L {
			hist = hist[len(hist)-L:]
		}
		padded := make([]int64, L) // zero-filled == PAD
		for i, tok := range hist {
			padded[i] = e.lookup(s.Vocab, tok)
		}
		out.Seqs["seq__"+s.Column] = padded
		out.SeqLens["seqlen__"+s.Column] = int64(len(hist))
	}
	return out
}

// CappedHistory returns the same most-recent-L slice Encode pads from, so the
// caller can zip attention weights[i] to the historical token at position i.
func (e *Encoder) CappedHistory(tokens []string) []string {
	L := e.Spec.MaxSeqLen
	if len(tokens) > L {
		return tokens[len(tokens)-L:]
	}
	return tokens
}

func readJSON(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, v)
}