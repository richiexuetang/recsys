package main

import (
	"fmt"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// Runner owns the ONNX session. The exported graph has multiple outputs:
//   pctr            [N]      click probability
//   attn__<col>     [N, L]   attention weights per attended sequence
// Infer runs the graph and returns every output as a flat slice keyed by name.
type Runner struct {
	enc      *Encoder
	session  *ort.DynamicAdvancedSession
	inNames  []string
	outNames []string
	seqLen   int64
}

func NewRunner(enc *Encoder, onnxPath, ortLibPath string) (*Runner, error) {
	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}
	in := enc.IO.InputOrder
	out := enc.IO.Outputs
	if len(out) == 0 {
		return nil, fmt.Errorf("sidecar has no outputs; re-export the model with the updated din.py")
	}
	session, err := ort.NewDynamicAdvancedSession(onnxPath, in, out, nil)
	if err != nil {
		return nil, fmt.Errorf("open session: %w", err)
	}
	return &Runner{
		enc: enc, session: session,
		inNames: in, outNames: out, seqLen: int64(enc.Spec.MaxSeqLen),
	}, nil
}

func (r *Runner) Close() {
	if r.session != nil {
		r.session.Destroy()
	}
	ort.DestroyEnvironment()
}

// Infer runs one batch and returns each output flattened, keyed by output name.
// "pctr" has length N; "attn__<col>" has length N*L (row-major).
func (r *Runner) Infer(batch []*Encoded) (map[string][]float32, error) {
	n := int64(len(batch))
	if n == 0 {
		return map[string][]float32{}, nil
	}
	denseDim := int64(len(r.enc.Spec.Dense))

	inputs := make([]ort.Value, len(r.inNames))
	defer func() {
		for _, v := range inputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	for i, name := range r.inNames {
		switch {
		case name == "dense":
			data := make([]float32, n*denseDim)
			for b, e := range batch {
				copy(data[int64(b)*denseDim:], e.Dense)
			}
			t, err := ort.NewTensor(ort.NewShape(n, denseDim), data)
			if err != nil {
				return nil, err
			}
			inputs[i] = t
		case strings.HasPrefix(name, "seq__"):
			data := make([]int64, n*r.seqLen)
			for b, e := range batch {
				copy(data[int64(b)*r.seqLen:], e.Seqs[name])
			}
			t, err := ort.NewTensor(ort.NewShape(n, r.seqLen), data)
			if err != nil {
				return nil, err
			}
			inputs[i] = t
		case strings.HasPrefix(name, "seqlen__"):
			data := make([]int64, n)
			for b, e := range batch {
				data[b] = e.SeqLens[name]
			}
			t, err := ort.NewTensor(ort.NewShape(n), data)
			if err != nil {
				return nil, err
			}
			inputs[i] = t
		default: // sparse__<col>
			data := make([]int64, n)
			for b, e := range batch {
				data[b] = e.Sparse[name]
			}
			t, err := ort.NewTensor(ort.NewShape(n), data)
			if err != nil {
				return nil, err
			}
			inputs[i] = t
		}
	}

	// allocate one output tensor per declared output name
	outTensors := make([]*ort.Tensor[float32], len(r.outNames))
	outVals := make([]ort.Value, len(r.outNames))
	defer func() {
		for _, t := range outTensors {
			if t != nil {
				t.Destroy()
			}
		}
	}()
	for i, name := range r.outNames {
		var shape ort.Shape
		if strings.HasPrefix(name, "attn__") {
			shape = ort.NewShape(n, r.seqLen)
		} else {
			shape = ort.NewShape(n)
		}
		t, err := ort.NewEmptyTensor[float32](shape)
		if err != nil {
			return nil, err
		}
		outTensors[i] = t
		outVals[i] = t
	}

	if err := r.session.Run(inputs, outVals); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	res := make(map[string][]float32, len(r.outNames))
	for i, name := range r.outNames {
		src := outTensors[i].GetData()
		cp := make([]float32, len(src))
		copy(cp, src)
		res[name] = cp
	}
	return res, nil
}

// Scores is the fast path for ranking: just the pCTR vector.
func (r *Runner) Scores(batch []*Encoded) ([]float32, error) {
	out, err := r.Infer(batch)
	if err != nil {
		return nil, err
	}
	return out["pctr"], nil
}