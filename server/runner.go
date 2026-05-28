package main

import (
	"fmt"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// Runner owns the ONNX session and turns batches of Encoded into pCTR scores.
type Runner struct {
	enc     *Encoder
	session *ort.DynamicAdvancedSession
	inNames []string
	outName string
	seqLen  int64
}

// NewRunner initializes the ORT environment and opens a session on din.onnx.
// onnxPath is the model; the shared library path is the platform's ORT .so/.dylib/.dll.
func NewRunner(enc *Encoder, onnxPath, ortLibPath string) (*Runner, error) {
	if ortLibPath != "" {
		ort.SetSharedLibraryPath(ortLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ort init: %w", err)
	}

	in := enc.IO.InputOrder
	out := enc.IO.Output
	session, err := ort.NewDynamicAdvancedSession(onnxPath, in, []string{out}, nil)
	if err != nil {
		return nil, fmt.Errorf("open session: %w", err)
	}
	return &Runner{
		enc:     enc,
		session: session,
		inNames: in,
		outName: out,
		seqLen:  int64(enc.Spec.MaxSeqLen),
	}, nil
}

func (r *Runner) Close() {
	if r.session != nil {
		r.session.Destroy()
	}
	ort.DestroyEnvironment()
}

// Score runs a batch and returns one pCTR per request, in input order.
func (r *Runner) Score(batch []*Encoded) ([]float32, error) {
	n := int64(len(batch))
	if n == 0 {
		return nil, nil
	}

	// Build one ORT tensor per declared input name. The name prefix tells us
	// its shape: sparse__/seqlen__ are [N], seq__ is [N, L], dense is [N, D].
	inputs := make([]ort.Value, len(r.inNames))
	defer func() {
		for _, v := range inputs {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	denseDim := int64(len(r.enc.Spec.Dense))

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

	out, err := ort.NewEmptyTensor[float32](ort.NewShape(n))
	if err != nil {
		return nil, err
	}
	defer out.Destroy()

	if err := r.session.Run(inputs, []ort.Value{out}); err != nil {
		return nil, fmt.Errorf("inference: %w", err)
	}

	scores := make([]float32, n)
	copy(scores, out.GetData())
	return scores, nil
}
