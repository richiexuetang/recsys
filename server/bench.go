package main

import (
	"fmt"
	"sort"
	"time"
)

// runBench measures the pure inference path (Runner.Infer) -- the same code /rank
// uses -- across a range of batch sizes. HTTP/JSON is deliberately excluded so the
// numbers reflect feature assembly + ONNX forward pass, not network or serializer.
//
// For each batch size it reports p50/p95/p99 latency and throughput in scores/sec.
// Candidates are built by pairing the first catalog user with catalog ads, cycling
// if the requested batch exceeds the catalog size.
func runBench(s *Server) {
	if len(s.cat.Users) == 0 || len(s.cat.Ads) == 0 {
		fmt.Println("[bench] empty catalog")
		return
	}
	user := &s.cat.Users[0]

	const warmup = 20
	const iters = 200
	sizes := []int{1, 8, 32, 64, 128, 256}

	fmt.Printf("[bench] user=%s  ads=%d  warmup=%d  iters=%d\n",
		user.ID, len(s.cat.Ads), warmup, iters)
	fmt.Println("  batch    p50(ms)   p95(ms)   p99(ms)   mean(ms)   throughput(scores/s)")
	fmt.Println("  -----    -------   -------   -------   --------   --------------------")

	for _, size := range sizes {
		batch := s.buildBatch(user, size)

		for i := 0; i < warmup; i++ {
			if _, err := s.run.Infer(batch); err != nil {
				fmt.Printf("[bench] error: %v\n", err)
				return
			}
		}

		lat := make([]time.Duration, iters)
		for i := 0; i < iters; i++ {
			t0 := time.Now()
			if _, err := s.run.Infer(batch); err != nil {
				fmt.Printf("[bench] error: %v\n", err)
				return
			}
			lat[i] = time.Since(t0)
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })

		p50 := lat[iters*50/100]
		p95 := lat[iters*95/100]
		p99 := lat[iters*99/100]
		var sum time.Duration
		for _, d := range lat {
			sum += d
		}
		mean := sum / time.Duration(iters)
		throughput := float64(size) / mean.Seconds()

		fmt.Printf("  %5d   %8.3f  %8.3f  %8.3f  %9.3f   %18.0f\n",
			size, ms(p50), ms(p95), ms(p99), ms(mean), throughput)
	}
}

// buildBatch encodes `size` user x ad pairs, cycling through the catalog ads.
func (s *Server) buildBatch(u *User, size int) []*Encoded {
	batch := make([]*Encoded, size)
	for i := 0; i < size; i++ {
		ad := &s.cat.Ads[i%len(s.cat.Ads)]
		batch[i] = s.enc.Encode(s.buildRequest(u, ad))
	}
	return batch
}

func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }