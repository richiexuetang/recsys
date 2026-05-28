package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"sort"
	"time"
)

// --- Catalog: sample users & ads the web app picks from ----------------------
// Loaded from catalog.json (a small hand-built or sampled file). Keeping a
// catalog server-side means the React app stays a thin client: it asks for
// users/ads by id and lets the server assemble the model features.

type User struct {
	ID     string            `json:"id"`
	Sparse map[string]string `json:"sparse"` // user-side fields, e.g. age_level
	// History tokens per sequence column (cate_his, brand_his, btag_his)
	Seqs map[string][]string `json:"seqs"`
}

type Ad struct {
	ID     string             `json:"id"`
	Title  string             `json:"title"`  // human label for the UI
	Sparse map[string]string  `json:"sparse"` // ad-side fields incl cate_id, brand
	Dense  map[string]float32 `json:"dense"`  // price
}

type Catalog struct {
	Users []User `json:"users"`
	Ads   []Ad   `json:"ads"`
}

type Server struct {
	enc *Encoder
	run *Runner
	cat *Catalog
	usr map[string]*User
	ad  map[string]*Ad
}

// buildRequest merges a user and an ad into one model Request.
func (s *Server) buildRequest(u *User, a *Ad) *Request {
	sparse := map[string]string{}
	for k, v := range u.Sparse {
		sparse[k] = v
	}
	for k, v := range a.Sparse {
		sparse[k] = v // ad fields (cate_id, brand, adgroup_id...) override/extend
	}
	return &Request{Sparse: sparse, Dense: a.Dense, Seqs: u.Seqs}
}

// --- Handlers ----------------------------------------------------------------

type scoreReq struct {
	UserID string `json:"user_id"`
	AdID   string `json:"ad_id"`
}

func (s *Server) handleScore(w http.ResponseWriter, r *http.Request) {
	var req scoreReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", 400)
		return
	}
	u, ok := s.usr[req.UserID]
	if !ok {
		http.Error(w, "unknown user", 404)
		return
	}
	a, ok := s.ad[req.AdID]
	if !ok {
		http.Error(w, "unknown ad", 404)
		return
	}
	enc := s.enc.Encode(s.buildRequest(u, a))
	scores, err := s.run.Score([]*Encoded{enc})
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	writeJSON(w, map[string]any{"user_id": req.UserID, "ad_id": req.AdID, "pctr": scores[0]})
}

type rankReq struct {
	UserID string   `json:"user_id"`
	AdIDs  []string `json:"ad_ids"` // candidate set; empty => rank whole catalog
	TopN   int      `json:"top_n"`
}

type rankedAd struct {
	AdID  string  `json:"ad_id"`
	Title string  `json:"title"`
	PCTR  float32 `json:"pctr"`
}

func (s *Server) handleRank(w http.ResponseWriter, r *http.Request) {
	var req rankReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", 400)
		return
	}
	u, ok := s.usr[req.UserID]
	if !ok {
		http.Error(w, "unknown user", 404)
		return
	}

	candidates := req.AdIDs
	if len(candidates) == 0 {
		for _, a := range s.cat.Ads {
			candidates = append(candidates, a.ID)
		}
	}

	batch := make([]*Encoded, 0, len(candidates))
	valid := make([]*Ad, 0, len(candidates))
	for _, id := range candidates {
		if a, ok := s.ad[id]; ok {
			batch = append(batch, s.enc.Encode(s.buildRequest(u, a)))
			valid = append(valid, a)
		}
	}

	scores, err := s.run.Score(batch)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	ranked := make([]rankedAd, len(valid))
	for i, a := range valid {
		ranked[i] = rankedAd{AdID: a.ID, Title: a.Title, PCTR: scores[i]}
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].PCTR > ranked[j].PCTR })

	n := req.TopN
	if n <= 0 || n > len(ranked) {
		n = len(ranked)
	}
	writeJSON(w, map[string]any{"user_id": req.UserID, "results": ranked[:n]})
}

func (s *Server) handleCatalog(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.cat)
}

// --- wiring ------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

func cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(204)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func main() {
	buildDir := flag.String("build", "./build", "dir with feature_spec.json, vocabs/, din.onnx")
	onnx := flag.String("onnx", "./build/din.onnx", "path to din.onnx")
	ortLib := flag.String("ort", "", "path to onnxruntime shared lib (.so/.dylib/.dll)")
	catalogPath := flag.String("catalog", "./build/catalog.json", "sample users/ads")
	addr := flag.String("addr", ":8080", "listen address")
	flag.Parse()

	enc, err := LoadEncoder(*buildDir)
	if err != nil {
		log.Fatalf("encoder: %v", err)
	}
	run, err := NewRunner(enc, *onnx, *ortLib)
	if err != nil {
		log.Fatalf("runner: %v", err)
	}
	defer run.Close()

	var cat Catalog
	if err := readJSON(*catalogPath, &cat); err != nil {
		log.Fatalf("catalog: %v", err)
	}
	srv := &Server{enc: enc, run: run, cat: &cat,
		usr: map[string]*User{}, ad: map[string]*Ad{}}
	for i := range cat.Users {
		srv.usr[cat.Users[i].ID] = &cat.Users[i]
	}
	for i := range cat.Ads {
		srv.ad[cat.Ads[i].ID] = &cat.Ads[i]
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/score", srv.handleScore)
	mux.HandleFunc("/rank", srv.handleRank)
	mux.HandleFunc("/catalog", srv.handleCatalog)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]string{"status": "ok"})
	})

	log.Printf("recsys server on %s  (users=%d ads=%d)", *addr, len(cat.Users), len(cat.Ads))
	httpSrv := &http.Server{
		Addr:         *addr,
		Handler:      cors(mux),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 30 * time.Second,
	}
	log.Fatal(httpSrv.ListenAndServe())
}
