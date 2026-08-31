package main

// distill generates search-distillation training data using all CPU cores.
//
// Each worker plays full games where the alpha-beta solver picks every move
// (with a little random exploration so games vary). Every position visited
// is recorded with the solver's best move + game value, so one game yields
// ~n^2 labeled positions instead of one.
//
// A single Solver is reused per worker, so its transposition table carries
// across positions within a game (huge speedup) — and we keep the table
// warm across games too (bounded).
//
// Output format (binary, little-endian):
//   header : [n int32][d_model int32][n_layers int32]
//   record : n^3 x int8 cells, int32 best_move, int32 value, int32 game_id
// (game_id lets the trainer split train/eval by whole games — no leakage)

import (
	"bufio"
	"encoding/binary"
	"flag"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

type Sample struct {
	cells []int8
	move  int
	value int
	game  int
}

func main() {
	n := flag.Int("n", 3, "cube size")
	games := flag.Int("games", 100_000, "full games to play")
	explore := flag.Float64("explore", 0.15, "prob of a random move per ply (variety)")
	workers := flag.Int("workers", 0, "goroutines (default = NumCPU)")
	seed := flag.Int64("seed", 0, "rng seed")
	out := flag.String("out", "distill_data.bin", "output file")
	appendMode := flag.Bool("append", false, "append to existing file (resume)")
	flag.Parse()

	if *workers <= 0 {
		*workers = runtime.NumCPU()
	}
	log.Printf("n=%d workers=%d games=%d explore=%.2f -> %s", *n, *workers, *games, *explore, *out)

	var produced atomic.Int64
	start := time.Now()
	samples := make(chan Sample, *workers*32)
	done := make(chan struct{})

	var f *os.File
	var err error
	if *appendMode {
		f, err = os.OpenFile(*out, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		f, err = os.Create(*out)
	}
	if err != nil {
		log.Fatal(err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	go func() {
		if !*appendMode {
			header := make([]byte, 12)
			binary.LittleEndian.PutUint32(header[0:], uint32(*n))
			binary.LittleEndian.PutUint32(header[4:], 0) // d_model placeholder
			binary.LittleEndian.PutUint32(header[8:], 0) // n_layers placeholder
			bw.Write(header)
		}
		var buf [12]byte
		written := 0
		last := time.Now()
		for s := range samples {
			for _, c := range s.cells {
				bw.WriteByte(byte(c))
			}
			binary.LittleEndian.PutUint32(buf[0:], uint32(s.move))
			bw.Write(buf[0:4])
			binary.LittleEndian.PutUint32(buf[4:], uint32(s.value))
			bw.Write(buf[4:8])
			binary.LittleEndian.PutUint32(buf[8:], uint32(s.game))
			bw.Write(buf[8:12])
			written++
			if written%10000 == 0 || time.Since(last) > 2*time.Second {
				bw.Flush()
				last = time.Now()
				log.Printf("writer: %d samples written", written)
			}
		}
		bw.Flush()
		f.Close()
		close(done)
	}()

	var wg sync.WaitGroup
	for w := 0; w < *workers; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(*seed + int64(worker)*7919))
			solver := NewSolver(*n)
			for {
				nn := produced.Add(1) - 1
				if nn >= int64(*games) {
					return
				}
				playGame(*n, solver, rng, *explore, samples)
			}
		}(w)
	}
	wg.Wait()
	close(samples)
	<-done
	log.Printf("done: %d games in %s", produced.Load(), time.Since(start).Round(time.Second))
}

var gameCounter atomic.Int64

// playGame plays one full game, recording every position with the solver's
// best move + value from the side-to-move's perspective.
func playGame(n int, solver *Solver, rng *rand.Rand, explore float64, out chan<- Sample) {
	b := NewBoard(n)
	player := int8(1)
	game := int(gameCounter.Add(1))
	steps := 0
	for !b.over() {
		steps++
		if steps > n*n*n+1 {
			log.Printf("guard: game %d exceeded %d plies", game, n*n*n+1)
			return
		}
		mv, val := solver.BestMove(b.clone(), player)
		if mv < 0 || mv >= n*n*n {
			log.Printf("guard: game %d solver returned bad move %d", game, mv)
			return
		}

		if rng.Float64() < explore {
			ms := b.moves()
			mv = ms[rng.Intn(len(ms))]
		}

		cells := make([]int8, len(b.cells))
		copy(cells, b.cells)
		out <- Sample{cells: cells, move: mv, value: val, game: game}

		b.cells[mv] = player
		player = 3 - player
	}
}