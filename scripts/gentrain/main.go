// gentrain generates training data for the 3x3x3 tic-tac-toe transformer.
//
// It plays out games where both sides move via an alpha-beta search teacher
// (iterative deepening within a per-move time budget) and records every
// position as:
//
//	struct Record {
//	    state [27]int8 // normalized: empty=0, side-to-move=1, opponent=2
//	    move  int8     // teacher's best move (0..26)
//	    value float32  // teacher's win-prob for the side to move (0..1)
//	}
//
// Records are written compactly (32 bytes each) to a binary file that the
// Python trainer reads. Producers run across all CPU cores.
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

const (
	N     = 3
	EMPTY = 0
	P1    = 1
	P2    = 2
	CELLS = N * N * N
)

var lines [][]int
var stop atomic.Bool

func buildLines() {
	dirs := [][3]int{}
	for dx := -1; dx <= 1; dx++ {
		for dy := -1; dy <= 1; dy++ {
			for dz := -1; dz <= 1; dz++ {
				if dx == 0 && dy == 0 && dz == 0 {
					continue
				}
				if dx > 0 || (dx == 0 && dy > 0) || (dx == 0 && dy == 0 && dz > 0) {
					dirs = append(dirs, [3]int{dx, dy, dz})
				}
			}
		}
	}
	for x := 0; x < N; x++ {
		for y := 0; y < N; y++ {
			for z := 0; z < N; z++ {
				for _, d := range dirs {
					line := make([]int, 0, N)
					ok := true
					for k := 0; k < N; k++ {
						cx, cy, cz := x+d[0]*k, y+d[1]*k, z+d[2]*k
						if cx < 0 || cx >= N || cy < 0 || cy >= N || cz < 0 || cz >= N {
							ok = false
							break
						}
						line = append(line, cx+N*(cy+N*cz))
					}
					if ok {
						lines = append(lines, line)
					}
				}
			}
		}
	}
}

type Board struct {
	cells [CELLS]int8
}

func (b *Board) moves() []int {
	m := make([]int, 0, CELLS)
	for i, c := range b.cells {
		if c == EMPTY {
			m = append(m, i)
		}
	}
	return m
}

func (b *Board) outcome() (int, bool) {
	for _, l := range lines {
		v := b.cells[l[0]]
		if v != EMPTY {
			all := true
			for _, c := range l[1:] {
				if b.cells[c] != v {
					all = false
					break
				}
			}
			if all {
				return int(v), true
			}
		}
	}
	for _, c := range b.cells {
		if c == EMPTY {
			return EMPTY, false
		}
	}
	return EMPTY, true
}

func pow10(n int) float64 {
	v := 1.0
	for i := 0; i < n; i++ {
		v *= 10
	}
	return v
}

// heuristic scores the position from `player`'s perspective.
func heuristic(b *Board, player int8) float64 {
	var s float64
	for _, l := range lines {
		var own, opp int
		for _, c := range l {
			v := b.cells[c]
			if v == player {
				own++
			} else if v != EMPTY {
				opp++
			}
		}
		if own > 0 && opp > 0 {
			continue
		}
		switch {
		case own > 0:
			if own >= N {
				s += 1e9
			} else {
				s += pow10(own)
			}
		case opp > 0:
			if opp >= N {
				s -= 1e9
			} else {
				s -= pow10(opp)
			}
		default:
			s += 1
		}
	}
	return s
}

func orderedMoves(b *Board, player int8) []int {
	moves := b.moves()
	type scored struct {
		h float64
		m int
	}
	sc := make([]scored, len(moves))
	for i, m := range moves {
		b.cells[m] = player
		sc[i] = scored{heuristic(b, player), m}
		b.cells[m] = EMPTY
	}
	// simple insertion sort by descending h
	for i := 1; i < len(sc); i++ {
		for j := i; j > 0 && sc[j].h > sc[j-1].h; j-- {
			sc[j], sc[j-1] = sc[j-1], sc[j]
		}
	}
	out := make([]int, len(moves))
	for i, s := range sc {
		out[i] = s.m
	}
	return out
}

func negamax(b *Board, depth int, alpha, beta float64, player int8) float64 {
	w, over := b.outcome()
	if over {
		if w == EMPTY {
			return 0
		}
		if int8(w) == player {
			return 1
		}
		return -1
	}
	if depth <= 0 {
		return heuristic(b, player) / 1e5
	}
	var opp int8 = P2
	if player == P2 {
		opp = P1
	}
	best := -1e18
	for _, m := range orderedMoves(b, player) {
		b.cells[m] = player
		v := -negamax(b, depth-1, -beta, -alpha, opp)
		b.cells[m] = EMPTY
		if v > best {
			best = v
		}
		if v > alpha {
			alpha = v
		}
		if alpha >= beta {
			break
		}
	}
	return best
}

func bestMove(b *Board, player int8, depth int) (float64, int) {
	w, over := b.outcome()
	if over {
		if w == EMPTY {
			return 0, -1
		}
		if int8(w) == player {
			return 1, -1
		}
		return -1, -1
	}
	moves := b.moves()
	var opp int8 = P2
	if player == P2 {
		opp = P1
	}
	bestVal := -1e18
	bestM := moves[0]
	alpha, beta := -1e18, 1e18
	for _, m := range orderedMoves(b, player) {
		b.cells[m] = player
		v := -negamax(b, depth-1, -beta, -alpha, opp)
		b.cells[m] = EMPTY
		if v > bestVal {
			bestVal = v
			bestM = m
		}
		if v > alpha {
			alpha = v
		}
		if alpha >= beta {
			break
		}
	}
	return bestVal, bestM
}

// teacherMove: iterative deepening within a time budget (nanoseconds).
func teacherMove(b *Board, player int8, budget time.Duration, maxDepth int) (int, float64) {
	moves := b.moves()
	bestM := moves[0]
	bestV := 0.0
	t0 := time.Now()
	for d := 1; d <= maxDepth; d++ {
		if time.Since(t0) > budget {
			break
		}
		v, m := bestMove(b, player, d)
		if m >= 0 {
			bestM = m
		}
		bestV = v
	}
	return bestM, bestV
}

// Record is the on-disk format: 27 state bytes + 1 move byte + 4 value bytes.
type Record struct {
	State [CELLS]int8
	Move  int8
	Value float32
}

func normalize(b *Board, player int8) [CELLS]int8 {
	var s [CELLS]int8
	for i, c := range b.cells {
		switch {
		case c == EMPTY:
			s[i] = 0
		case c == player:
			s[i] = 1
		default:
			s[i] = 2
		}
	}
	return s
}

type worker struct {
	rng *rand.Rand
}

func (w *worker) generate(out chan<- []Record, games *atomic.Int64, budget time.Duration, maxDepth int, noise float64) {
	batch := make([]Record, 0, 256)
	flush := func() {
		if len(batch) > 0 {
			cp := make([]Record, len(batch))
			copy(cp, batch)
			out <- cp
			batch = batch[:0]
		}
	}
	for {
		var b Board
		player := int8(P1)
		_, over := b.outcome()
		for !over {
			// label: greedy teacher move + search value at this position
			move, value := teacherMove(&b, player, budget, maxDepth)
			s := normalize(&b, player)
			batch = append(batch, Record{State: s, Move: int8(move), Value: float32((value + 1.0) / 2.0)})
			// play with noise for game/position diversity
			if w.rng.Float64() < noise {
				moves := b.moves()
				move = moves[w.rng.Intn(len(moves))]
			}
			b.cells[move] = player
			if player == P1 {
				player = P2
			} else {
				player = P1
			}
			_, over = b.outcome()
		}
		games.Add(1)
		flush()
		if stop.Load() {
			return
		}
	}
}

func main() {
	outPath := flag.String("out", "train_data.bin", "output binary file")
	budgetMs := flag.Int("budget", 100, "per-move search time budget in ms")
	maxDepth := flag.Int("depth", 5, "max search depth")
	target := flag.Int("records", 200000, "target number of records")
	seconds := flag.Float64("seconds", 0, "stop after this many seconds (0 = use records target)")
	play := flag.Bool("play", false, "debug: play and print a single game")
	stats := flag.Bool("stats", false, "debug: report avg game length")
	noise := flag.Float64("noise", 0.25, "fraction of played moves that are random (diversity)")
	flag.Parse()

	if *play {
		buildLines()
		fmt.Println("lines:", len(lines))
		b := &Board{}
		player := int8(P1)
		_, over := b.outcome()
		movesPlayed := 0
		for !over {
			move, value := teacherMove(b, player, 60*time.Millisecond, 4)
			fmt.Printf("ply %d: player=%d move=%d value=%.3f\n", movesPlayed, player, move, value)
			b.cells[move] = player
			movesPlayed++
			if player == P1 {
				player = P2
			} else {
				player = P1
			}
			_, over = b.outcome()
		}
		fmt.Println("game over in", movesPlayed, "moves")
		return
	}

	if *stats {
		buildLines()
		totalPlies, totalRecords, ng := 0, 0, 500
		for g := 0; g < ng; g++ {
			b := &Board{}
			player := int8(P1)
			plies := 0
			_, over := b.outcome()
			for !over {
				move, _ := teacherMove(b, player, 60*time.Millisecond, 4)
				b.cells[move] = player
				plies++
				totalRecords++
				if player == P1 {
					player = P2
				} else {
					player = P1
				}
				_, over = b.outcome()
			}
			totalPlies += plies
		}
		fmt.Printf("stats: %d games, avg plies=%.2f, avg records=%.2f\n", ng, float64(totalPlies)/float64(ng), float64(totalRecords)/float64(ng))
		return
	}

	buildLines()
	procs := runtime.NumCPU()
	fmt.Printf("gentrain: %d workers, budget=%dms depth=%d target=%d records\n", procs, *budgetMs, *maxDepth, *target)

	f, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer f.Close()
	bw := bufio.NewWriterSize(f, 1<<20)

	ch := make(chan []Record, procs*4)
	var wg sync.WaitGroup
	var count atomic.Int64
	var games atomic.Int64

	// writer
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		var buf [32]byte
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case batch, ok := <-ch:
				if !ok {
					bw.Flush()
					return
				}
				for _, r := range batch {
					for i := 0; i < CELLS; i++ {
						buf[i] = byte(r.State[i])
					}
					buf[27] = byte(r.Move)
					binary.LittleEndian.PutUint32(buf[28:], math.Float32bits(r.Value))
					if _, err := bw.Write(buf[:]); err != nil {
						fmt.Fprintln(os.Stderr, "write:", err)
						os.Exit(1)
					}
				}
				count.Add(int64(len(batch)))
			case <-ticker.C:
				bw.Flush()
			}
		}
	}()

	t0 := time.Now()
	timeLimit := time.Duration(*seconds * float64(time.Second))

	// producers
	done := make(chan struct{})
	for i := 0; i < procs; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			w := &worker{rng: rand.New(rand.NewSource(seed))}
			w.generate(ch, &games, time.Duration(*budgetMs)*time.Millisecond, *maxDepth, *noise)
		}(int64(i) + 12345)
	}

	// monitor
	go func() {
		for {
			time.Sleep(2 * time.Second)
			c := count.Load()
			g := games.Load()
			el := time.Since(t0)
			rate := float64(c) / el.Seconds()
			fmt.Printf("  records=%d games=%d rate=%.0f/s elapsed=%ds\n", c, g, rate, int(el.Seconds()))
			if (*seconds > 0 && el >= timeLimit) || (*seconds == 0 && c >= int64(*target)) {
				stop.Store(true)
				close(done)
				return
			}
		}
	}()

	<-done
	wg.Wait()
	close(ch)
	<-writerDone

	fmt.Printf("done: %d records (%d games) in %s\n", count.Load(), games.Load(), time.Since(t0))
}