package main

// Solver: alpha-beta with transposition table and move ordering for
// n×n×n tic-tac-toe. Values are from the side-to-move's perspective:
//   +WIN = side to move wins with perfect play
//   -WIN = side to move loses
//   0    = draw
// When depth-limited, a line heuristic provides the cutoff value.
//
// The board is a flat []int8 of length n^3: 0 empty, 1 X, 2 O.

import (
	"math/rand"
	"sync"
)

const WIN = 1_000_000

// BoardState is a copy-friendly game state.
type BoardState struct {
	n     int
	cells []int8
	lines [][]int
}

func NewBoard(n int) *BoardState {
	b := &BoardState{n: n, cells: make([]int8, n*n*n)}
	b.buildLines()
	return b
}

func (b *BoardState) buildLines() {
	dirs := [][3]int{
		{1, 0, 0}, {0, 1, 0}, {0, 0, 1},
		{1, 1, 0}, {1, -1, 0}, {1, 0, 1}, {1, 0, -1}, {0, 1, 1}, {0, 1, -1},
		{1, 1, 1}, {1, 1, -1}, {1, -1, 1}, {-1, 1, 1},
	}
	n := b.n
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			for z := 0; z < n; z++ {
				for _, d := range dirs {
					pts := make([]int, 0, n)
					ok := true
					for k := 0; k < n; k++ {
						px := x + d[0]*k
						py := y + d[1]*k
						pz := z + d[2]*k
						if px < 0 || py < 0 || pz < 0 || px >= n || py >= n || pz >= n {
							ok = false
							break
						}
						pts = append(pts, px+n*(py+n*pz))
					}
					if ok {
						b.lines = append(b.lines, pts)
					}
				}
			}
		}
	}
}

func (b *BoardState) clone() *BoardState {
	cp := &BoardState{n: b.n, lines: b.lines}
	cp.cells = make([]int8, len(b.cells))
	copy(cp.cells, b.cells)
	return cp
}

func (b *BoardState) moves() []int {
	m := make([]int, 0, len(b.cells))
	for i, c := range b.cells {
		if c == 0 {
			m = append(m, i)
		}
	}
	return m
}

func (b *BoardState) emptyCount() int {
	c := 0
	for _, v := range b.cells {
		if v == 0 {
			c++
		}
	}
	return c
}

// winner returns the player (1 or 2) that has completed a line, else 0.
func (b *BoardState) winner() int8 {
	for _, line := range b.lines {
		v := b.cells[line[0]]
		if v == 0 {
			continue
		}
		same := true
		for _, idx := range line[1:] {
			if b.cells[idx] != v {
				same = false
				break
			}
		}
		if same {
			return v
		}
	}
	return 0
}

func (b *BoardState) over() bool {
	return b.winner() != 0 || b.emptyCount() == 0
}

// Solver runs alpha-beta with a shared transposition table guarded by a lock.
type Solver struct {
	n       int
	mu      sync.RWMutex
	table   map[uint64]int
	rng     *rand.Rand
	maxDeep int
}

// NewSolver returns a solver for an n×n×n cube.
// maxDeep = n² plies: enough to see every forced win/loss line (a game ends
// in at most n² plies: the winning side plays n moves). Beyond that the
// heuristic cutoff handles larger boards.
func NewSolver(n int) *Solver {
	return &Solver{
		n:       n,
		table:   make(map[uint64]int, 1<<20),
		rng:     rand.New(rand.NewSource(1)),
		maxDeep: n * n,
	}
}

// key hashes the cells (no player needed: side-to-move is implied by parity,
// but we include a parity bit for safety).
func (s *Solver) key(b *BoardState, player int8) uint64 {
	var h uint64 = 0xcbf29ce484222325
	for _, c := range b.cells {
		h ^= uint64(c)
		h *= 0x100000001b3
	}
	h ^= uint64(player)
	return h
}

// BestMove returns the best move and its value from `player`'s perspective.
func (s *Solver) BestMove(b *BoardState, player int8) (int, int) {
	ms := b.moves()
	if len(ms) == 0 {
		return -1, 0 // no legal moves
	}
	opp := 3 - player
	for _, m := range ms {
		b.cells[m] = player
		if b.winner() == player {
			b.cells[m] = 0
			return m, WIN
		}
		b.cells[m] = 0
	}
	for _, m := range ms {
		b.cells[m] = opp
		if b.winner() == opp {
			b.cells[m] = 0
			return m, -WIN
		}
		b.cells[m] = 0
	}

	best := -WIN - 1
	bestM := ms[0]
	for _, m := range s.ordered(b, player) {
		b.cells[m] = player
		v := -s.search(b, opp, s.maxDeep-1, -WIN-1, WIN+1)
		b.cells[m] = 0
		if v > best {
			best = v
			bestM = m
			if v >= WIN {
				break
			}
		}
	}
	return bestM, best
}

func (s *Solver) search(b *BoardState, player int8, depth, alpha, beta int) int {
	if w := b.winner(); w != 0 {
		if w == player {
			return WIN
		}
		return -WIN
	}
	if b.emptyCount() == 0 {
		return 0
	}
	if depth <= 0 {
		return s.heuristic(b, player)
	}

	k := s.key(b, player)
	s.mu.RLock()
	if v, ok := s.table[k]; ok {
		s.mu.RUnlock()
		return v
	}
	s.mu.RUnlock()

	opp := 3 - player
	best := -WIN - 1
	for _, m := range s.ordered(b, player) {
		b.cells[m] = player
		v := -s.search(b, opp, depth-1, -beta, -alpha)
		b.cells[m] = 0
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

	s.mu.Lock()
	s.table[k] = best
	s.mu.Unlock()
	return best
}

// ordered returns moves sorted by proximity to the cube center.
func (s *Solver) ordered(b *BoardState, player int8) []int {
	center := float64(b.n-1) / 2
	type sc struct {
		d  float64
		ix int
	}
	arr := make([]sc, 0, len(b.cells))
	for i, c := range b.cells {
		if c != 0 {
			continue
		}
		x := i % b.n
		y := (i / b.n) % b.n
		z := i / (b.n * b.n)
		dx := float64(x) - center
		dy := float64(y) - center
		dz := float64(z) - center
		arr = append(arr, sc{dx*dx + dy*dy + dz*dz, i})
	}
	// insertion sort (small slices)
	for i := 1; i < len(arr); i++ {
		for j := i; j > 0 && arr[j].d < arr[j-1].d; j-- {
			arr[j], arr[j-1] = arr[j-1], arr[j]
		}
	}
	out := make([]int, len(arr))
	for i, a := range arr {
		out[i] = a.ix
	}
	return out
}

func (s *Solver) heuristic(b *BoardState, player int8) int {
	opp := 3 - player
	score := 0
	for _, line := range b.lines {
		own, oppn := 0, 0
		for _, ix := range line {
			v := b.cells[ix]
			if v == player {
				own++
			} else if v == opp {
				oppn++
			}
		}
		if own > 0 && oppn > 0 {
			continue
		}
		if own > 1 {
			score += pow10(own - 1)
		} else if own == 1 {
			score += 1
		} else if oppn > 1 {
			score -= pow10(oppn - 1)
		} else if oppn == 1 {
			score -= 1
		}
	}
	return score
}

func pow10(e int) int {
	r := 1
	for i := 0; i < e; i++ {
		r *= 10
	}
	return r
}