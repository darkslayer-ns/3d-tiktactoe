package main

/*
#cgo CFLAGS: -O2
#include <stdlib.h>
#include <stdint.h>
*/
import "C"

import (
	"sync"
	"unsafe"
)

// A persistent solver shared across calls (fast warm table).
var (
	mu     sync.Mutex
	solver *Solver
)

//export ttt_solver_init
func ttt_solver_init(n C.int) {
	mu.Lock()
	defer mu.Unlock()
	solver = NewSolver(int(n))
}

//export ttt_best_move
func ttt_best_move(cells *C.int8_t, n C.int, player C.int, outMove *C.int, outVal *C.int) C.int {
	mu.Lock()
	defer mu.Unlock()
	if solver == nil || solver.n != int(n) {
		return -1
	}
	b := NewBoard(int(n))
	nc := int(n * n * n)
	for i := 0; i < nc; i++ {
		b.cells[i] = int8(*(*C.int8_t)(unsafe.Pointer(uintptr(unsafe.Pointer(cells)) + uintptr(i))))
	}
	mv, val := solver.BestMove(b, int8(player))
	*outMove = C.int(mv)
	*outVal = C.int(val)
	return 0
}

func main() {} // c-shared requires an empty main