package server

import "sync/atomic"

// AdmissionController bounds the number of concurrently active requests.
// This is one half of the project's two-layer backpressure: the gateway
// protects the network/goroutine layer, and the C++ engine's waiting_queue_
// independently protects KV cache memory (see CLAUDE.md "How does
// backpressure work?"). Either layer can reject independently.
type AdmissionController struct {
	active atomic.Int64
	max    int64
}

func NewAdmissionController(max int) *AdmissionController {
	return &AdmissionController{max: int64(max)}
}

// TryAcquire admits one request if under the limit. On success (ok==true)
// the caller must call Release exactly once when the request completes;
// on failure the slot was never held and Release must not be called.
func (a *AdmissionController) TryAcquire() (active int64, ok bool) {
	n := a.active.Add(1)
	if n > a.max {
		a.active.Add(-1)
		return n - 1, false
	}
	return n, true
}

func (a *AdmissionController) Release() {
	a.active.Add(-1)
}

func (a *AdmissionController) Active() int64 {
	return a.active.Load()
}
