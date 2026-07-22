package server

import "testing"

func TestAdmissionControllerAllowsUpToMax(t *testing.T) {
	a := NewAdmissionController(2)

	if _, ok := a.TryAcquire(); !ok {
		t.Fatal("expected first acquire to succeed")
	}
	if _, ok := a.TryAcquire(); !ok {
		t.Fatal("expected second acquire to succeed")
	}
	if _, ok := a.TryAcquire(); ok {
		t.Fatal("expected third acquire to be rejected (max=2)")
	}
	if got := a.Active(); got != 2 {
		t.Fatalf("Active() = %d, want 2 (rejected acquire must not hold a slot)", got)
	}
}

func TestAdmissionControllerReleaseFreesASlot(t *testing.T) {
	a := NewAdmissionController(1)

	if _, ok := a.TryAcquire(); !ok {
		t.Fatal("expected first acquire to succeed")
	}
	if _, ok := a.TryAcquire(); ok {
		t.Fatal("expected second acquire to be rejected while first is held")
	}

	a.Release()

	if _, ok := a.TryAcquire(); !ok {
		t.Fatal("expected acquire to succeed after Release")
	}
}

func TestAdmissionControllerZeroMaxRejectsImmediately(t *testing.T) {
	a := NewAdmissionController(0)
	if _, ok := a.TryAcquire(); ok {
		t.Fatal("expected acquire to be rejected when max=0")
	}
	if got := a.Active(); got != 0 {
		t.Fatalf("Active() = %d, want 0", got)
	}
}
