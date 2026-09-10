//go:build set3lab

package setbench

import (
	"testing"
)

func TestNativeMapEmptyNativeWithCapacity(t *testing.T) {
	set := emptyNativeWithCapacity[int](0)
	if set == nil {
		t.Error("Expected non-nil set")
	}
	if set.count() != uint32(0) {
		t.Errorf("Expected count to be 0, got %d", set.count())
	}
}

func TestNativeMapAddAndContains(t *testing.T) {
	set := emptyNativeWithCapacity[int](0)
	set.add(1)
	set.add(2)
	set.add(3)

	if !set.contains(1) {
		t.Error("Expected set to contain 1")
	}
	if !set.contains(2) {
		t.Error("Expected set to contain 2")
	}
	if !set.contains(3) {
		t.Error("Expected set to contain 3")
	}
	if set.contains(4) {
		t.Error("Expected set not to contain 4")
	}
}

func TestNativeMapCount(t *testing.T) {
	set := emptyNativeWithCapacity[int](0)
	set.add(1)
	set.add(2)
	set.add(3)

	if set.count() != 3 {
		t.Errorf("Expected count to be 3, got %d", set.count())
	}

	set.add(3) // Adding duplicate
	if set.count() != 3 {
		t.Errorf("Expected count to be 3 after adding duplicate, got %d", set.count())
	}
}
