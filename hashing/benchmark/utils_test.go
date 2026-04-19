package benchmark

import (
	"testing"
)

func TestFilteredNumbers(t *testing.T) {
	count := uint64(23)
	expected := []uint64{2, 3, 4, 5, 7, 8, 10, 11, 13, 16, 17, 19, 20, 23, 29, 30, 31, 32, 37, 40, 41, 43, 47}
	result := make([]uint64, 0, count)
	for n := range FilteredNumbers(count) {
		result = append(result, n)
	}
	if len(result) != int(count) {
		t.Fatalf("expected %d numbers, got %d", count, len(result))
	}
	for i, v := range expected {
		if result[i] != v {
			t.Fatalf("at index %d: expected %d, got %d", i, v, result[i])
		}
	}
}

// TestXorShift16StarPeriod verifies that the 16-bit xorshift generator
// has the expected period of 2^16-1.
func TestXorShift16StarPeriod(t *testing.T) {
	x := XorShift16Star{State: 1}
	start := x.State
	count := 0
	for {
		x.Uint16()
		count++
		if x.State == start {
			break
		}
		if count > 70000 {
			t.Fatalf("period exceeds 70000; expected 2^16-1 = 65535")
		}
	}
	if count != (1<<16)-1 {
		t.Fatalf("period = %d, expected %d", count, (1<<16)-1)
	}
}

// TestXorShift24StarPeriod verifies that the 24-bit xorshift generator
// has a stable long period for the current parameterization.
func TestXorShift24StarPeriod(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long xorshift24 period test in short mode")
	}
	x := XorShift24Star{State: 1}
	start := x.State
	count := 0
	const expectedPeriod = 5_504_982
	for {
		x.Uint24()
		count++
		if x.State == start {
			break
		}
		if count > expectedPeriod+1 {
			t.Fatalf("period exceeds expected bound; expected %d", expectedPeriod)
		}
	}
	if count != expectedPeriod {
		t.Fatalf("period = %d, expected %d", count, expectedPeriod)
	}
}

// TestXorShift32StarPeriod verifies that the 32-bit xorshift generator
// has the expected period of 2^32-1. This is a very long-running test.
func TestXorShift32StarPeriod(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping extremely long xorshift32 period test in short mode")
	}
	x := XorShift32Star{State: 1}
	start := x.State
	count := uint64(0)
	for {
		x.Uint32()
		count++
		if x.State == start {
			break
		}
		if count > 5_000_000_000 {
			t.Fatalf("period exceeds 5B; expected 2^32-1 = 4294967295")
		}
	}
	expected := uint64((1 << 32) - 1)
	if count != expected {
		t.Fatalf("period = %d, expected %d", count, expected)
	}
}

func TestNextPrime(t *testing.T) {
	tests := []struct {
		in   uint64
		want uint64
	}{
		{0, 2}, {1, 2}, {2, 2}, {3, 3}, {4, 5}, {10, 11}, {14, 17}, {17, 17}, {100, 101},
	}
	for _, tc := range tests {
		got := NextPrime(tc.in)
		if got != tc.want {
			t.Fatalf("NextPrime(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}
