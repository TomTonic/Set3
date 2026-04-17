package set3

import (
	"testing"
	"unsafe"
)

func TestWhXmemHash_Comparison(t *testing.T) {
	seed := uint64(0x1234567890abcdef)

	// Test cases with different lengths and patterns
	lengths := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 49, 50, 95, 96, 97}

	// Different data patterns
	patterns := []struct {
		name string
		gen  func(n int) []byte
	}{
		{"all-zero", func(n int) []byte { return make([]byte, n) }},
		{"all-0xFF", func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = 0xFF
			}
			return b
		}},
		{"sequential", func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = byte(i)
			}
			return b
		}},
		{"alternating-0x55", func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = 0x55
			}
			return b
		}},
		{"alternating-0xAA", func(n int) []byte {
			b := make([]byte, n)
			for i := range b {
				b[i] = 0xAA
			}
			return b
		}},
		{"pattern-0x01020304", func(n int) []byte {
			b := make([]byte, n)
			pattern := []byte{0x01, 0x02, 0x03, 0x04}
			for i := range b {
				b[i] = pattern[i%len(pattern)]
			}
			return b
		}},
	}

	for _, pattern := range patterns {
		for _, length := range lengths {
			data := pattern.gen(length)

			// Compare memhashFallbackPort (unsafe.Pointer-based) and whX ([]byte-based)
			// Both are deterministic (no random hashkeys)
			hashMH := memhashFallbackPort(unsafe.Pointer(&data[0]), uint64(length), seed)
			hashX := whX(data, seed)

			// Compare results
			if hashMH != hashX {
				t.Errorf("pattern=%s len=%d: MH != whX: 0x%016x != 0x%016x",
					pattern.name, length, hashMH, hashX)
			}

			// Optional: log successful cases for visibility
			if testing.Verbose() {
				t.Logf("✓ pattern=%s len=%d: both match 0x%016x",
					pattern.name, length, hashMH)
			}
		}
	}
}

func TestWhX_EdgeCases(t *testing.T) {
	seed := uint64(0x0)

	tests := []struct {
		name string
		data []byte
		want uint64 // Set to 0 initially; we'll just check consistency
	}{
		{"empty", []byte{}, 0},
		{"single-null", []byte{0}, 0},
		{"single-0xFF", []byte{0xFF}, 0},
		{"two-nulls", []byte{0, 0}, 0},
		{"boundary-3bytes", []byte{1, 2, 3}, 0},
		{"boundary-4bytes", []byte{1, 2, 3, 4}, 0},
		{"boundary-7bytes", []byte{1, 2, 3, 4, 5, 6, 7}, 0},
		{"boundary-8bytes", []byte{1, 2, 3, 4, 5, 6, 7, 8}, 0},
		{"boundary-16bytes", []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 0},
		{"boundary-48bytes", make([]byte, 48), 0},
		{"boundary-49bytes", make([]byte, 49), 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Fill non-zero test data for the "make" cases
			if len(tt.data) > 16 {
				for i := range tt.data {
					tt.data[i] = byte(i)
				}
			}

			hashZ := whX(tt.data, seed)

			// Also test with different seed
			hashZ2 := whX(tt.data, seed+1)

			// Hash should change with different seed (unless data is empty)
			if len(tt.data) > 0 && hashZ == hashZ2 {
				t.Errorf("Hash did not change with different seed: 0x%016x", hashZ)
			}

			if testing.Verbose() {
				t.Logf("len=%d seed=0x%x -> 0x%016x", len(tt.data), seed, hashZ)
			}
		})
	}
}

func TestWhX_NullBytes(t *testing.T) {
	seed := uint64(42)

	// Test that different-length null-byte slices produce different hashes
	hashes := make(map[uint64]int)

	for length := 0; length <= 100; length++ {
		data := make([]byte, length)
		hash := whX(data, seed)

		if prevLen, exists := hashes[hash]; exists {
			t.Errorf("Collision: len=%d and len=%d both hash to 0x%016x",
				length, prevLen, hash)
		}
		hashes[hash] = length
	}

	t.Logf("Tested %d different null-byte lengths, all produced unique hashes", len(hashes))
}

func TestWhFunctions_DifferentSeeds(t *testing.T) {
	data := []byte("The quick brown fox jumps over the lazy dog")

	seeds := []uint64{0, 1, 42, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0}

	for _, seed := range seeds {
		hashMH := memhashFallbackPort(unsafe.Pointer(&data[0]), uint64(len(data)), seed)
		hashX := whX(data, seed)

		if hashMH != hashX {
			t.Errorf("seed=0x%x: hashMH != hashX: MH=0x%016x X=0x%016x",
				seed, hashMH, hashX)
		}

		if testing.Verbose() {
			t.Logf("seed=0x%016x -> 0x%016x", seed, hashMH)
		}
	}
}
