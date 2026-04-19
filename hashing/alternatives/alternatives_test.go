package alternatives

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

			// Compare MemhashFallbackPort (unsafe.Pointer-based) and WhX ([]byte-based)
			// Both are deterministic (no random hashkeys)
			hashMH := MemhashFallbackPort(unsafe.Pointer(&data[0]), uint64(length), seed)
			hashX := WhX(data, seed)

			// Compare results
			if hashMH != hashX {
				t.Errorf("pattern=%s len=%d: MH != WhX: 0x%016x != 0x%016x",
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

			hashZ := WhX(tt.data, seed)

			// Also test with different seed
			hashZ2 := WhX(tt.data, seed+1)

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
		hash := WhX(data, seed)

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
		hashMH := MemhashFallbackPort(unsafe.Pointer(&data[0]), uint64(len(data)), seed)
		hashX := WhX(data, seed)

		if hashMH != hashX {
			t.Errorf("seed=0x%x: hashMH != hashX: MH=0x%016x X=0x%016x",
				seed, hashMH, hashX)
		}

		if testing.Verbose() {
			t.Logf("seed=0x%016x -> 0x%016x", seed, hashMH)
		}
	}
}

// --- Slice-as-byte-array tests ---

func isLittleEndian() bool {
	var x uint16 = 1
	p := (*[2]byte)(unsafe.Pointer(&x))
	return p[0] == 1
}

func TestAnySliceAsByteSlice_EmptyAndCapOnly(t *testing.T) {
	var nilSlice []uint16
	empty := []uint16{}
	capOnly := make([]uint16, 0, 8)

	if AnySliceAsByteSlice[uint16](unsafe.Pointer(&nilSlice)) != nil {
		t.Fatalf("expected nil for nil slice")
	}
	if AnySliceAsByteSlice[uint16](unsafe.Pointer(&empty)) != nil {
		t.Fatalf("expected nil for empty slice")
	}
	if AnySliceAsByteSlice[uint16](unsafe.Pointer(&capOnly)) != nil {
		t.Fatalf("expected nil for zero-len slice even if cap>0")
	}
}

func TestAnySliceAsByteSlice_Uint16Roundtrip(t *testing.T) {
	s := []uint16{0x1122, 0x3344, 0x5566}
	eL := []byte{0x22, 0x11, 0x44, 0x33, 0x66, 0x55}
	eB := []byte{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}
	b := AnySliceAsByteSlice[uint16](unsafe.Pointer(&s))
	if b == nil {
		t.Fatalf("expected non-nil byte view for non-empty slice")
	}
	if got, want := len(b), len(s)*int(unsafe.Sizeof(s[0])); got != want {
		t.Fatalf("byte view length mismatch: got %d want %d", got, want)
	}

	if isLittleEndian() {
		for i := range eL {
			if b[i] != eL[i] {
				t.Fatalf("mismatch at byte %d: got %#x want %#x", i, b[i], eL[i])
			}
		}
	} else {
		for i := range eB {
			if b[i] != eB[i] {
				t.Fatalf("mismatch at byte %d: got %#x want %#x", i, b[i], eB[i])
			}
		}
	}
}

func TestAnySliceAsByteSlice_Uint32AndUint64(t *testing.T) {
	u32 := []uint32{0x01020304, 0xAABBCCDD}
	e32L := []byte{0x04, 0x03, 0x02, 0x01, 0xDD, 0xCC, 0xBB, 0xAA}
	e32B := []byte{0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD}
	b32 := AnySliceAsByteSlice[uint32](unsafe.Pointer(&u32))
	if b32 == nil || len(b32) != len(u32)*4 {
		t.Fatalf("unexpected byte view for uint32 slice: len=%d", len(b32))
	}

	if isLittleEndian() {
		for i := range e32L {
			if b32[i] != e32L[i] {
				t.Fatalf("uint32 view mismatch at byte %d: got %#x want %#x", i, b32[i], e32L[i])
			}
		}
	} else {
		for i := range e32B {
			if b32[i] != e32B[i] {
				t.Fatalf("uint32 view mismatch at byte %d: got %#x want %#x", i, b32[i], e32B[i])
			}
		}
	}

	u64 := []uint64{0x0102030405060708, 0xFFEEDDCCBBAA9988}
	e64L := []byte{0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF}
	e64B := []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88}
	b64 := AnySliceAsByteSlice[uint64](unsafe.Pointer(&u64))
	if b64 == nil || len(b64) != len(u64)*8 {
		t.Fatalf("unexpected byte view for uint64 slice: len=%d", len(b64))
	}

	if isLittleEndian() {
		for i := range e64L {
			if b64[i] != e64L[i] {
				t.Fatalf("uint64 view mismatch at byte %d: got %#x want %#x", i, b64[i], e64L[i])
			}
		}
	} else {
		for i := range e64B {
			if b64[i] != e64B[i] {
				t.Fatalf("uint64 view mismatch at byte %d: got %#x want %#x", i, b64[i], e64B[i])
			}
		}
	}
}

func TestHashAnySliceAsByteSlice_Uint16DeterministicAndDistinct(t *testing.T) {
	seed := uint64(0xCAFEBABEDEADBEEF)

	var nilSlice []uint16
	empty := []uint16{}
	capOnly := make([]uint16, 0, 4)

	hn := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&nilSlice), seed)
	he := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&empty), seed)
	hc := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&capOnly), seed)
	if hn != he || he != hc {
		t.Fatalf("expected nil, empty and zero-len-with-cap to hash equal: got %#x %#x %#x", hn, he, hc)
	}

	a := []uint16{0x1122}
	b := []uint16{0x1122}
	c := []uint16{0x2211}

	ha := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&a), seed)
	hb := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&b), seed)
	hc2 := HashAnySliceAsByteSlice[uint16](unsafe.Pointer(&c), seed)

	// determinism
	if ha != hb {
		t.Fatalf("non-deterministic: same uint16 slice produced different hashes %#x %#x", ha, hb)
	}
	// different content -> very likely different hash
	if ha == hc2 {
		t.Fatalf("different uint16 slices produced same hash %#x == %#x", ha, hc2)
	}
}

func TestHashAnySliceAsByteSlice_Uint32AndUint64(t *testing.T) {
	seed := uint64(0x1234567890ABCDEF)

	u32a := []uint32{0x01020304, 0xAABBCCDD}
	u32b := []uint32{0x01020304, 0xAABBCCDD}
	u32c := []uint32{0x04030201, 0xDDCCBBAA}

	h1 := HashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32a), seed)
	h2 := HashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32b), seed)
	h3 := HashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32c), seed)

	if h1 != h2 {
		t.Fatalf("non-deterministic uint32 slice hash: %#x != %#x", h1, h2)
	}
	if h1 == h3 {
		t.Fatalf("different uint32 slices produced same hash %#x == %#x", h1, h3)
	}

	u64a := []uint64{0x0102030405060708, 0xFFEEDDCCBBAA9988}
	u64b := []uint64{0x0102030405060708, 0xFFEEDDCCBBAA9988}
	u64c := []uint64{0x0807060504030201, 0x8899AABBCCDDEEFF}

	hu1 := HashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64a), seed)
	hu2 := HashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64b), seed)
	hu3 := HashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64c), seed)

	if hu1 != hu2 {
		t.Fatalf("non-deterministic uint64 slice hash: %#x != %#x", hu1, hu2)
	}
	if hu1 == hu3 {
		t.Fatalf("different uint64 slices produced same hash %#x == %#x", hu1, hu3)
	}
}
