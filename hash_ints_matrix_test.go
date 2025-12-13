package set3

import (
	"fmt"
	"math/rand"
	"testing"
	"unsafe"
)

// TestHashSmallAndSampledIntDomains verifies exhaustively for small domains (bool,
// 8-bit and 16-bit integer types) that hashX(x) is deterministic and
// that the number of distinct outputs equals the domain size. For
// 32-bit and 64-bit integer types we perform a large sampled test (2M unique
// inputs) due to the impracticality of exhaustive enumeration.
func TestHashSmallAndSampledIntDomains(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long hash types test in short mode")
	}

	seeds := []uint64{0, 1, 12345, 0xfffffffffffffffe, 0xffffffffffffffff}
	const samples = 2_000_000

	for _, seed := range seeds {
		t.Run(fmt.Sprintf("seed_%x/bool", seed), func(t *testing.T) {
			seen := make(map[uint64]struct{}, 2)
			for _, b := range []bool{false, true} {
				v1 := hashBool(unsafe.Pointer(&b), seed)
				v2 := hashBool(unsafe.Pointer(&b), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashBool for %v: %x != %x", seed, b, v1, v2)
				}
				seen[v1] = struct{}{}
			}
			if len(seen) != 2 {
				t.Fatalf("seed %x: expected 2 distinct outputs for bool, got %d", seed, len(seen))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/uint8", seed), func(t *testing.T) {
			seen := make(map[uint64]struct{}, 256)
			for i := 0; i < 256; i++ {
				b := uint8(i)
				v1 := hashUint8(unsafe.Pointer(&b), seed)
				v2 := hashUint8(unsafe.Pointer(&b), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashUint8 for %d: %x != %x", seed, i, v1, v2)
				}
				seen[v1] = struct{}{}
			}
			if len(seen) != 256 {
				t.Fatalf("seed %x: expected 256 distinct outputs for uint8, got %d", seed, len(seen))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/int8", seed), func(t *testing.T) {
			seen := make(map[uint64]struct{}, 256)
			for i := 0; i < 256; i++ {
				b := int8(uint8(i))
				v1 := hashInt8(unsafe.Pointer(&b), seed)
				v2 := hashInt8(unsafe.Pointer(&b), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashInt8 for %d: %x != %x", seed, b, v1, v2)
				}
				seen[v1] = struct{}{}
			}
			if len(seen) != 256 {
				t.Fatalf("seed %x: expected 256 distinct outputs for int8, got %d", seed, len(seen))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/uint16", seed), func(t *testing.T) {
			seen := make(map[uint64]struct{}, 65536)
			for i := 0; i < 65536; i++ {
				v := uint16(i)
				v1 := hashUint16(unsafe.Pointer(&v), seed)
				v2 := hashUint16(unsafe.Pointer(&v), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashUint16 for %d: %x != %x", seed, i, v1, v2)
				}
				seen[v1] = struct{}{}
			}
			if len(seen) != 65536 {
				t.Fatalf("seed %x: expected 65536 distinct outputs for uint16, got %d", seed, len(seen))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/int16", seed), func(t *testing.T) {
			seen := make(map[uint64]struct{}, 65536)
			for i := 0; i < 65536; i++ {
				v := int16(uint16(i))
				v1 := hashInt16(unsafe.Pointer(&v), seed)
				v2 := hashInt16(unsafe.Pointer(&v), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashInt16 for %d: %x != %x", seed, v, v1, v2)
				}
				seen[v1] = struct{}{}
			}
			if len(seen) != 65536 {
				t.Fatalf("seed %x: expected 65536 distinct outputs for int16, got %d", seed, len(seen))
			}
		})

		// Sampled tests for 32-bit types: exhaustive enumeration is impractical,
		// so we sample a large set of unique inputs and require determinism and
		// that sampled outputs are distinct (no collisions among sampled set).
		t.Run(fmt.Sprintf("seed_%x/uint32_sampled", seed), func(t *testing.T) {
			seenInputs := make(map[uint32]struct{}, samples)
			r := rand.New(rand.NewSource(42 + int64(seed)))
			inputs := make([]uint32, 0, samples)
			for len(inputs) < samples {
				x := r.Uint32()
				if _, ok := seenInputs[x]; ok {
					continue
				}
				seenInputs[x] = struct{}{}
				inputs = append(inputs, x)
			}
			seenOut := make(map[uint64]struct{}, samples)
			for _, x := range inputs {
				v1 := hashUint32(unsafe.Pointer(&x), seed)
				v2 := hashUint32(unsafe.Pointer(&x), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashUint32 for %v: %x != %x", seed, x, v1, v2)
				}
				seenOut[v1] = struct{}{}
			}
			if len(seenOut) != len(inputs) {
				t.Fatalf("seed %x: expected %d distinct outputs for sampled uint32 inputs, got %d", seed, len(inputs), len(seenOut))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/int32_sampled", seed), func(t *testing.T) {
			seenInputs := make(map[int32]struct{}, samples)
			r := rand.New(rand.NewSource(99 + int64(seed)))
			inputs := make([]int32, 0, samples)
			for len(inputs) < samples {
				x := int32(r.Uint32())
				if _, ok := seenInputs[x]; ok {
					continue
				}
				seenInputs[x] = struct{}{}
				inputs = append(inputs, x)
			}
			seenOut := make(map[uint64]struct{}, samples)
			for _, x := range inputs {
				v1 := hashInt32(unsafe.Pointer(&x), seed)
				v2 := hashInt32(unsafe.Pointer(&x), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashInt32 for %v: %x != %x", seed, x, v1, v2)
				}
				seenOut[v1] = struct{}{}
			}
			if len(seenOut) != len(inputs) {
				t.Fatalf("seed %x: expected %d distinct outputs for sampled int32 inputs, got %d", seed, len(inputs), len(seenOut))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/uint64_sampled", seed), func(t *testing.T) {
			seenInputs := make(map[uint64]struct{}, samples)
			r := rand.New(rand.NewSource(123 + int64(seed)))
			inputs := make([]uint64, 0, samples)
			for len(inputs) < samples {
				x := r.Uint64()
				if _, ok := seenInputs[x]; ok {
					continue
				}
				seenInputs[x] = struct{}{}
				inputs = append(inputs, x)
			}
			seenOut := make(map[uint64]struct{}, samples)
			for _, x := range inputs {
				v1 := hashUint64(unsafe.Pointer(&x), seed)
				v2 := hashUint64(unsafe.Pointer(&x), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashUint64 for %v: %x != %x", seed, x, v1, v2)
				}
				seenOut[v1] = struct{}{}
			}
			if len(seenOut) != len(inputs) {
				t.Fatalf("seed %x: expected %d distinct outputs for sampled uint64 inputs, got %d", seed, len(inputs), len(seenOut))
			}
		})

		t.Run(fmt.Sprintf("seed_%x/int64_sampled", seed), func(t *testing.T) {
			seenInputs := make(map[int64]struct{}, samples)
			r := rand.New(rand.NewSource(321 + int64(seed)))
			inputs := make([]int64, 0, samples)
			for len(inputs) < samples {
				x := int64(r.Uint64())
				if _, ok := seenInputs[x]; ok {
					continue
				}
				seenInputs[x] = struct{}{}
				inputs = append(inputs, x)
			}
			seenOut := make(map[uint64]struct{}, samples)
			for _, x := range inputs {
				v1 := hashInt64(unsafe.Pointer(&x), seed)
				v2 := hashInt64(unsafe.Pointer(&x), seed)
				if v1 != v2 {
					t.Fatalf("seed %x: non-deterministic hashInt64 for %v: %x != %x", seed, x, v1, v2)
				}
				seenOut[v1] = struct{}{}
			}
			if len(seenOut) != len(inputs) {
				t.Fatalf("seed %x: expected %d distinct outputs for sampled int64 inputs, got %d", seed, len(inputs), len(seenOut))
			}
		})
	}
}
