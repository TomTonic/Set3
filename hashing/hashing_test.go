// nolint:gosec // Test file: intentional unsafe.Pointer use and weak RNG for deterministic testing
package hashing

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"
)

func TestSplitmix64Deterministic(t *testing.T) {
	inputs := []uint64{
		0,
		1,
		2,
		12345,
		0xDEADBEEFCAFEBABE,
		0xFFFFFFFFFFFFFFFF,
	}
	for _, x := range inputs {
		a := Splitmix64(x)
		b := Splitmix64(x)
		if a != b {
			t.Fatalf("Splitmix64 not deterministic for input %v: %v != %v", x, a, b)
		}
	}
}

func TestSplitmix64NoCollisionsSmallRange(t *testing.T) {
	const N = 65536
	seen := make(map[uint64]struct{}, N)
	for i := range N {
		v := Splitmix64(uint64(i))
		if _, ok := seen[v]; ok {
			t.Fatalf("collision at input %d produced value %#x", i, v)
		}
		seen[v] = struct{}{}
	}
}

// TestHashBytesBlockNoCollisionsSmallRange ensures HashBytesBlock does not
// collide over a modest sequence of 64-bit words encoded as little-endian
// byte slices.
func TestHashBytesBlockNoCollisionsSmallRange(t *testing.T) {
	const N = 65536
	seen := make(map[uint64]struct{}, N)
	const maxLen = 73
	buf := make([]byte, maxLen+8)
	for i := uint64(0); i < N; i++ {
		binary.LittleEndian.PutUint64(buf, i)
		l := int(i % maxLen)
		b := buf[:(l + 8)]
		for j := 0; j < l; j++ {
			b[j+8] = byte(j)
		}
		v := HashBytesBlock(0x1234567890ABCDEF, buf)
		if _, ok := seen[v]; ok {
			t.Fatalf("collision at input %d produced value %#x", i, v)
		}
		seen[v] = struct{}{}
	}
}

// TestHashBytesBlockUniformDistribution performs an expensive uniformity
// check for HashBytesBlock. It is skipped in -short mode.
func TestHashBytesBlockUniformDistribution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = 1 << 20
	const samplesPerBucket = 128
	N := uint64(buckets * samplesPerBucket)

	counts := make([]uint32, buckets)
	var sum uint64
	const maxLen = 73
	buf := make([]byte, maxLen+8)
	for i := uint64(0); i < N; i++ {
		binary.LittleEndian.PutUint64(buf, i)
		l := int(i % maxLen)
		b := buf[:(l + 8)]
		for j := 0; j < l; j++ {
			b[j+8] = byte(j + 17)
		}
		v := HashBytesBlock(0xC0FFEE1234567890, b)
		idx := int(v & uint64(buckets-1))
		counts[idx]++
		sum++
	}

	mean := float64(sum) / float64(buckets)
	var sqsum float64
	var maxDev float64
	for _, c := range counts {
		d := float64(c) - mean
		sqsum += d * d
		if math.Abs(d)/mean > maxDev {
			maxDev = math.Abs(d) / mean
		}
	}
	obsStd := math.Sqrt(sqsum / float64(buckets))
	obsRelStd := obsStd / mean
	expectedRelStd := math.Sqrt((1.0 - 1.0/float64(buckets)) / mean)

	t.Logf("HashBytesBlock uniformity: samples=%d buckets=%d mean=%.3f obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f",
		N, buckets, mean, obsRelStd, expectedRelStd, maxDev)

	if obsRelStd > 3.0*expectedRelStd {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (3x expected)", obsRelStd, 3.0*expectedRelStd)
	}
	if maxDev > 2.0 {
		t.Fatalf("a bucket deviated too much from mean: maxRelDev=%.6f", maxDev)
	}
}

// TestHashBytesBlockSlicesEdgeCases verifies HashBytesBlock behaviour on
// a collection of small edge-case slices: nil, empty, single element,
// exactly one 8-byte block, and tails of length 0..7. It asserts
// determinism and guards against trivial collisions among these cases.
func TestHashBytesBlockSlicesEdgeCases(t *testing.T) {
	seed := uint64(0xDEADBEEFCAFEBABE)

	var nilSlice []byte
	empty := []byte{}

	// nil and empty must produce identical hashes
	if HashBytesBlock(seed, nilSlice) != HashBytesBlock(seed, empty) {
		t.Fatalf("nil and empty slices produced different hashes")
	}

	// single element should differ from empty
	one := []byte{0x5A}
	if HashBytesBlock(seed, one) == HashBytesBlock(seed, empty) {
		t.Fatalf("single-element hash equals empty-slice hash")
	}

	// one full 8-byte block
	block := make([]byte, 8)
	for i := 0; i < 8; i++ {
		block[i] = byte(i + 1)
	}
	bhash := HashBytesBlock(seed, block)
	// deterministic
	if bhash != HashBytesBlock(seed, block) {
		t.Fatalf("HashBytesBlock not deterministic for block")
	}

	// tails 0..7 should yield distinct outputs (very unlikely to collide)
	tailResults := make(map[int]uint64)
	for tail := 0; tail <= 7; tail++ {
		buf := make([]byte, tail)
		for i := 0; i < tail; i++ {
			buf[i] = byte(10 + i)
		}
		tailResults[tail] = HashBytesBlock(seed, buf)
	}
	seen := make(map[uint64]int)
	for tail, v := range tailResults {
		if prev, ok := seen[v]; ok {
			t.Fatalf("tail collision: lengths %d and %d produced same hash %#x", prev, tail, v)
		}
		seen[v] = tail
	}

	// one block + tails 0..7 should also be distinct
	seen2 := make(map[uint64]int)
	for tail := 0; tail <= 7; tail++ {
		buf := make([]byte, 8+tail)
		copy(buf, block)
		for i := 0; i < tail; i++ {
			buf[8+i] = byte(100 + i)
		}
		v := HashBytesBlock(seed, buf)
		if prev, ok := seen2[v]; ok {
			t.Fatalf("block+tail collision: lengths %d and %d produced same hash %#x", prev, tail, v)
		}
		seen2[v] = tail
	}

	// ensure block hash differs from empty and one-element
	if bhash == HashBytesBlock(seed, empty) || bhash == HashBytesBlock(seed, one) {
		t.Fatalf("block hash unexpectedly equals empty or single-element hash")
	}
}

// checkRuntimeHasher is a small helper that verifies determinism and
// reasonable distinction for two different values of a comparable type.
func checkRuntimeHasher[T comparable](t *testing.T, seed uint64, name string, v1, v2 T) {
	t.Helper()
	h := MakeRuntimeHasher[T](seed)
	a := h.Hash(v1)
	b := h.Hash(v1)
	if a != b {
		t.Fatalf("%s: non-deterministic for seed %x: %x != %x", name, seed, a, b)
	}
	// If v1 and v2 are different, we expect hashes usually differ.
	if v1 != v2 {
		c := h.Hash(v2)
		if a == c {
			t.Fatalf("%s: different values produced same hash for seed %x: %x", name, seed, a)
		}
	}
}

// TestHashEntrypoints exercises a wide set of entry points
// for several seeds to ensure determinism and basic non-triviality.
func TestHashEntrypoints(t *testing.T) {
	seeds := []uint64{0, 1, 0x12345678abcdef}

	for _, s := range seeds {
		// Direct string hashing checks (deterministic + different from empty)
		sEmpty := ""
		sHello := "hello"
		h1 := HashString(unsafe.Pointer(&sEmpty), s)
		h2 := HashString(unsafe.Pointer(&sEmpty), s)
		if h1 != h2 {
			t.Fatalf("HashString non-deterministic for seed %x: %x != %x", s, h1, h2)
		}
		hHello := HashString(unsafe.Pointer(&sHello), s)
		if hHello == h1 {
			t.Fatalf("HashString produced same value for empty and \"hello\" for seed %x: %x", s, hHello)
		}

		// Wide type coverage using MakeRuntimeHasher
		type MyInt int
		type MyArray [8]byte

		checkRuntimeHasher[uint8](t, s, "uint8", 0x7f, 0x80)
		checkRuntimeHasher[int8](t, s, "int8", -7, 7)
		checkRuntimeHasher[uint16](t, s, "uint16", 0x1337, 0x1338)
		checkRuntimeHasher[int16](t, s, "int16", -1337, 1337)
		checkRuntimeHasher[uint32](t, s, "uint32", 0xdeadbeef, 0xdeadbeee)
		checkRuntimeHasher[int32](t, s, "int32", -123456, 123456)
		checkRuntimeHasher[uint64](t, s, "uint64", 0xfeedfacecafebeef, 0xfeedfacecafebeee)
		checkRuntimeHasher[int64](t, s, "int64", int64(-9223372036854775807), int64(1))
		checkRuntimeHasher[uint](t, s, "uint", uint(1234567890), uint(1234567891))
		checkRuntimeHasher[int](t, s, "int", int(-42), int(42))
		checkRuntimeHasher[uintptr](t, s, "uintptr", uintptr(0xdeadbeef), uintptr(0xdeadbeee))
		checkRuntimeHasher[bool](t, s, "bool", true, false)
		checkRuntimeHasher[float32](t, s, "float32", float32(3.14159), float32(2.71828))
		checkRuntimeHasher[float64](t, s, "float64", float64(-2.718281828), float64(3.1415926535))
		checkRuntimeHasher[string](t, s, "string", "hello world", "goodbye")

		checkRuntimeHasher[[0]byte](t, s, "[0]byte", [0]byte{}, [0]byte{})
		checkRuntimeHasher[[1]byte](t, s, "[1]byte", [1]byte{1}, [1]byte{2})
		checkRuntimeHasher[[8]byte](t, s, "[8]byte", [8]byte{1, 2, 3, 4, 5, 6, 7, 8}, [8]byte{8, 7, 6, 5, 4, 3, 2, 1})
		checkRuntimeHasher[[16]byte](t, s, "[16]byte", [16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, [16]byte{})

		// Named types exercising reflect fallback
		checkRuntimeHasher[MyInt](t, s, "MyInt", MyInt(7), MyInt(8))
		checkRuntimeHasher[MyArray](t, s, "MyArray", MyArray{1, 2, 3, 4, 5, 6, 7, 8}, MyArray{8, 7, 6, 5, 4, 3, 2, 1})

		// Additional cases: fixed-size int array as raw bytes
		checkRuntimeHasher[[0]int](t, s, "[0]int", [0]int{}, [0]int{})
		checkRuntimeHasher[[1]int](t, s, "[1]int", [1]int{1}, [1]int{2})
		checkRuntimeHasher[[2]int](t, s, "[2]int", [2]int{1, 2}, [2]int{2, 1})
		checkRuntimeHasher[[3]int](t, s, "[3]int", [3]int{1, 2, 3}, [3]int{3, 2, 1})
	}
}
