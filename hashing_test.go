package set3

import (
	"encoding/binary"
	"math"
	"math/rand"
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
		a := splitmix64(x)
		b := splitmix64(x)
		if a != b {
			t.Fatalf("splitmix64 not deterministic for input %v: %v != %v", x, a, b)
		}
	}
}

func TestSplitmix64NoCollisionsSmallRange(t *testing.T) {
	const N = 65536
	seen := make(map[uint64]struct{}, N)
	for i := 0; i < N; i++ {
		v := splitmix64(uint64(i))
		if _, ok := seen[v]; ok {
			t.Fatalf("collision at input %d produced value %#x", i, v)
		}
		seen[v] = struct{}{}
	}
}

// TestSplitmix64UniformDistribution performs a long-running uniformity
// check for splitmix64. It hashes a large number of sequential inputs
// and accumulates counts into 2^20 buckets. The test verifies that the
// observed coefficient of variation (relative standard deviation) across
// buckets is on the same order as the statistical expectation for a
// multinomial distribution. This test is intentionally expensive and is
// skipped when `go test -short` is used.
func TestSplitmix64UniformDistribution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = 1 << 22 // 2^22 buckets
	// samplesPerBucket controls how many samples fall on average into
	// each bucket. Choose 128 for a reasonable tradeoff between runtime
	// and statistical stability: mean ~= 128, expected rel stddev ~= 1/sqrt(128).
	const samplesPerBucket = 128
	N := uint64(buckets * samplesPerBucket)

	counts := make([]uint32, buckets)
	for i := range N {
		v := splitmix64(i)
		idx := int(v & uint64(buckets-1))
		counts[idx]++
	}

	expected := float64(samplesPerBucket)
	// compute observed variance
	var sqsum float64
	var maxDev float64
	var maxDevIdx int = -1
	for i, c := range counts {
		d := float64(c) - expected
		sqsum += d * d
		if math.Abs(d)/expected > maxDev {
			maxDev = math.Abs(d) / expected
			maxDevIdx = i
		}
	}
	obsStd := math.Sqrt(sqsum / float64(buckets))
	obsRelStd := obsStd / expected

	// expected relative stddev for multinomial distribution ≈ 1/sqrt(expected)
	expectedRelStd := math.Sqrt((1.0 - 1.0/float64(buckets)) / expected)

	t.Logf("splitmix64 uniformity: samples=%d buckets=%d mean=%.3f obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f",
		N, buckets, expected, obsRelStd, expectedRelStd, maxDev)

	// Allow some slack: require observed rel std to be within 1.01x expected.
	if obsRelStd > 1.01*expectedRelStd {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (1.01x expected)", obsRelStd, 1.01*expectedRelStd)
	}

	// Also ensure no single bucket deviates excessively (e.g. > 50%). This
	// guards against pathological clustering even if the overall stddev is ok.
	if maxDev > 0.5 {
		t.Fatalf("a bucket deviated too much from mean: maxRelDev=%.6f at index %d", maxDev, maxDevIdx)
	}
}

// TestHashBytesBlockNoCollisionsSmallRange ensures hashBytesBlock does not
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
		v := hashBytesBlock(0x1234567890ABCDEF, buf)
		if _, ok := seen[v]; ok {
			t.Fatalf("collision at input %d produced value %#x", i, v)
		}
		seen[v] = struct{}{}
	}
}

// TestHashBytesBlockUniformDistribution performs an expensive uniformity
// check for hashBytesBlock similar to TestSplitmix64UniformDistribution.
// It is skipped in -short mode.
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
		v := hashBytesBlock(0xC0FFEE1234567890, b)
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

	t.Logf("hashBytesBlock uniformity: samples=%d buckets=%d mean=%.3f obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f",
		N, buckets, mean, obsRelStd, expectedRelStd, maxDev)

	if obsRelStd > 3.0*expectedRelStd {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (3x expected)", obsRelStd, 3.0*expectedRelStd)
	}
	if maxDev > 2.0 {
		t.Fatalf("a bucket deviated too much from mean: maxRelDev=%.6f", maxDev)
	}
}

// TestHashBytesBlockSlicesEdgeCases verifies hashBytesBlock behaviour on
// a collection of small edge-case slices: nil, empty, single element,
// exactly one 8-byte block, and tails of length 0..7. It asserts
// determinism and guards against trivial collisions among these cases.
func TestHashBytesBlockSlicesEdgeCases(t *testing.T) {
	seed := uint64(0xDEADBEEFCAFEBABE)

	var nilSlice []byte
	empty := []byte{}

	// nil and empty must produce identical hashes
	if hashBytesBlock(seed, nilSlice) != hashBytesBlock(seed, empty) {
		t.Fatalf("nil and empty slices produced different hashes")
	}

	// single element should differ from empty
	one := []byte{0x5A}
	if hashBytesBlock(seed, one) == hashBytesBlock(seed, empty) {
		t.Fatalf("single-element hash equals empty-slice hash")
	}

	// one full 8-byte block
	block := make([]byte, 8)
	for i := 0; i < 8; i++ {
		block[i] = byte(i + 1)
	}
	bhash := hashBytesBlock(seed, block)
	// deterministic
	if bhash != hashBytesBlock(seed, block) {
		t.Fatalf("hashBytesBlock not deterministic for block")
	}

	// tails 0..7 should yield distinct outputs (very unlikely to collide)
	tailResults := make(map[int]uint64)
	for tail := 0; tail <= 7; tail++ {
		buf := make([]byte, tail)
		for i := 0; i < tail; i++ {
			buf[i] = byte(10 + i)
		}
		tailResults[tail] = hashBytesBlock(seed, buf)
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
		v := hashBytesBlock(seed, buf)
		if prev, ok := seen2[v]; ok {
			t.Fatalf("block+tail collision: lengths %d and %d produced same hash %#x", prev, tail, v)
		}
		seen2[v] = tail
	}

	// ensure block hash differs from empty and one-element
	if bhash == hashBytesBlock(seed, empty) || bhash == hashBytesBlock(seed, one) {
		t.Fatalf("block hash unexpectedly equals empty or single-element hash")
	}
}

// TestMakeRuntimeHasherBasic verifies that MakeRuntimeHasher constructs a
// RuntimeHasher for a variety of comparable types and that the resulting
// hasher can compute a deterministic hash for a sample value without panicking.
func TestMakeRuntimeHasherBasic(t *testing.T) {
	seed := uint64(42)

	t.Run("uint8", func(t *testing.T) {
		h := MakeRuntimeHasher[uint8](seed)
		a := h.Hash(uint8(0x7f))
		b := h.Hash(uint8(0x7f))
		c := h.Hash(uint8(0x80))
		if a != b {
			t.Fatalf("non-deterministic hash for uint8: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uint8 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("int8", func(t *testing.T) {
		h := MakeRuntimeHasher[int8](seed)
		a := h.Hash(int8(-7))
		b := h.Hash(int8(-7))
		c := h.Hash(int8(7))
		if a != b {
			t.Fatalf("non-deterministic hash for int8: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different int8 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("bool", func(t *testing.T) {
		h := MakeRuntimeHasher[bool](seed)
		a := h.Hash(true)
		b := h.Hash(true)
		c := h.Hash(false)
		if a != b {
			t.Fatalf("non-deterministic hash for bool: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different bool values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("uint16", func(t *testing.T) {
		h := MakeRuntimeHasher[uint16](seed)
		a := h.Hash(uint16(0x1337))
		b := h.Hash(uint16(0x1337))
		c := h.Hash(uint16(0x1338))
		if a != b {
			t.Fatalf("non-deterministic hash for uint16: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uint16 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("int16", func(t *testing.T) {
		h := MakeRuntimeHasher[int16](seed)
		a := h.Hash(int16(-1337))
		b := h.Hash(int16(-1337))
		c := h.Hash(int16(1337))
		if a != b {
			t.Fatalf("non-deterministic hash for int16: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different int16 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("uint32", func(t *testing.T) {
		h := MakeRuntimeHasher[uint32](seed)
		a := h.Hash(uint32(0xdeadbeef))
		b := h.Hash(uint32(0xdeadbeef))
		c := h.Hash(uint32(0xdeadbeee))
		if a != b {
			t.Fatalf("non-deterministic hash for uint32: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uint32 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("int32", func(t *testing.T) {
		h := MakeRuntimeHasher[int32](seed)
		a := h.Hash(int32(-123456))
		b := h.Hash(int32(-123456))
		c := h.Hash(int32(123456))
		if a != b {
			t.Fatalf("non-deterministic hash for int32: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different int32 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("uint64", func(t *testing.T) {
		h := MakeRuntimeHasher[uint64](seed)
		a := h.Hash(uint64(0xfeedfacecafebeef))
		b := h.Hash(uint64(0xfeedfacecafebeef))
		c := h.Hash(uint64(0xfeedfacecafebeee))
		if a != b {
			t.Fatalf("non-deterministic hash for uint64: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uint64 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("int64", func(t *testing.T) {
		h := MakeRuntimeHasher[int64](seed)
		a := h.Hash(int64(-9223372036854775807))
		b := h.Hash(int64(-9223372036854775807))
		c := h.Hash(int64(1))
		if a != b {
			t.Fatalf("non-deterministic hash for int64: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different int64 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("uint", func(t *testing.T) {
		h := MakeRuntimeHasher[uint](seed)
		a := h.Hash(uint(1234567890))
		b := h.Hash(uint(1234567890))
		c := h.Hash(uint(1234567891))
		if a != b {
			t.Fatalf("non-deterministic hash for uint: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uint values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("int", func(t *testing.T) {
		h := MakeRuntimeHasher[int](seed)
		a := h.Hash(int(-42))
		b := h.Hash(int(-42))
		c := h.Hash(int(42))
		if a != b {
			t.Fatalf("non-deterministic hash for int: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different int values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("uintptr", func(t *testing.T) {
		h := MakeRuntimeHasher[uintptr](seed)
		a := h.Hash(uintptr(0xdeadbeef))
		b := h.Hash(uintptr(0xdeadbeef))
		c := h.Hash(uintptr(0xdeadbeee))
		if a != b {
			t.Fatalf("non-deterministic hash for uintptr: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different uintptr values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("float32", func(t *testing.T) {
		h := MakeRuntimeHasher[float32](seed)
		a := h.Hash(float32(3.14159))
		b := h.Hash(float32(3.14159))
		c := h.Hash(float32(2.71828))
		if a != b {
			t.Fatalf("non-deterministic hash for float32: %x != %x", a, b)
		}

		if a == c {
			t.Fatalf("different float32 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("float64", func(t *testing.T) {
		h := MakeRuntimeHasher[float64](seed)
		a := h.Hash(float64(-2.718281828))
		b := h.Hash(float64(-2.718281828))
		c := h.Hash(float64(3.1415926535))
		if a != b {
			t.Fatalf("non-deterministic hash for float64: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different float64 values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("string", func(t *testing.T) {
		h := MakeRuntimeHasher[string](seed)
		a := h.Hash("hello world")
		b := h.Hash("hello world")
		c := h.Hash("goodbye")
		if a != b {
			t.Fatalf("non-deterministic hash for string: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different string values produced same hash: %x == %x", a, c)
		}
	})

	// Arrays are comparable and exercise the array-byte path.
	t.Run("[0]byte", func(t *testing.T) {
		h := MakeRuntimeHasher[[0]byte](seed)
		a := h.Hash([0]byte{})
		b := h.Hash([0]byte{})
		if a != b {
			t.Fatalf("non-deterministic hash for [0]byte: %x != %x", a, b)
		}
	})
	t.Run("[1]byte", func(t *testing.T) {
		h := MakeRuntimeHasher[[1]byte](seed)
		a := h.Hash([1]byte{0x1})
		b := h.Hash([1]byte{0x1})
		c := h.Hash([1]byte{0x2})
		if a != b {
			t.Fatalf("non-deterministic hash for [1]byte: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different [1]byte values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("[8]byte", func(t *testing.T) {
		h := MakeRuntimeHasher[[8]byte](seed)
		a := h.Hash([8]byte{1, 2, 3, 4, 5, 6, 7, 8})
		b := h.Hash([8]byte{1, 2, 3, 4, 5, 6, 7, 8})
		c := h.Hash([8]byte{8, 7, 6, 5, 4, 3, 2, 1})
		if a != b {
			t.Fatalf("non-deterministic hash for [8]byte: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different [8]byte values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("[16]byte", func(t *testing.T) {
		h := MakeRuntimeHasher[[16]byte](seed)
		a := h.Hash([16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
		b := h.Hash([16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
		c := h.Hash([16]byte{})
		if a != b {
			t.Fatalf("non-deterministic hash for [16]byte: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different [16]byte values produced same hash: %x == %x", a, c)
		}
	})

	// Named types to exercise reflect fallback branches.
	type MyInt int
	type MyArray [8]byte
	t.Run("MyInt", func(t *testing.T) {
		h := MakeRuntimeHasher[MyInt](seed)
		a := h.Hash(MyInt(7))
		b := h.Hash(MyInt(7))
		c := h.Hash(MyInt(8))
		if a != b {
			t.Fatalf("non-deterministic hash for MyInt: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different MyInt values produced same hash: %x == %x", a, c)
		}
	})
	t.Run("MyArray", func(t *testing.T) {
		h := MakeRuntimeHasher[MyArray](seed)
		a := h.Hash(MyArray{1, 2, 3, 4, 5, 6, 7, 8})
		b := h.Hash(MyArray{1, 2, 3, 4, 5, 6, 7, 8})
		c := h.Hash(MyArray{8, 7, 6, 5, 4, 3, 2, 1})
		if a != b {
			t.Fatalf("non-deterministic hash for MyArray: %x != %x", a, b)
		}
		if a == c {
			t.Fatalf("different MyArray values produced same hash: %x == %x", a, c)
		}
	})
}

// TestHashBoolTwoValues ensures that hashBool produces exactly two
// distinct outputs across many random boolean inputs. We use a fixed RNG
// seed for reproducibility; the inputs are boolean as required.
func TestHashBoolTwoValues(t *testing.T) {
	const trials = 1000
	r := rand.New(rand.NewSource(42))
	seed := uint64(0x12345678abcdef)
	seen := make(map[uint64]struct{})
	for i := 0; i < trials; i++ {
		b := r.Intn(2) == 1
		v := hashBool(unsafe.Pointer(&b), seed)
		seen[v] = struct{}{}
	}
	if len(seen) != 2 {
		t.Fatalf("expected exactly 2 distinct hash values for boolean inputs, got %d", len(seen))
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
		h1 := hashString(unsafe.Pointer(&sEmpty), s)
		h2 := hashString(unsafe.Pointer(&sEmpty), s)
		if h1 != h2 {
			t.Fatalf("hashString non-deterministic for seed %x: %x != %x", s, h1, h2)
		}
		hHello := hashString(unsafe.Pointer(&sHello), s)
		if hHello == h1 {
			t.Fatalf("hashString produced same value for empty and \"hello\" for seed %x: %x", s, hHello)
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

func TestAnySliceAsByteSlice_EmptyAndCapOnly(t *testing.T) {
	var nilSlice []uint16
	empty := []uint16{}
	capOnly := make([]uint16, 0, 8)

	if anySliceAsByteSlice[uint16](unsafe.Pointer(&nilSlice)) != nil {
		t.Fatalf("expected nil for nil slice")
	}
	if anySliceAsByteSlice[uint16](unsafe.Pointer(&empty)) != nil {
		t.Fatalf("expected nil for empty slice")
	}
	if anySliceAsByteSlice[uint16](unsafe.Pointer(&capOnly)) != nil {
		t.Fatalf("expected nil for zero-len slice even if cap>0")
	}
}

func isLittleEndian() bool {
	var x uint16 = 1
	p := (*[2]byte)(unsafe.Pointer(&x))
	return p[0] == 1
}

func TestAnySliceAsByteSlice_Uint16Roundtrip(t *testing.T) {
	s := []uint16{0x1122, 0x3344, 0x5566}
	eL := []byte{0x22, 0x11, 0x44, 0x33, 0x66, 0x55}
	eB := []byte{0x11, 0x22, 0x33, 0x44, 0x55, 0x66}
	b := anySliceAsByteSlice[uint16](unsafe.Pointer(&s))
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
	b32 := anySliceAsByteSlice[uint32](unsafe.Pointer(&u32))
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
	b64 := anySliceAsByteSlice[uint64](unsafe.Pointer(&u64))
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

	hn := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&nilSlice), seed)
	he := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&empty), seed)
	hc := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&capOnly), seed)
	if hn != he || he != hc {
		t.Fatalf("expected nil, empty and zero-len-with-cap to hash equal: got %#x %#x %#x", hn, he, hc)
	}

	a := []uint16{0x1122}
	b := []uint16{0x1122}
	c := []uint16{0x2211}

	ha := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&a), seed)
	hb := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&b), seed)
	hc2 := hashAnySliceAsByteSlice[uint16](unsafe.Pointer(&c), seed)

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

	h1 := hashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32a), seed)
	h2 := hashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32b), seed)
	h3 := hashAnySliceAsByteSlice[uint32](unsafe.Pointer(&u32c), seed)

	if h1 != h2 {
		t.Fatalf("non-deterministic uint32 slice hash: %#x != %#x", h1, h2)
	}
	if h1 == h3 {
		t.Fatalf("different uint32 slices produced same hash %#x == %#x", h1, h3)
	}

	u64a := []uint64{0x0102030405060708, 0xFFEEDDCCBBAA9988}
	u64b := []uint64{0x0102030405060708, 0xFFEEDDCCBBAA9988}
	u64c := []uint64{0x0807060504030201, 0x8899AABBCCDDEEFF}

	hu1 := hashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64a), seed)
	hu2 := hashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64b), seed)
	hu3 := hashAnySliceAsByteSlice[uint64](unsafe.Pointer(&u64c), seed)

	if hu1 != hu2 {
		t.Fatalf("non-deterministic uint64 slice hash: %#x != %#x", hu1, hu2)
	}
	if hu1 == hu3 {
		t.Fatalf("different uint64 slices produced same hash %#x == %#x", hu1, hu3)
	}
}
