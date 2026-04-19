package set3

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
	for i := range N {
		v := splitmix64(uint64(i))
		if _, ok := seen[v]; ok {
			t.Fatalf("collision at input %d produced value %#x", i, v)
		}
		seen[v] = struct{}{}
	}
}

// Test64BitHasherUniformDistribution performs a long-running uniformity
// check for splitmix64. It hashes a large number of sequential inputs
// and accumulates counts into 2^20 buckets. The test verifies that the
// observed coefficient of variation (relative standard deviation) across
// buckets is on the same order as the statistical expectation for a
// multinomial distribution. This test is intentionally expensive and is
// skipped when `go test -short` is used.
/* func Test32BitHasherUniformDistribution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = 1 << 20 // 2^20 buckets = 1048576
	// samplesPerBucket controls how many samples fall on average into
	// each bucket. Choose 256 for a reasonable tradeoff between runtime
	// and statistical stability: mean = 256, expected rel stddev ~= 1/sqrt(256).
	const samplesPerBucket = 256
	N := uint64(buckets * samplesPerBucket)

	cprng := rtcompare.NewCPRNG(8192) // 8KB buffer

	countsHashes := make([]uint32, buckets)
	countsRands := make([]uint32, buckets)
	mask := uint64(buckets - 1)

	for range N {
		//r := getRealRandUint64() // perturb input to avoid any correlations
		r := cprng.Uint32()
		//r := i
		countsRands[uint64(r)&mask]++
		v := splitmix64(uint64(r)) // splitmix64 does not match test criteria --- IGNORE ---
		//v := wh32(r, 0xC0FFEE1234567890) // better than splitmix64
		countsHashes[v&mask]++
	}

	// compute observed variance

	mean := float64(samplesPerBucket)
	var sqsumHashes float64
	var maxDevHashes float64
	var maxDevIdx int = -1
	var sqsumRands float64
	var maxDevRands float64

	for i := range buckets {
		c := float64(countsHashes[i])
		d := float64(c) - mean
		sqsumHashes += d * d
		if math.Abs(d)/mean > maxDevHashes {
			maxDevHashes = math.Abs(d) / mean
			maxDevIdx = i
		}
		cR := float64(countsRands[i])
		dR := float64(cR) - mean
		sqsumRands += dR * dR
		if math.Abs(dR)/mean > maxDevRands {
			maxDevRands = math.Abs(dR) / mean
		}
	}
	obsVar := sqsumHashes / float64(buckets)
	obsStdDev := math.Sqrt(obsVar)
	rndVar := sqsumRands / float64(buckets)
	rndStdDev := math.Sqrt(rndVar)

	t.Logf("uniformity: samples=%d buckets=%d mean=%.3f obsStdDevHash=%.6f obsStdDevRands=%.6f maxDevHash=%.6f maxDevRands=%.6f",
		N, buckets, mean, obsStdDev, rndStdDev, maxDevHashes, maxDevRands)

	// Allow some slack: require observed rel std to be within 1.04x expected.
	if obsStdDev > 1.04*rndStdDev {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (1.04x expected)", obsStdDev, 1.04*rndStdDev)
	}

	// Also ensure no single bucket deviates excessively (e.g. > 35%). This
	// guards against pathological clustering even if the overall stddev is ok.
	if maxDevHashes > 0.35 {
		t.Fatalf("a bucket deviated too much from mean: maxDev=%.6f at index %d, while maxDevRands=%.6f", maxDevHashes, maxDevIdx, maxDevRands)
	}
} */

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

// This test ensures that all 32 bit inputs (0..math.MaxUint32) produce a uniform
// distribution on the lower 7 bits after hashing with wh32 or splitmix64. It is
// a long-running test and is skipped in -short mode.
/* func Test32BitHasherUniformDistribution_Lower7Bits(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = uint64(1) << 7 // 128 buckets
	const N = uint64(1) << 32
	const samplesPerBucket = N / buckets

	countsWh32 := make([]uint32, buckets)
	countsSplitmix := make([]uint32, buckets)
	const mask = uint64(buckets - 1)

	for i := range N {
		v1 := wh32(uint32(i), 0xC0FFEE1234567890)
		countsWh32[v1&mask]++
		v2 := splitmix64(uint64(i) ^ 0xC0FFEE1234567890)
		countsSplitmix[v2&mask]++
	}

	mean := float64(samplesPerBucket)
	var sqsumWh32 float64
	var maxDevWh32 float64
	var maxDevIdxWh32 int64 = -1
	var sqsumSplitmix float64
	var maxDevSplitmix float64
	var maxDevIdxSplitmix int64 = -1

	for i := range buckets {
		c1 := float64(countsWh32[i])
		d1 := float64(c1) - mean
		sqsumWh32 += d1 * d1
		if math.Abs(d1)/mean > maxDevWh32 {
			maxDevWh32 = math.Abs(d1) / mean
			maxDevIdxWh32 = int64(i)
		}
		c2 := float64(countsSplitmix[i])
		d2 := float64(c2) - mean
		sqsumSplitmix += d2 * d2
		if math.Abs(d2)/mean > maxDevSplitmix {
			maxDevSplitmix = math.Abs(d2) / mean
			maxDevIdxSplitmix = int64(i)
		}
	}
	expectedRelStd := math.Sqrt((1.0 - 1.0/float64(buckets)) / mean)

	obsStdWh32 := math.Sqrt(sqsumWh32 / float64(buckets))
	obsRelStdWh32 := obsStdWh32 / mean

	obsStdSplitmix := math.Sqrt(sqsumSplitmix / float64(buckets))
	obsRelStdSplitmix := obsStdSplitmix / mean

	t.Logf("lower 7 bits uniformity test: samples=%s buckets=%s mean=%s", Pow2String(uint64(N)), Pow2String(buckets), Pow2String(uint64(mean)))
	t.Logf("wh32       : obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f at idx %d",
		obsRelStdWh32, expectedRelStd, maxDevWh32, maxDevIdxWh32)
	t.Logf("splitmix64 : obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f at idx %d",
		obsRelStdSplitmix, expectedRelStd, maxDevSplitmix, maxDevIdxSplitmix)

	if obsRelStdWh32 > 1.03*expectedRelStd {
		t.Fatalf("wh32: observed relative stddev too large: got %.6f, want <= %.6f (3%% expected)", obsRelStdWh32, 3.0*expectedRelStd)
	}
	if maxDevWh32 > .003 {
		t.Fatalf("wh32: a bucket deviated too much from mean: maxRelDev=%.6f at index %d", maxDevWh32, maxDevIdxWh32)
	}

	if obsRelStdSplitmix > 1.03*expectedRelStd {
		t.Fatalf("splitmix64: observed relative stddev too large: got %.6f, want <= %.6f (3%% expected)", obsRelStdSplitmix, 3.0*expectedRelStd)
	}
	if maxDevSplitmix > .003 {
		t.Fatalf("splitmix64: a bucket deviated too much from mean: maxRelDev=%.6f at index %d", maxDevSplitmix, maxDevIdxSplitmix)
	}
}

// Pow2String returns "2^k" if n is an exact power of two (n>0), otherwise the
// decimal representation of n. Works on uint64 inputs.
func Pow2String(n uint64) string {
	if n == 0 {
		return "0"
	}
	if n&(n-1) == 0 { // power of two
		k := bits.TrailingZeros64(n)
		return fmt.Sprintf("2^%d", k)
	}
	return strconv.FormatUint(n, 10)
}

// This test compares the performance of splitmix64 vs wh32 on hashing 32-bit inputs.
// It is skipped in -short mode.
func Test32BitHasherPerformanceComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long performance test in short mode")
	}

	const N = 100_000_000

	const repeats = 5555
	const innerLoops = uint32(10_000)
	const expectedSpeedup = 0.1 // expect wh32 to be at least 10% faster than splitmix64
	const minConfidence = 0.95  // require at least 95% confidence

	timesWh32 := make([]float64, 0, repeats)
	timesSplitmix := make([]float64, 0, repeats)
	x := uint64(0)

	for range repeats {
		runtime.GC()
		prev := debug.SetGCPercent(-1)
		t1 := rtcompare.SampleTime()
		for i := range innerLoops {
			x ^= wh32(i, 0xC0FFEE1234567890)
		}
		t2 := rtcompare.SampleTime()
		_ = debug.SetGCPercent(prev)
		timesWh32 = append(timesWh32, float64(rtcompare.DiffTimeStamps(t1, t2))/float64(innerLoops))

		runtime.GC()
		_ = debug.SetGCPercent(-1)
		t3 := rtcompare.SampleTime()
		for i := range innerLoops {
			x ^= splitmix64(uint64(i) ^ 0xC0FFEE1234567890)
		}
		t4 := rtcompare.SampleTime()
		_ = debug.SetGCPercent(prev)
		timesSplitmix = append(timesSplitmix, float64(rtcompare.DiffTimeStamps(t3, t4))/float64(innerLoops))
	}

	mWh32 := rtcompare.QuickMedian(timesWh32)
	mSplitmix := rtcompare.QuickMedian(timesSplitmix)
	t.Logf("x=%d", x)
	t.Logf("median call (wh32)=%.1f ns, (splitmix64)=%.1f ns", mWh32, mSplitmix)

	if mSplitmix < mWh32 {
		t.Fatalf("expected wh32 to be faster: splitmix64=%.1f >= wh32=%.1f", mSplitmix, mWh32)
	}

	speedups := []float64{expectedSpeedup}
	results, err := rtcompare.CompareSamples(timesWh32, timesSplitmix, speedups, 10_000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	if len(results) < 1 {
		t.Fatalf("expected at least 1 result from CompareSamples, got %d", len(results))
	}
	for _, r := range results {
		t.Logf("Speedup ≥ %.2f%% → Confidence: %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
	res := results[0]
	if res.Confidence < minConfidence {
		t.Fatalf("expected confidence >= %.2f for speedup %.1f, got %.3f", minConfidence, res.RelativeSpeedupSampleAvsSampleB, res.Confidence)
	}
}

func TestCompareWh64Splitmix64(t *testing.T) {
	type stats struct {
		name       string
		cv         float64
		maxDevPct  float64
		collisions uint32
		avgHamming float64
		duration   time.Duration
	}

	const (
		N       = 1 << 21 // number of input values to hash
		Buckets = 3317    // number of buckets
		AvM     = 100_000 // samples for avalanche test
		seed    = 0xC0FFEE1234567890
	)

	run := func(name string, hf func(uint64, uint64) uint64) stats {
		start := time.Now()

		counts := make([]uint32, Buckets)
		seen := make(map[uint64]uint32)
		for i := range uint64(N) {
			h := hf(i, seed)
			counts[int(h&(Buckets-1))]++
			seen[h]++
		}

		// collisions
		coll := uint32(0)
		for _, v := range seen {
			if v > 1 {
				coll += v - 1
			}
		}

		// coefficient of variation
		mean := float64(N) / float64(Buckets)
		var sumsq float64
		maxDev := 0.0
		for _, c := range counts {
			diff := float64(c) - mean
			sumsq += diff * diff
			if dev := math.Abs(diff) / mean; dev > maxDev {
				maxDev = dev
			}
		}
		variance := sumsq / float64(Buckets)
		cv := math.Sqrt(variance) / mean

		// avalanche: flip one input bit and measure output hamming distance
		var totalHd int
		for i := range uint64(AvM) {
			x := i
			h1 := hf(x, seed)
			// flip one bit to test sensitivity
			h2 := hf(x^(1<<uint64(i&0x3F)), seed)
			hammingDistance := bits.OnesCount64(h1 ^ h2)
			totalHd += hammingDistance
		}
		avgHd := float64(totalHd) / float64(AvM)

		return stats{
			name:       name,
			cv:         cv,
			maxDevPct:  maxDev * 100,
			collisions: coll,
			avgHamming: avgHd,
			duration:   time.Since(start),
		}
	}

	s1 := run("splitmix64", func(v, seed uint64) uint64 { return splitmix64(v ^ seed) })
	s2 := run("wh64      ", func(v, seed uint64) uint64 { return wh64(v, seed) })

	fmt.Println("Hash comparison (N=", N, " buckets=", Buckets, "):")
	for _, s := range []stats{s1, s2} {
		fmt.Printf("%s: CV=%.6g maxDev=%.3f%% collisions=%d avgHamming=%.2f time=%v\n",
			s.name, s.cv, s.maxDevPct, s.collisions, s.avgHamming, s.duration)
	}
} */

// FilteredNumbers returns a channel that emits natural numbers starting at 2
// that are prime, a power of two, or a multiple of 10. It emits exactly `count` values.
func FilteredNumbers(count uint64) <-chan uint64 {
	ch := make(chan uint64)
	go func() {
		defer close(ch)
		if count <= 0 {
			return
		}
		emitted := uint64(0)
		for n := uint64(2); emitted < count; n++ {
			if n%100 == 0 || (n <= 500 && n%50 == 0) || (n <= 100 && n%10 == 0) || isPowerOfTwo(n) || isPrime(n) {
				ch <- n
				emitted++
			}
		}
	}()
	return ch
}

func isPrimeNaive(n uint64) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for i := uint64(3); i*i <= n; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

func isPowerOfTwo(n uint64) bool {
	return n > 0 && (n&(n-1)) == 0
}
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

/* func TestCompareWh64Splitmix64_MultiBuckets(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-bucket hash comparison in short mode")
	}

	const (
		samplesPerBucket = uint64(1024)
		avm              = uint64(10_000) // avalanche samples per bucket case
		seed             = 0xC0FFEE1234567890
		bucketCases      = uint64(245)
	)

	// prepare result slices
	var (
		cvSMresults       []float64
		cvWHresults       []float64
		maxDevSMresults   []float64
		maxDevWHresults   []float64
		collSMresults     []float64
		collWHresults     []float64
		avgHdSMresults    []float64
		avgHdWHresults    []float64
		durationSMresults []float64
		durationWHresults []float64
	)

	bucketsCh := FilteredNumbers(bucketCases)
	for b := range bucketsCh {
		if b <= 0 {
			continue
		}
		buckets := b
		N := uint64(buckets * samplesPerBucket)

		run := func(hf func(uint64, uint64) uint64) (cv, maxDevPct, collF, avgHdDFO, durSec float64) {
			start := time.Now()

			counts := make([]uint64, buckets)
			seen := make(map[uint64]uint32)

			for i := range N {
				h := hf(i, seed)
				idx := int(h % uint64(buckets))
				counts[idx]++
				seen[h]++
			}

			// collisions
			var coll uint64
			for _, v := range seen {
				if v > 1 {
					coll += uint64(v - 1)
				}
			}

			mean := float64(N) / float64(buckets)
			var sumsq float64
			var maxDev float64
			for _, c := range counts {
				diff := float64(c) - mean
				sumsq += diff * diff
				if dev := math.Abs(diff) / mean; dev > maxDev {
					maxDev = dev
				}
			}
			variance := sumsq / float64(buckets)
			cvVal := math.Sqrt(variance) / mean

			// avalanche: flip a few different low bits to test sensitivity
			var totalHd uint64
			for i := range avm {
				x := i * buckets
				h1 := hf(x, seed)
				h2 := hf(x^(uint64(1)<<(uint(i)&0x3F)), seed)
				totalHd += uint64(bits.OnesCount64(h1 ^ h2))
			}
			avgHd := float64(totalHd) / float64(avm)
			avgHdDFO = math.Abs(avgHd - 32.0)

			return cvVal, maxDev * 100.0, float64(coll), avgHdDFO, time.Since(start).Seconds()
		}

		cvS, maxDevS, collS, avgHdS, durS := run(func(v, seed uint64) uint64 { return splitmix64(v ^ seed) })
		cvW, maxDevW, collW, avgHdW, durW := run(func(v, seed uint64) uint64 { return wh64(v, seed) })

		// append results
		cvSMresults = append(cvSMresults, cvS)
		cvWHresults = append(cvWHresults, cvW)
		maxDevSMresults = append(maxDevSMresults, maxDevS)
		maxDevWHresults = append(maxDevWHresults, maxDevW)
		collSMresults = append(collSMresults, collS)
		collWHresults = append(collWHresults, collW)
		avgHdSMresults = append(avgHdSMresults, avgHdS)
		avgHdWHresults = append(avgHdWHresults, avgHdW)
		durationSMresults = append(durationSMresults, durS)
		durationWHresults = append(durationWHresults, durW)

		t.Logf("buckets=%d N=%d\nsplitmix(cv=%.6g,maxDev=%.3f%%,coll=%.0f,avgHd=%.2f,dur=%.3fs)\n    wh64(cv=%.6g,maxDev=%.3f%%,coll=%.0f,avgHd=%.2f,dur=%.3fs)",
			buckets, N, cvS, maxDevS, collS, avgHdS, durS, cvW, maxDevW, collW, avgHdW, durW)
	}

	// Log summary lengths for inspection
	// t.Logf("summary lens: buckets=%d cvSM=%d cvWH=%d", bucketCases, len(cvSMresults), len(cvWHresults))
	// t.Logf("cvSM=%v", cvSMresults)
	// t.Logf("cvWH=%v", cvWHresults)
	// t.Logf("maxDevSM=%v", maxDevSMresults)
	// t.Logf("maxDevWH=%v", maxDevWHresults)
	// t.Logf("collSM=%v", collSMresults)
	// t.Logf("collWH=%v", collWHresults)
	// t.Logf("avgHdSM=%v", avgHdSMresults)
	// t.Logf("avgHdWH=%v", avgHdWHresults)
	// t.Logf("durSM=%v", durationSMresults)
	// t.Logf("durWH=%v", durationWHresults)

	// non-failing test; results are for manual inspection / further assertions

	// Pairwise statistical comparisons using rtcompare
	relativeGains := []float64{-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20}
	iterations := 10_000

	compare := func(name string, a, b []float64) {
		results, err := rtcompare.CompareSamples(a, b, relativeGains, uint64(iterations))
		if err != nil {
			t.Fatalf("CompareSamples failed for %s: %v", name, err)
		}
		if len(results) < 1 {
			t.Fatalf("expected at least 1 result from CompareSamples for %s", name)
		}
		for _, r := range results {
			t.Logf("Compare %s: gain=%.3f -> confidence=%.4f", name, r.RelativeSpeedupSampleAvsSampleB, r.Confidence)
		}
	}

	// compare CV (lower is better)
	compare("CV (WH/SM)", cvWHresults, cvSMresults)
	compare("MaxDevPct (WH/SM)", maxDevWHresults, maxDevSMresults)
	compare("Collisions (WH/SM)", collWHresults, collSMresults)
	compare("AvgHamming (WH/SM)", avgHdWHresults, avgHdSMresults)
	compare("DurationSec (WH/SM)", durationWHresults, durationSMresults)
} */

func TestCanonicalizeFloat64Bits_ZeroAndNaN(t *testing.T) {
	const seed = uint64(0x1234_5678_9abc_def0)

	// +0 and -0 must hash the same (because +0 == -0).
	p0 := 0.0
	n0 := math.Copysign(0, -1)
	hp0 := hashF64SM(unsafe.Pointer(&p0), seed)
	hn0 := hashF64SM(unsafe.Pointer(&n0), seed)
	if hp0 != hn0 {
		t.Fatalf("expected +0 and -0 to hash equally: %x vs %x", hp0, hn0)
	}

	// Different NaN payloads (and sign) should canonicalize to the same bits.
	nanA := math.Float64frombits(0x7ff0000000000001) // NaN (payload 1)
	nanB := math.Float64frombits(0x7ff8000000000002) // NaN (payload 2)
	nanC := math.Float64frombits(0xfff8000000000003) // NaN (payload 3, sign set)

	ha := hashF64SM(unsafe.Pointer(&nanA), seed)
	hb := hashF64SM(unsafe.Pointer(&nanB), seed)
	hc := hashF64SM(unsafe.Pointer(&nanC), seed)
	if ha != hb || ha != hc {
		t.Fatalf("expected NaNs to hash equally: %x %x %x", ha, hb, hc)
	}
}

func TestCanonicalizeFloat64BitsBranchless_SignPreserved(t *testing.T) {
	const seed = uint64(0x1234_5678_9abc_def0)

	// Sign must be preserved for non-zero numbers; +x and -x are not equal.
	p := 1.0
	n := -1.0
	hp := hashF64SM(unsafe.Pointer(&p), seed)
	hn := hashF64SM(unsafe.Pointer(&n), seed)
	if hp == hn {
		t.Fatalf("expected +1 and -1 to hash differently, got %x", hp)
	}
}

func TestHashCanonicalizedFloat64_ZeroCanonicalization(t *testing.T) {
	seed := uint64(0xDEADBEEFCAFEBABE)

	pz := 0.0
	nz := math.Copysign(0.0, -1.0)

	hpz := hashF64SM(unsafe.Pointer(&pz), seed)
	hnz := hashF64SM(unsafe.Pointer(&nz), seed)

	if hpz != hnz {
		t.Fatalf("+0 and -0 must hash equal: hpz=%#x hnz=%#x", hpz, hnz)
	}
	if want := splitmix64(seed); hpz != want {
		t.Fatalf("zero hash mismatch: got=%#x want=%#x", hpz, want)
	}

	// determinism
	if hpz2 := hashF64SM(unsafe.Pointer(&pz), seed); hpz2 != hpz {
		t.Fatalf("non-deterministic for +0: got=%#x want=%#x", hpz2, hpz)
	}
	if hnz2 := hashF64SM(unsafe.Pointer(&nz), seed); hnz2 != hnz {
		t.Fatalf("non-deterministic for -0: got=%#x want=%#x", hnz2, hnz)
	}
}

func TestHashCanonicalizedFloat64_NaNsCanonicalized(t *testing.T) {
	seed := uint64(0x0123456789ABCDEF)
	const canonNaNBits = 0x7ff8000000000000

	nans := []uint64{
		canonNaNBits,       // canonical qNaN
		0x7ff8000000000001, // qNaN with payload
		0x7ff8000000001234, // qNaN with different payload
		0x7ff0000000000001, // sNaN-like payload
		0xfff8000000000000, // negative qNaN
		0xfff0000000000001, // negative sNaN-like payload
		0x7fffffffffffffff, // exponent all ones, mantissa all ones (NaN)
		0xffffffffffffffff, // negative NaN with all mantissa bits
		0x7ff0123400000000, // NaN (exp all ones, mantissa non-zero)
		0xfff0123400000000, // negative NaN variant
	}

	want := splitmix64(seed ^ canonNaNBits)

	var first uint64
	for i, bits := range nans {
		v := math.Float64frombits(bits)
		got := hashF64SM(unsafe.Pointer(&v), seed)

		if i == 0 {
			first = got
		}
		if got != first {
			t.Fatalf("NaNs must hash equal after canonicalization: i=%d bits=%#x got=%#x first=%#x", i, bits, got, first)
		}
		if got != want {
			t.Fatalf("NaN canonical hash mismatch: i=%d bits=%#x got=%#x want=%#x", i, bits, got, want)
		}

		// determinism
		got2 := hashF64SM(unsafe.Pointer(&v), seed)
		if got2 != got {
			t.Fatalf("non-deterministic NaN hash: i=%d bits=%#x got=%#x got2=%#x", i, bits, got, got2)
		}
	}
}

func TestHashCanonicalizedFloat64_NonNaNMatchesSplitmixOfBits(t *testing.T) {
	seed := uint64(0xC0FFEE1234567890)

	vals := []float64{
		1.0,
		-1.0,
		1.5,
		-2.718281828,
		3.1415926535,
		math.SmallestNonzeroFloat64,
		math.MaxFloat64,
		math.Inf(1),
		math.Inf(-1),
	}

	for _, v := range vals {
		got := hashF64SM(unsafe.Pointer(&v), seed)
		u := math.Float64bits(v)
		want := splitmix64(seed ^ u)

		if got != want {
			t.Fatalf("non-NaN hash mismatch for v=%v: got=%#x want=%#x", v, got, want)
		}

		// determinism
		if got2 := hashF64SM(unsafe.Pointer(&v), seed); got2 != got {
			t.Fatalf("non-deterministic hash for v=%v: got=%#x got2=%#x", v, got, got2)
		}
	}
}

func TestHashCanonicalizedFloat64_InfIsNotTreatedAsNaN(t *testing.T) {
	seed := uint64(0xBADC0FFEE0DDF00D)

	posInf := math.Inf(1)
	negInf := math.Inf(-1)
	nan := math.NaN()

	hPos := hashF64SM(unsafe.Pointer(&posInf), seed)
	hNeg := hashF64SM(unsafe.Pointer(&negInf), seed)
	hNaN := hashF64SM(unsafe.Pointer(&nan), seed)

	wantPos := splitmix64(seed ^ math.Float64bits(posInf))
	wantNeg := splitmix64(seed ^ math.Float64bits(negInf))

	if hPos != wantPos {
		t.Fatalf("+Inf hash mismatch: got=%#x want=%#x", hPos, wantPos)
	}
	if hPos == hNaN {
		t.Fatalf("+Inf and NaN must not hash equal under bit-hash: +Inf=%#x NaN=%#x", hPos, hNaN)
	}
	if hNeg != wantNeg {
		t.Fatalf("-Inf hash mismatch: got=%#x want=%#x", hNeg, wantNeg)
	}
	if hNeg == hNaN {
		t.Fatalf("-Inf and NaN must not hash equal under bit-hash: -Inf=%#x NaN=%#x", hNeg, hNaN)
	}
	if hPos == hNeg {
		t.Fatalf("+Inf and -Inf must not hash equal under bit-hash: +Inf=%#x -Inf=%#x", hPos, hNeg)
	}
}
