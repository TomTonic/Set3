package set3

import (
	"encoding/binary"
	"math"
	"testing"
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

	const buckets = 1 << 20 // 2^20 buckets
	// samplesPerBucket controls how many samples fall on average into
	// each bucket. Choose 128 for a reasonable tradeoff between runtime
	// and statistical stability: mean ~= 128, expected rel stddev ~= 1/sqrt(128).
	const samplesPerBucket = 128
	N := uint64(buckets * samplesPerBucket)

	counts := make([]uint32, buckets)
	var sum uint64
	for i := uint64(0); i < N; i++ {
		v := splitmix64(i)
		idx := int(v & uint64(buckets-1))
		counts[idx]++
		sum++
	}

	mean := float64(sum) / float64(buckets)
	// compute observed variance
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

	// expected relative stddev for multinomial distribution ≈ 1/sqrt(mean)
	expectedRelStd := math.Sqrt((1.0 - 1.0/float64(buckets)) / mean)

	t.Logf("splitmix64 uniformity: samples=%d buckets=%d mean=%.3f obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f",
		N, buckets, mean, obsRelStd, expectedRelStd, maxDev)

	// Allow some slack: require observed rel std to be within 3x expected.
	if obsRelStd > 3.0*expectedRelStd {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (3x expected)", obsRelStd, 3.0*expectedRelStd)
	}

	// Also ensure no single bucket deviates excessively (e.g. > 200%). This
	// guards against pathological clustering even if the overall stddev is ok.
	if maxDev > 2.0 {
		t.Fatalf("a bucket deviated too much from mean: maxRelDev=%.6f", maxDev)
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
