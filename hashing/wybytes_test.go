package hashing

import (
	"fmt"
	"math"
	"math/bits"
	"testing"
	"unsafe"
)

// The tests in this file guard the byte-oriented hash routines against the
// failure modes that the statistical suites structurally cannot see.
//
// Avalanche and chi-squared are measured over random inputs. An input byte the
// hash never reads is still uniform; a collision family reachable only from one
// specific 64-bit constant never appears in a random sample. Both of those
// defects were present in the first draft of the lane-parallel body, both
// passed every statistical test in lab/hashquality, and each one is pinned
// below by a test that constructs the input rather than sampling for it.

// ── helpers ────────────────────────────────────────────────────────────────

// testRNG is a deterministic generator, so any failure is reproducible from
// the seed printed with it.
type testRNG struct{ s uint64 }

func (r *testRNG) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *testRNG) fill(b []byte) {
	for i := range b {
		b[i] = byte(r.next())
	}
}

// inputSpace is the number of distinct inputs of length n, or +Inf once that
// count is far past any sample size used here.
func inputSpace(n int) float64 {
	if n >= 8 {
		return math.Inf(1)
	}
	return math.Pow(256, float64(n))
}

// enumerate reports whether a statistic over `samples` inputs of length n would
// be measuring the generator rather than the hash.
//
// A one-byte input has 256 possible values. Drawing 65536 random ones gives
// each value about 256 times, and every bin count is then 256 times a count
// over 256 items — which inflates a chi-squared statistic by the same factor
// and fails a perfectly good hash. Where that is the case the tests below walk
// the input space instead and vary the seed, so every sample is a distinct
// (key, seed) pair.
func enumerate(n, samples int) bool {
	// Four times the sample count, not one: drawing n values at random from a
	// space of exactly n still repeats most of them, and the multiplicities
	// inflate the statistic by about a factor of two.
	return inputSpace(n) < 4*float64(samples)
}

// nthInput writes the i-th sample of length n into buf and returns the seed to
// use with it: the enumerated space for short inputs, a fresh random fill
// otherwise.
func nthInput(buf []byte, i int, base uint64, rng *testRNG, enum bool) uint64 {
	if !enum {
		rng.fill(buf)
		return base
	}
	space := int(inputSpace(len(buf)))
	v := i % space
	for j := range buf {
		buf[j] = byte(v >> (8 * j)) //nolint:gosec
	}
	// Once the space is exhausted the seed moves on. The step is the golden
	// ratio rather than one, so that a statistic taken across seeds is not
	// measuring the counter: consecutive seeds differ only in their lowest
	// bits, which is exactly where the H2 tag is read from.
	return base + uint64(i/space)*0x9e3779b97f4a7c15 //nolint:gosec
}

// hashLengths is the set of input lengths the structural tests walk. It covers
// every branch boundary of the dispatch (0, 1, 3, 4, 7, 8, 16, 17, 32, 33), the
// six specialized fixed sizes, and a full stride of residues past the long
// path's 24-byte loop so that no `n mod 24` is left untested.
func hashLengths() []int {
	seen := map[int]bool{}
	var out []int
	add := func(n int) {
		if n >= 0 && !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	for n := 0; n <= 120; n++ {
		add(n)
	}
	for _, n := range []int{127, 128, 129, 200, 255, 256, 257, 1000, 4096} {
		add(n)
	}
	return out
}

// ── the structural properties ──────────────────────────────────────────────

// TestHashReadsExactlyTheInput pins the read window to [0,n) from both sides:
// every byte inside the input must be able to change the hash, and no byte
// outside it may.
//
// This is the test that matters most in this file, because the lane routines
// buy their speed by reading overlapping windows instead of walking a tail byte
// by byte, and an overlap that is off by a few bytes fails silently. The first
// draft of laneLong stopped its 24-byte loop at `rest >= 24` and closed with a
// 16-byte window, leaving up to seven bytes unread for 29% of all lengths — a
// 41-byte key whose byte 24 could be changed freely without changing the hash.
// Every statistical test passed.
//
// The two directions are checked together because they are the same property.
// Reading too little loses input; reading too much is an out-of-bounds read
// that no bounds check would catch, since these routines work through unsafe
// pointers. Placing the input in the middle of a larger buffer turns both into
// ordinary assertions.
func TestHashReadsExactlyTheInput(t *testing.T) {
	const pad = 64
	const seed = uint64(0x243f6a8885a308d3)
	rng := &testRNG{s: 0x1234}

	for _, n := range hashLengths() {
		buf := make([]byte, pad+n+pad)
		rng.fill(buf)
		in := buf[pad : pad+n]
		want := HashBytesBlock(seed, in)

		// Inside: every single bit must reach the hash.
		for bit := range n * 8 {
			in[bit>>3] ^= 1 << (bit & 7)
			got := HashBytesBlock(seed, in)
			in[bit>>3] ^= 1 << (bit & 7)
			if got == want {
				t.Errorf("n=%d: flipping bit %d of byte %d does not change the hash; that byte is not read",
					n, bit&7, bit>>3)
				break
			}
		}

		// Outside: nothing before or after the input may be read.
		for i := range buf {
			if i >= pad && i < pad+n {
				continue
			}
			buf[i] ^= 0xff
			got := HashBytesBlock(seed, in)
			buf[i] ^= 0xff
			if got != want {
				t.Fatalf("n=%d: byte %d outside the input changes the hash; the routine reads out of bounds",
					n, i-pad)
			}
		}
	}
}

// TestHashRespondsToTheSeed verifies that the seed reaches the output for every
// input length, including the empty one.
//
// Set3 does not treat the seed as decoration. When a rehash is triggered by a
// collision pattern it draws a fresh seed specifically to break that pattern
// up, and an input whose hash ignores the seed defeats that mechanism outright:
// the same keys collide again after the rehash, and again after the next one.
//
// The empty input is the regression this test was written for. The superseded
// routine hashed it to zero for every seed, because the tail constant it used
// for the empty case was P1 and the mixer starts by XORing P1 into its input —
// so the chain was multiplied by zero before the seed ever reached it.
func TestHashRespondsToTheSeed(t *testing.T) {
	rng := &testRNG{s: 0x5eed}
	for _, n := range hashLengths() {
		buf := make([]byte, n)
		rng.fill(buf)

		seen := make(map[uint64]uint64, 256)
		for i := range 256 {
			seed := uint64(i)*0x9e3779b97f4a7c15 + 1
			h := HashBytesBlock(seed, buf)
			if other, ok := seen[h]; ok {
				t.Errorf("n=%d: seeds %#x and %#x give the same hash %#x", n, other, seed, h)
				break
			}
			seen[h] = seed
		}
	}
}

// TestNoSeedIndependentErasure constructs the inputs a multiply-based hash is
// vulnerable to, rather than hoping a random sample finds them.
//
// Mix is a widening multiply, so it returns zero whenever either operand is
// zero, and the final mix of zero is zero. Every operand here is an input word
// XORed with a secret word, so a key that matches a secret erases everything
// next to it: every such key hashes alike no matter what the rest of it holds.
// wyhash has this property and so does the Go runtime's map hasher; what makes
// it harmless is that the secret is not public.
//
// This test asserts the part that is not allowed to be true: that no *public*
// constant erases anything. Every constant in the package is placed at every
// word position of keys of every length class, the rest of the key is varied,
// and the results must stay distinct — for many seeds, so that a single lucky
// seed cannot hide a systematic hole.
//
// The seeds are drawn rather than chosen. A hand-picked seed equal to one of
// the constants does degenerate — k1 is seed^P1, which is zero exactly when the
// seed is P1 — and that is the same one-in-2^64 risk the reference carries when
// its random hashkey[1] comes up zero. TestErasureDoesNotSurviveAReseed covers
// what Set3 actually relies on instead.
func TestNoSeedIndependentErasure(t *testing.T) {
	constants := []struct {
		name string
		v    uint64
	}{
		{"zero", 0},
		{"ones", ^uint64(0)},
		{"P0", P0},
		{"P1", P1},
		{"P2", P2},
		{"P3", P3},
		{"M5", M5},
	}
	lengths := []int{8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96, 128}

	seedRNG := &testRNG{s: 0xf00d}
	for range 16 {
		seed := seedRNG.next()
		for _, n := range lengths {
			words := n / 8
			for _, c := range constants {
				for w := range words {
					seen := make(map[uint64]string)
					for v := range uint64(64) {
						buf := make([]byte, n)
						for j := range words {
							val := v*0x9e3779b97f4a7c15 + uint64(j)*0x1234567 //nolint:gosec
							if j == w {
								val = c.v
							}
							putWord(buf, j*8, val)
						}
						h := HashBytesBlock(seed, buf)
						key := fmt.Sprintf("%x", buf)
						if prev, ok := seen[h]; ok && prev != key {
							t.Fatalf("seed=%#x n=%d word %d pinned to %s: %s and %s collide at %#x; "+
								"a public constant erased the rest of the key",
								seed, n, w, c.name, prev, key, h)
						}
						seen[h] = key
					}
				}
			}
		}
	}
}

// TestErasureDoesNotSurviveAReseed pins the property Set3 depends on where the
// reference cannot promise more.
//
// A key whose first word equals the secret does erase the word beside it — that
// is wyhash's structure and this test constructs it deliberately rather than
// pretending otherwise. What must hold is that the family is a property of one
// seed and not of the algorithm: Set3 draws a fresh seed whenever a rehash is
// triggered by a collision pattern, and that only helps if the colliding keys
// stop colliding afterwards.
func TestErasureDoesNotSurviveAReseed(t *testing.T) {
	const seed = uint64(0x243f6a8885a308d3)

	// The erasing family: sixteen-byte keys whose first word is the secret.
	// Under this seed the first mix is zero and the second word is lost.
	family := make([][]byte, 8)
	for i := range family {
		b := make([]byte, 16)
		putWord(b, 0, seed^P1)
		putWord(b, 8, uint64(i)*0x9e3779b97f4a7c15+1) //nolint:gosec
		family[i] = b
	}

	first := HashBytesBlock(seed, family[0])
	for _, b := range family[1:] {
		if HashBytesBlock(seed, b) != first {
			t.Fatalf("the erasing family was expected to collide under its own seed; "+
				"if this now passes, the construction no longer matches the code and the "+
				"reseed guarantee below is being tested against nothing (key %x)", b)
		}
	}

	// Under any other seed they must be distinct again.
	rng := &testRNG{s: 0xbeef}
	for range 32 {
		other := rng.next()
		seen := make(map[uint64]int, len(family))
		for i, b := range family {
			h := HashBytesBlock(other, b)
			if prev, ok := seen[h]; ok {
				t.Errorf("seed %#x: keys %d and %d still collide after a reseed (%#x)", other, prev, i, h)
			}
			seen[h] = i
		}
	}
}

func putWord(b []byte, off int, v uint64) {
	for i := range 8 {
		b[off+i] = byte(v >> (8 * i)) //nolint:gosec
	}
}

// TestLengthIsPartOfTheHash verifies that inputs differing only in length
// hash differently.
//
// Without the length in the final mix, a key and the same key with trailing
// zeros would collide — and for the overlapping-window reads used here, so
// would several pairs whose windows happen to cover the same bytes.
func TestLengthIsPartOfTheHash(t *testing.T) {
	const seed = uint64(0xC0FFEE)
	t.Run("all zero", func(t *testing.T) {
		buf := make([]byte, 4200)
		seen := make(map[uint64]int)
		for _, n := range hashLengths() {
			h := HashBytesBlock(seed, buf[:n])
			if prev, ok := seen[h]; ok {
				t.Errorf("all-zero inputs of length %d and %d hash the same", prev, n)
			}
			seen[h] = n
		}
	})
	t.Run("shared prefix", func(t *testing.T) {
		buf := make([]byte, 4200)
		rng := &testRNG{s: 99}
		rng.fill(buf)
		seen := make(map[uint64]int)
		for _, n := range hashLengths() {
			h := HashBytesBlock(seed, buf[:n])
			if prev, ok := seen[h]; ok {
				t.Errorf("prefixes of length %d and %d hash the same", prev, n)
			}
			seen[h] = n
		}
	})
}

// TestStringAndByteSliceAgree verifies that the two entry points cannot drift
// apart. They share a body, and this is what keeps it that way.
func TestStringAndByteSliceAgree(t *testing.T) {
	rng := &testRNG{s: 7}
	for _, n := range hashLengths() {
		buf := make([]byte, n)
		rng.fill(buf)
		s := string(buf)
		want := HashBytesBlock(99, buf)
		if got := HashString(unsafe.Pointer(&s), 99); got != want {
			t.Errorf("n=%d: HashString gives %#x where HashBytesBlock gives %#x", n, got, want)
		}
	}
	// nil and empty must agree with each other and with the empty string.
	var nilSlice []byte
	empty := ""
	if HashBytesBlock(5, nilSlice) != HashBytesBlock(5, []byte{}) {
		t.Error("nil and empty slices hash differently")
	}
	if HashBytesBlock(5, nilSlice) != HashString(unsafe.Pointer(&empty), 5) {
		t.Error("the empty slice and the empty string hash differently")
	}
}

// TestFixedBlockHelpersAreTheGenericPath pins the specialized fixed-size
// helpers to the generic body they are specializations of.
//
// They exist for speed, not for a different answer: a [24]byte array and a
// 24-byte slice must hash alike, or a set keyed on one would disagree with a
// set keyed on the other.
func TestFixedBlockHelpersAreTheGenericPath(t *testing.T) {
	rng := &testRNG{s: 4242}
	for _, size := range []int{12, 16, 20, 24, 28, 32} {
		fn := fixedSizeByteBlockHasher(size)
		if fn == nil {
			t.Fatalf("size %d has no specialized helper", size)
		}
		for range 64 {
			buf := make([]byte, size)
			rng.fill(buf)
			for _, seed := range []uint64{0, 1, 0xdeadbeef} {
				want := HashBytesBlock(seed, buf)
				if got := fn(unsafe.Pointer(&buf[0]), seed); got != want {
					t.Fatalf("size %d seed %#x: helper gives %#x, generic path gives %#x", size, seed, got, want)
				}
			}
		}
	}
	// Sizes without a specialization must say so rather than return a helper
	// for the wrong width.
	for _, size := range []int{0, 1, 8, 11, 13, 33, 64} {
		if fn := fixedSizeByteBlockHasher(size); fn != nil {
			t.Errorf("size %d unexpectedly has a specialized helper", size)
		}
	}
}

// TestHashIsDeterministic verifies the property the whole package is named for.
func TestHashIsDeterministic(t *testing.T) {
	rng := &testRNG{s: 31337}
	for _, n := range hashLengths() {
		buf := make([]byte, n)
		rng.fill(buf)
		first := HashBytesBlock(0xABCD, buf)
		for range 4 {
			if got := HashBytesBlock(0xABCD, buf); got != first {
				t.Fatalf("n=%d: not deterministic, %#x then %#x", n, first, got)
			}
		}
	}
}

// TestShortKeysAreCollisionFree enumerates the small input spaces outright.
//
// The branches below eight bytes are the ones that cannot be judged by a
// statistic: a one-byte key has 256 possible values and a two-byte key 65536,
// which is too few for a chi-squared to say anything and few enough to simply
// check completely. Those branches fold the same bytes into both operands of
// the mixer, and that is exactly the kind of shortcut that can turn out to be
// two-to-one. Three to seven bytes use the same mixer over a larger space and
// are sampled instead.
func TestShortKeysAreCollisionFree(t *testing.T) {
	for _, seed := range []uint64{1, 0x243f6a8885a308d3, P2} {
		for n := 1; n <= 2; n++ {
			space := 1 << (8 * n)
			seen := make(map[uint64]int, space)
			buf := make([]byte, n)
			for v := range space {
				for j := range buf {
					buf[j] = byte(v >> (8 * j)) //nolint:gosec
				}
				h := HashBytesBlock(seed, buf)
				if prev, ok := seen[h]; ok {
					t.Fatalf("seed=%#x n=%d: values %d and %d both hash to %#x", seed, n, prev, v, h)
				}
				seen[h] = v
			}
		}
		// Three to seven bytes go through the same two-operand mixer over a
		// space too large to enumerate. A quarter of a
		// million distinct values is far past where a two-to-one map would show
		// itself, and far short of where a sound 64-bit hash would produce a
		// birthday collision.
		samples := 1 << 18
		if testing.Short() {
			samples = 1 << 15
		}
		for n := 3; n <= 7; n++ {
			// Keyed by hash, valued by the input that produced it, so a
			// generator that happened to repeat an input is reported as such
			// rather than as a collision.
			seen := make(map[uint64]string, samples)
			buf := make([]byte, n)
			for i := range uint64(samples) { //nolint:gosec
				v := i * 0x9e3779b97f4a7c15
				for j := range buf {
					buf[j] = byte(v >> (8 * j)) //nolint:gosec
				}
				h := HashBytesBlock(seed, buf)
				if prev, ok := seen[h]; ok && prev != string(buf) {
					t.Fatalf("seed=%#x n=%d: %x and %x collide at %#x", seed, n, prev, buf, h)
				}
				seen[h] = string(buf)
			}
		}
	}
}

// TestBlockOrderMatters verifies that rearranging whole blocks of a key changes
// its hash.
//
// This is the failure mode a lane-parallel hash invites and a serial one cannot
// have. The long path feeds byte offsets 0, 24, 48 … to one accumulator, 8, 32,
// 56 … to a second and 16, 40, 64 … to a third, and combines the three with
// XOR. If the per-lane step were commutative, or if the lanes were symmetric in
// the wrong way, a key and a permutation of its 24-byte blocks would hash alike
// — and a set of such keys is trivial to construct by accident, since records
// laid out as repeated fixed-size fields are exactly that shape.
func TestBlockOrderMatters(t *testing.T) {
	const seed = uint64(0x243f6a8885a308d3)
	const block = 24
	rng := &testRNG{s: 0x0b10c}

	blocks := make([][]byte, 4)
	for i := range blocks {
		blocks[i] = make([]byte, block)
		rng.fill(blocks[i])
	}

	seen := make(map[uint64][]int)
	var perm func(prefix []int, rest []int)
	perm = func(prefix, rest []int) {
		if len(rest) == 0 {
			key := make([]byte, 0, len(prefix)*block)
			for _, b := range prefix {
				key = append(key, blocks[b]...)
			}
			h := HashBytesBlock(seed, key)
			if prev, ok := seen[h]; ok {
				t.Errorf("block orders %v and %v hash the same (%#x)", prev, prefix, h)
			}
			seen[h] = append([]int{}, prefix...)
			return
		}
		for i := range rest {
			next := append(append([]int{}, rest[:i]...), rest[i+1:]...)
			perm(append(prefix, rest[i]), next)
		}
	}
	perm(nil, []int{0, 1, 2, 3})

	// The same question one lane at a time: moving a single byte to a different
	// position within the key must change the hash, even when the position it
	// moves to lands in the same lane.
	base := make([]byte, 96)
	rng.fill(base)
	want := HashBytesBlock(seed, base)
	for _, off := range []int{0, 8, 16, 24, 48, 72, 95} {
		moved := append([]byte{}, base...)
		moved[off], moved[(off+24)%96] = moved[(off+24)%96], moved[off]
		if moved[off] == moved[(off+24)%96] {
			continue // the swap was a no-op
		}
		if got := HashBytesBlock(seed, moved); got == want {
			t.Errorf("swapping bytes %d and %d does not change the hash", off, (off+24)%96)
		}
	}
}

// ── the statistical properties ─────────────────────────────────────────────

// chiSquaredPerDF bins values and returns the chi-squared statistic divided by
// its degrees of freedom. A uniform distribution gives about one.
func chiSquaredPerDF(counts []uint64, total uint64) float64 {
	k := float64(len(counts))
	expected := float64(total) / k
	var chi float64
	for _, c := range counts {
		d := float64(c) - expected
		chi += d * d / expected
	}
	return chi / (k - 1)
}

// TestHashBucketsAreUniform checks the two slices of the hash that Set3
// actually consumes, rather than the 64-bit value as a whole.
//
// Set3 takes the lowest seven bits as the H2 tag that every control-word
// comparison matches against, and higher bits to choose the group. A hash whose
// low bits carry structure makes every probe hit a false tag match; one whose
// high bits cluster makes the probe sequences long. At the default occupancy of
// 83% there is very little headroom for either.
func TestHashBucketsAreUniform(t *testing.T) {
	const samples = 1 << 16
	const seed = uint64(0x243f6a8885a308d3)
	const groups = 1021 // a prime, as Set3's group count always is
	// Chi-squared over this many bins is tightly concentrated around one; two
	// is far outside sampling noise and is a clear signal of structure.
	const maxChi = 2.0

	// The empty input is absent on purpose. It has exactly one value, so a
	// distribution over it is a distribution over seeds, and the reference
	// returns the seeded accumulator for it untouched — which makes the
	// statistic a statement about how the test picks seeds. What matters for
	// the empty key is that its hash moves with the seed, and
	// TestHashRespondsToTheSeed says so directly.
	for _, n := range []int{1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 41, 64, 128} {
		rng := &testRNG{s: 0xabcd}
		buf := make([]byte, n)
		enum := enumerate(n, samples)
		tags := make([]uint64, 128)
		grps := make([]uint64, groups)
		for i := range samples {
			h := HashBytesBlock(nthInput(buf, i, seed, rng, enum), buf)
			tags[h&0x7f]++
			grps[(h>>7)%groups]++
		}
		tagChi := chiSquaredPerDF(tags, samples)
		grpChi := chiSquaredPerDF(grps, samples)
		t.Logf("n=%-4d H2 tag chi2/df %.3f, group chi2/df %.3f", n, tagChi, grpChi)
		if tagChi > maxChi {
			t.Errorf("n=%d: the 7-bit H2 tag is not uniform, chi2/df %.3f over %.1f", n, tagChi, maxChi)
		}
		if grpChi > maxChi {
			t.Errorf("n=%d: the group index is not uniform, chi2/df %.3f over %.1f", n, grpChi, maxChi)
		}
	}
}

// TestHashSpreadsNearIdenticalKeys checks the case random inputs cannot.
//
// Real keys are not random. They are identifiers sharing a prefix, counters
// differing in one byte, paths under a common root. A hash that mixes random
// input well can still map a family of near-identical keys onto a handful of
// groups, and that is the input a hash table is most often given.
func TestHashSpreadsNearIdenticalKeys(t *testing.T) {
	const seed = uint64(0x243f6a8885a308d3)
	const groups = 1021
	const maxChi = 2.0

	families := []struct {
		name string
		// samples is per family: the families differ in how many distinct keys
		// they can produce, and a chi-squared over more samples than distinct
		// keys measures the generator rather than the hash.
		samples int
		make    func(i int) []byte
	}{
		{"shared 16-byte prefix, counter in the tail", 1 << 16, func(i int) []byte {
			b := []byte("user:prefix00000________")
			b[16], b[17], b[18] = byte(i), byte(i>>8), byte(i>>16)
			return b
		}},
		{"counter in the first bytes, shared tail", 1 << 16, func(i int) []byte {
			b := []byte("________/var/lib/objects")
			b[0], b[1], b[2] = byte(i), byte(i>>8), byte(i>>16)
			return b
		}},
		{"all zero but a counter in one field", 1 << 16, func(i int) []byte {
			b := make([]byte, 24)
			b[8], b[9], b[10] = byte(i), byte(i>>8), byte(i>>16)
			return b
		}},
		{"41 bytes, counter straddling the long-path seam", 1 << 16, func(i int) []byte {
			// 41 = 24 + 17: the length at which the loop stops and the closing
			// window must reach back over bytes the loop already read. The
			// counter sits exactly on the seam.
			b := make([]byte, 41)
			b[24], b[25], b[26] = byte(i), byte(i>>8), byte(i>>16)
			return b
		}},
		{"exactly two bits set", 1 << 14, func(i int) []byte {
			// Minimally different keys: every pair differs in at most four
			// bits. Walk the unordered pairs of the 192 bit positions directly
			// so that every index gives a distinct key.
			b := make([]byte, 24)
			lo, k := 0, i
			for k >= 191-lo {
				k -= 191 - lo
				lo++
			}
			hi := lo + 1 + k
			b[lo>>3] |= 1 << (lo & 7)
			b[hi>>3] |= 1 << (hi & 7)
			return b
		}},
	}

	for _, fam := range families {
		// A family that repeats keys cannot be judged by a chi-squared over
		// groups: the statistic would then measure how few distinct keys there
		// are, not how well they are spread.
		distinct := make(map[string]struct{}, fam.samples)
		for i := range fam.samples {
			distinct[string(fam.make(i))] = struct{}{}
		}
		if len(distinct) < fam.samples {
			t.Fatalf("%s: generates only %d distinct keys for %d samples", fam.name, len(distinct), fam.samples)
		}

		counts := make([]uint64, groups)
		hashes := make(map[uint64]struct{}, fam.samples)
		for i := range fam.samples {
			h := HashBytesBlock(seed, fam.make(i))
			counts[(h>>7)%groups]++
			hashes[h] = struct{}{}
		}
		chi := chiSquaredPerDF(counts, uint64(fam.samples)) //nolint:gosec
		collisions := fam.samples - len(hashes)
		t.Logf("%-46s group chi2/df %.3f, %d full-hash collisions", fam.name, chi, collisions)
		if chi > maxChi {
			t.Errorf("%s: near-identical keys cluster, chi2/df %.3f over %.1f", fam.name, chi, maxChi)
		}
		// At 65536 keys over 2^64 outputs the birthday expectation is far below
		// one; a handful would already mean structure survived the mixing.
		if collisions > 2 {
			t.Errorf("%s: %d full 64-bit collisions among %d keys, expected none", fam.name, collisions, fam.samples)
		}
	}
}

// TestHashAvalanche measures, for each input length, how often flipping one
// input bit flips each output bit.
//
// A hash that mixes properly flips each output bit half the time and 32 bits on
// average. This is the property the lane design puts at risk: splitting the
// work into independent accumulators means each input word passes through fewer
// mixing rounds before it is combined, and the long-input loop gives each word
// a single multiply inside its lane.
//
// Every length is measured to the same number of flip pairs so that the
// acceptance threshold is the same for all of them and is derived from the
// sampling error rather than guessed. An earlier version sized the sample per
// input instead, and its threshold was tighter than the sampling error at short
// lengths — it failed the then-shipping routine on noise.
func TestHashAvalanche(t *testing.T) {
	targetPairs := 1 << 20
	if testing.Short() {
		targetPairs = 1 << 16
	}
	const seed = uint64(0x243f6a8885a308d3)

	for _, n := range []int{1, 2, 3, 4, 8, 12, 16, 17, 20, 24, 28, 32, 33, 41, 48, 64, 128, 256} {
		samples := max(targetPairs/(n*8), 1)
		rng := &testRNG{s: 0x5eed_1a4e}
		buf := make([]byte, n)
		enum := enumerate(n, samples)
		var flips [64]uint64
		var pairs, sumFlipped uint64

		for i := range samples {
			s := nthInput(buf, i, seed, rng, enum)
			h0 := HashBytesBlock(s, buf)
			for bit := range n * 8 {
				buf[bit>>3] ^= 1 << (bit & 7)
				diff := h0 ^ HashBytesBlock(s, buf)
				buf[bit>>3] ^= 1 << (bit & 7)
				pairs++
				sumFlipped += uint64(bits.OnesCount64(diff)) //nolint:gosec
				for ob := range 64 {
					flips[ob] += (diff >> ob) & 1
				}
			}
		}

		np := float64(pairs)
		var maxDev float64
		for _, c := range flips {
			maxDev = math.Max(maxDev, math.Abs(float64(c)/np-0.5))
		}
		// Six standard errors of a fair coin, floored at a value that stays
		// meaningful once the sample is large.
		//
		// The error is computed over the number of *distinct* flip pairs, not
		// over the number measured. A one-byte input admits 2048 of them in
		// total, so no amount of repetition makes the estimate more precise,
		// and a threshold derived from the repetition count would demand a
		// precision the input space cannot carry. An earlier version did that
		// and failed the then-shipping routine on noise.
		distinctPairs := math.Min(np, inputSpace(n)*float64(n*8))
		threshold := math.Max(0.004, 3/math.Sqrt(distinctPairs))
		mean := float64(sumFlipped) / np
		t.Logf("n=%-4d max deviation %.4f (threshold %.4f), mean bits flipped %.2f", n, maxDev, threshold, mean)

		if maxDev > threshold {
			t.Errorf("n=%d: an output bit deviates %.4f from an even flip, over the %.4f sampling threshold; "+
				"some input bits do not reach some output bits", n, maxDev, threshold)
		}
		if mean < 31 || mean > 33 {
			t.Errorf("n=%d: flips %.2f of 64 output bits on average, expected about 32", n, mean)
		}
	}
}
