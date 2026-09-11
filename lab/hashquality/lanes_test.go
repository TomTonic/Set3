//go:build set3lab

// Copyright 2024 TomTonic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package hashquality

import (
	"math"
	"math/bits"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/lab/hashalt"
)

// The lane-parallel candidates in lab/hashalt are two to four times faster
// than the production byte-block routines because they replace a serial chain
// of widening multiplies with independent lanes. Fewer rounds of mixing is
// exactly how a hash gets faster and worse at the same time, so this file is
// the gate: nothing moves into the production package until it passes here.
//
// Set3 leans on two properties in particular, and neither is "the output looks
// random". It takes the lowest seven bits as the H2 tag that a group's control
// word is matched against, and it takes higher bits to choose the group. A
// hash whose low bits correlate with the input makes every probe in a group
// hit a false tag match; one whose high bits cluster makes the probe sequences
// long. At Set3's default occupancy of 83% there is very little headroom for
// either.

// laneCandidate is one routine under test, wrapped so that the test can drive
// production and candidate through the same interface.
type laneCandidate struct {
	name string
	hash func(b []byte, seed uint64) uint64
}

// byteCandidates are the variable-length routines: what the package uses today
// against what could replace it.
var byteCandidates = []laneCandidate{
	{"production HashBytesBlock", func(b []byte, seed uint64) uint64 { return hashing.HashBytesBlock(seed, b) }},
	{"lane-parallel WHLaneBytes", func(b []byte, seed uint64) uint64 { return hashalt.WHLaneBytes(seed, b) }},
}

// fixedCandidates are the fixed-size raw-block routines, the path a struct key
// takes. The 24-byte entry is a three-field uint64 struct.
var fixedCandidates = map[int][]laneCandidate{
	16: {
		{"production block16", func(b []byte, seed uint64) uint64 {
			return hashing.HashAsByteArray[[16]byte](unsafe.Pointer(&b[0]), seed)
		}},
		{"lane-parallel block16", func(b []byte, seed uint64) uint64 { return hashalt.WHLaneBlock16(unsafe.Pointer(&b[0]), seed) }},
	},
	24: {
		{"production block24", func(b []byte, seed uint64) uint64 {
			return hashing.HashAsByteArray[[24]byte](unsafe.Pointer(&b[0]), seed)
		}},
		{"lane-parallel block24", func(b []byte, seed uint64) uint64 { return hashalt.WHLaneBlock24(unsafe.Pointer(&b[0]), seed) }},
	},
	32: {
		{"production block32", func(b []byte, seed uint64) uint64 {
			return hashing.HashAsByteArray[[32]byte](unsafe.Pointer(&b[0]), seed)
		}},
		{"lane-parallel block32", func(b []byte, seed uint64) uint64 { return hashalt.WHLaneBlock32(unsafe.Pointer(&b[0]), seed) }},
	},
}

// laneRNG is a deterministic generator, so a failure is reproducible.
type laneRNG struct{ s uint64 }

func (r *laneRNG) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (r *laneRNG) fill(b []byte) {
	for i := range b {
		b[i] = byte(r.next())
	}
}

// avalancheOf measures, for inputs of the given length, how often flipping one
// input bit flips each output bit.
//
// It returns the largest absolute deviation from one half across the 64 output
// bits, and the mean number of output bits flipped. A hash that mixes properly
// gives 0.5 and 32; anything meaningfully away from those means some part of
// the input does not reach some part of the output.
func avalancheOf(hash func([]byte, uint64) uint64, length, targetPairs int, seed uint64) (maxDev, meanFlipped, threshold float64) {
	// Enough inputs that every length is measured to the same statistical
	// power. Without this a short input yields few pairs, its sampling error
	// swamps the effect being looked for, and the test fires on noise — which
	// it did, against the production routine, before this was fixed.
	samples := targetPairs / (length * 8)
	if samples < 1 {
		samples = 1
	}
	rng := &laneRNG{s: 0x5eed_1a4e_0000_0001}
	buf := make([]byte, length)
	var flipCounts [64]uint64
	var pairs, sumFlipped uint64

	for range samples {
		rng.fill(buf)
		h0 := hash(buf, seed)
		for bit := range length * 8 {
			buf[bit>>3] ^= 1 << (bit & 7)
			diff := h0 ^ hash(buf, seed)
			buf[bit>>3] ^= 1 << (bit & 7)

			pairs++
			sumFlipped += uint64(bits.OnesCount64(diff))
			for ob := range 64 {
				flipCounts[ob] += (diff >> ob) & 1
			}
		}
	}
	if pairs == 0 {
		return 1, 0, 0
	}
	n := float64(pairs)
	for _, c := range flipCounts {
		maxDev = math.Max(maxDev, math.Abs(float64(c)/n-0.5))
	}
	// Six standard errors of a fair coin over n pairs, floored at a value that
	// stays meaningful for large n. This is the same shape the package's
	// existing avalanche test uses.
	threshold = math.Max(0.004, (6*0.5)/math.Sqrt(n))
	return maxDev, float64(sumFlipped) / n, threshold
}

// TestLaneCandidatesAvalanche verifies that every bit of the input reaches
// every bit of the output, for the lane-parallel candidates as strictly as for
// the routines they would replace.
//
// This is the property the lane design puts at risk. Splitting the work into
// independent accumulators means each input word passes through fewer mixing
// rounds before it is combined, and the long-input loop gives each word only a
// single multiply inside its lane. If that is not enough, it shows up here as
// an output bit that a particular input bit cannot reach.
//
// Both candidates are measured side by side rather than against an absolute
// bar alone, because "as good as what we have" is the question being asked.
func TestLaneCandidatesAvalanche(t *testing.T) {
	// Every length is measured to the same number of flip pairs, so the
	// acceptance threshold is the same for all of them and is derived from the
	// sampling error rather than guessed.
	const targetPairs = 1 << 20
	const seed = uint64(0x243f6a8885a308d3)

	t.Run("variable length", func(t *testing.T) {
		for _, length := range []int{8, 12, 16, 20, 24, 32, 64, 128, 256} {
			for _, c := range byteCandidates {
				dev, mean, threshold := avalancheOf(c.hash, length, targetPairs, seed)
				t.Logf("len=%-4d %-28s max deviation %.4f (threshold %.4f), mean bits flipped %.2f", length, c.name, dev, threshold, mean)
				if dev > threshold {
					t.Errorf("len=%d %s: output bit deviates %.4f from an even flip, over the %.4f sampling threshold; "+
						"some input bits do not reach some output bits", length, c.name, dev, threshold)
				}
				if mean < 31 || mean > 33 {
					t.Errorf("len=%d %s: flips %.2f of 64 output bits on average, expected about 32", length, c.name, mean)
				}
			}
		}
	})

	t.Run("fixed size", func(t *testing.T) {
		for _, length := range []int{16, 24, 32} {
			for _, c := range fixedCandidates[length] {
				dev, mean, threshold := avalancheOf(c.hash, length, targetPairs, seed)
				t.Logf("len=%-4d %-28s max deviation %.4f (threshold %.4f), mean bits flipped %.2f", length, c.name, dev, threshold, mean)
				if dev > threshold {
					t.Errorf("len=%d %s: output bit deviates %.4f from an even flip, over the %.4f sampling threshold",
						length, c.name, dev, threshold)
				}
				if mean < 31 || mean > 33 {
					t.Errorf("len=%d %s: flips %.2f of 64 output bits on average, expected about 32", length, c.name, mean)
				}
			}
		}
	})
}

// chiSquaredUniformity bins values into buckets and returns the chi-squared
// statistic normalised by its degrees of freedom. A value near one is what a
// uniform distribution produces; a large one means the bins are lumpy.
func chiSquaredUniformity(counts []uint64, total uint64) float64 {
	k := float64(len(counts))
	expected := float64(total) / k
	var chi float64
	for _, c := range counts {
		d := float64(c) - expected
		chi += d * d / expected
	}
	return chi / (k - 1)
}

// TestLaneCandidatesBucketUniformity verifies the two properties Set3 actually
// consumes: the seven-bit H2 tag and the group index.
//
// A hash can have flawless avalanche and still be unusable here if its low bits
// carry structure, because Set3 does not use the whole 64-bit value — it uses
// seven bits for the tag that every control-word comparison matches against,
// and a separate slice for the group. Both are checked against a chi-squared
// statistic normalised by its degrees of freedom, where one is the value a
// uniform distribution gives.
func TestLaneCandidatesBucketUniformity(t *testing.T) {
	const samples = 1 << 18
	const seed = uint64(0x243f6a8885a308d3)
	const groups = 1021 // a prime, as Set3's group count always is

	// Chi-squared over many bins is tightly concentrated around one; two is far
	// outside what sampling noise produces at this sample size and is a clear
	// signal of structure.
	const maxAcceptableChi = 2.0

	for _, length := range []int{8, 16, 20, 24, 32, 64} {
		candidates := byteCandidates
		if fixed, ok := fixedCandidates[length]; ok {
			candidates = append(append([]laneCandidate{}, byteCandidates...), fixed...)
		}
		for _, c := range candidates {
			rng := &laneRNG{s: 0xabcd_0000_0000_0001}
			buf := make([]byte, length)
			tagCounts := make([]uint64, 128)
			groupCounts := make([]uint64, groups)

			for range samples {
				rng.fill(buf)
				h := c.hash(buf, seed)
				tagCounts[h&0x7f]++
				// The same reduction Set3 uses to pick a group.
				groupCounts[(h>>7)%groups]++
			}
			tagChi := chiSquaredUniformity(tagCounts, samples)
			groupChi := chiSquaredUniformity(groupCounts, samples)
			t.Logf("len=%-4d %-28s H2 tag chi2/df %.3f, group chi2/df %.3f", length, c.name, tagChi, groupChi)

			if tagChi > maxAcceptableChi {
				t.Errorf("len=%d %s: the 7-bit H2 tag is not uniform, chi2/df %.3f over %.1f",
					length, c.name, tagChi, maxAcceptableChi)
			}
			if groupChi > maxAcceptableChi {
				t.Errorf("len=%d %s: the group index is not uniform, chi2/df %.3f over %.1f",
					length, c.name, groupChi, maxAcceptableChi)
			}
		}
	}
}

// TestLaneCandidatesOnStructuredKeys checks the case random inputs cannot: keys
// that differ in very little.
//
// Real keys are not random. They are identifiers that share a prefix, counters
// that differ in one byte, paths under a common root. A hash that mixes random
// input well can still map a family of near-identical keys onto a handful of
// groups, and that is the input a hash table is most often given.
func TestLaneCandidatesOnStructuredKeys(t *testing.T) {
	const seed = uint64(0x243f6a8885a308d3)
	const groups = 1021
	const maxAcceptableChi = 2.0

	families := []struct {
		name string
		// samples is per family because the families differ in how many
		// distinct keys they can produce, and a chi-squared over more samples
		// than distinct keys measures the generator rather than the hash.
		samples int
		make    func(i int) []byte
	}{
		{"shared 16-byte prefix, counter in the tail", 1 << 16, func(i int) []byte {
			b := []byte("user:prefix00000________")
			b[16] = byte(i)
			b[17] = byte(i >> 8)
			b[18] = byte(i >> 16)
			return b
		}},
		{"counter in the first bytes, shared tail", 1 << 16, func(i int) []byte {
			b := []byte("________/var/lib/objects")
			b[0] = byte(i)
			b[1] = byte(i >> 8)
			b[2] = byte(i >> 16)
			return b
		}},
		{"all zero but a counter in one field", 1 << 16, func(i int) []byte {
			// A struct key where only one field varies, which is the sparsest
			// input a real program is likely to produce.
			b := make([]byte, 24)
			b[8] = byte(i)
			b[9] = byte(i >> 8)
			b[10] = byte(i >> 16)
			return b
		}},
		{"exactly two bits set", 1 << 14, func(i int) []byte {
			// Minimally different keys: every pair differs in at most four
			// bits. A hash that carries input structure into the group index
			// shows it here before it shows it anywhere else.
			b := make([]byte, 24)
			// Walk the unordered pairs of the 192 bit positions directly, so
			// every index gives a distinct key: 192*191/2 = 18336 of them.
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
		// groups: the statistic then measures how few distinct keys there are,
		// not how well they are spread. An earlier version of this test did
		// exactly that and reported both candidates as catastrophic failures
		// for a family with 192 distinct values spread over 1021 buckets.
		distinct := map[string]struct{}{}
		for i := range fam.samples {
			distinct[string(fam.make(i))] = struct{}{}
		}
		if len(distinct) < fam.samples {
			t.Fatalf("%s: generates only %d distinct keys for %d samples; "+
				"the uniformity statistic would measure the generator, not the hash", fam.name, len(distinct), fam.samples)
		}

		for _, c := range byteCandidates {
			counts := make([]uint64, groups)
			for i := range fam.samples {
				h := c.hash(fam.make(i), seed)
				counts[(h>>7)%groups]++
			}
			chi := chiSquaredUniformity(counts, uint64(fam.samples)) //nolint:gosec
			t.Logf("%-42s %-28s group chi2/df %.3f", fam.name, c.name, chi)
			if chi > maxAcceptableChi {
				t.Errorf("%s / %s: near-identical keys cluster, chi2/df %.3f over %.1f",
					fam.name, c.name, chi, maxAcceptableChi)
			}
		}
	}
}

// TestLaneCandidatesAreDeterministic verifies the property the whole package
// is named for: the same input and seed give the same hash, every time and in
// every process.
func TestLaneCandidatesAreDeterministic(t *testing.T) {
	rng := &laneRNG{s: 7}
	for _, length := range []int{0, 1, 3, 7, 8, 15, 16, 23, 24, 31, 32, 63, 64, 200} {
		buf := make([]byte, length)
		rng.fill(buf)
		first := hashalt.WHLaneBytes(99, buf)
		for range 8 {
			if got := hashalt.WHLaneBytes(99, buf); got != first {
				t.Fatalf("len=%d: WHLaneBytes is not deterministic, %#x then %#x", length, first, got)
			}
		}
		// A string and a byte slice holding the same bytes must agree, or the
		// two entry points would disagree about the same key.
		s := string(buf)
		if got := hashalt.WHLaneString(unsafe.Pointer(&s), 99); got != first {
			t.Errorf("len=%d: WHLaneString gives %#x where WHLaneBytes gives %#x", length, got, first)
		}
	}
}
