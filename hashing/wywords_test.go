package hashing

import (
	"math/rand/v2"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// callMixer dispatches to the straight-line mixer for k = len(w).
func callMixer(t *testing.T, w []uint64, seed uint64) uint64 {
	t.Helper()
	switch len(w) {
	case 2:
		return mixWords2(w[0], w[1], seed)
	case 3:
		return mixWords3(w[0], w[1], w[2], seed)
	case 4:
		return mixWords4(w[0], w[1], w[2], w[3], seed)
	case 5:
		return mixWords5(w[0], w[1], w[2], w[3], w[4], seed)
	case 6:
		return mixWords6(w[0], w[1], w[2], w[3], w[4], w[5], seed)
	case 7:
		return mixWords7(w[0], w[1], w[2], w[3], w[4], w[5], w[6], seed)
	case 8:
		return mixWords8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], seed)
	default:
		t.Fatalf("no straight-line mixer for %d words", len(w))
		return 0
	}
}

// TestWordMixersAreTheGenericPath is the load-bearing test of wywords.go.
//
// Every mixer claims to be wyBlock with the length pinned to 8k and the loads
// replaced by register operands. That claim is checkable exactly rather than
// statistically: lay the same words out in memory, hash those bytes with the
// generic routine, and require the identical value. A transcription slip in
// any of the eight — a wrong overlap offset, a missed lane, a length constant
// off by eight — fails here.
func TestWordMixersAreTheGenericPath(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x5eed, 0xfeed))
	seeds := []uint64{0, 1, 0x9e3779b97f4a7c15, ^uint64(0), 0x0123456789abcdef}

	for k := 2; k <= 8; k++ {
		for _, seed := range seeds {
			for range 2000 {
				w := make([]uint64, k)
				for i := range w {
					w[i] = rng.Uint64()
				}
				want := wyBlock(unsafe.Pointer(&w[0]), k*8, seed)
				got := callMixer(t, w, seed)
				require.Equalf(t, want, got,
					"mixWords%d disagrees with wyBlock at %d bytes, seed %#x, words %#v",
					k, k*8, seed, w)
			}
		}
	}
}

// TestWordMixersOnDegenerateWords repeats the equality on the inputs a random
// generator will never produce: all zero, all ones, and one word differing.
// The zero case matters because Mix returns zero whenever either operand is
// zero, so it is where a transcription that dropped a key constant would still
// pass a random test.
func TestWordMixersOnDegenerateWords(t *testing.T) {
	patterns := [][]uint64{
		{0}, {^uint64(0)}, {1}, {P0}, {P1}, {M5},
	}
	seeds := []uint64{0, 1, P0, P1, M5, 0xffff_ffff_ffff_ffff}

	for k := 2; k <= 8; k++ {
		for _, pat := range patterns {
			for _, seed := range seeds {
				w := make([]uint64, k)
				for i := range w {
					w[i] = pat[0]
				}
				want := wyBlock(unsafe.Pointer(&w[0]), k*8, seed)
				require.Equalf(t, want, callMixer(t, w, seed),
					"mixWords%d disagrees on the uniform word %#x, seed %#x", k, pat[0], seed)

				// And with exactly one word perturbed, at every position.
				for pos := range k {
					w2 := make([]uint64, k)
					copy(w2, w)
					w2[pos] ^= 0x8000_0000_0000_0001
					want2 := wyBlock(unsafe.Pointer(&w2[0]), k*8, seed)
					require.Equalf(t, want2, callMixer(t, w2, seed),
						"mixWords%d disagrees with word %d perturbed, seed %#x", k, pos, seed)
				}
			}
		}
	}
}

// TestMixWordSliceIsTheGenericPath holds the slice path to the same equality,
// including the counts the straight-line mixers cover and the ones they do not.
func TestMixWordSliceIsTheGenericPath(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for k := 1; k <= 40; k++ {
		for _, seed := range []uint64{0, 1, 0xdeadbeefcafe} {
			w := make([]uint64, k)
			for i := range w {
				w[i] = rng.Uint64()
			}
			want := wyBlock(unsafe.Pointer(&w[0]), k*8, seed)
			require.Equalf(t, want, mixWordSlice(w, seed), "mixWordSlice disagrees at %d words", k)
			if k >= 2 && k <= 8 {
				require.Equalf(t, want, callMixer(t, w, seed),
					"mixWords%d and mixWordSlice disagree", k)
			}
		}
	}
}

// TestMixWordSliceHandlesTheEmptyCase pins the zero-word case, which cannot
// arise from a struct the generator accepts but is reachable through the
// helper and must not dereference anything.
func TestMixWordSliceHandlesTheEmptyCase(t *testing.T) {
	for _, seed := range []uint64{0, 1, P0} {
		require.Equal(t, wyBlock(nil, 0, seed), mixWordSlice(nil, seed))
		require.Equal(t, wyBlock(nil, 0, seed), mixWordSlice([]uint64{}, seed))
	}
}

// TestWordOrderMatters checks the property the whole design depends on: a
// permutation of the words must change the hash. If it did not, a struct's
// fields would be interchangeable and {X: 1, Y: 2} would collide with
// {X: 2, Y: 1}.
func TestWordOrderMatters(t *testing.T) {
	const seed = 0x1234_5678_9abc_def0
	for k := 2; k <= 8; k++ {
		base := make([]uint64, k)
		for i := range base {
			base[i] = uint64(i+1) * 0x9e3779b97f4a7c15
		}
		want := callMixer(t, base, seed)

		for i := range k {
			for j := i + 1; j < k; j++ {
				sw := make([]uint64, k)
				copy(sw, base)
				sw[i], sw[j] = sw[j], sw[i]
				require.NotEqualf(t, want, callMixer(t, sw, seed),
					"mixWords%d is blind to swapping words %d and %d", k, i, j)
			}
		}
	}
}

// TestEveryWordReachesTheHash checks that no word is dropped. A mixer that
// forgot an operand would still pass the order test for the words it does
// read, so this flips every bit of every word position and requires each one
// to move the result.
func TestEveryWordReachesTheHash(t *testing.T) {
	const seed = 0xa5a5_5a5a_a5a5_5a5a
	for k := 2; k <= 8; k++ {
		base := make([]uint64, k)
		for i := range base {
			base[i] = uint64(i+1) * 0xff51afd7ed558ccd
		}
		want := callMixer(t, base, seed)

		for pos := range k {
			for bit := range 64 {
				w := make([]uint64, k)
				copy(w, base)
				w[pos] ^= 1 << bit
				require.NotEqualf(t, want, callMixer(t, w, seed),
					"mixWords%d ignores bit %d of word %d", k, bit, pos)
			}
		}
	}
}

// TestWordMixersRespondToTheSeed checks that the seed reaches the output for
// every word count. A mixer that dropped the seed would hash identically for
// every set, which is the one failure a reseed cannot repair.
func TestWordMixersRespondToTheSeed(t *testing.T) {
	for k := 2; k <= 8; k++ {
		w := make([]uint64, k)
		for i := range w {
			w[i] = uint64(i) * 0x2545f4914f6cdd1d
		}
		seen := make(map[uint64]uint64, 64)
		for s := range uint64(64) {
			seed := s * 0x9e3779b97f4a7c15
			h := callMixer(t, w, seed)
			if prev, dup := seen[h]; dup {
				t.Fatalf("mixWords%d gave %#x for seeds %#x and %#x", k, h, prev, seed)
			}
			seen[h] = seed
		}
	}
}
