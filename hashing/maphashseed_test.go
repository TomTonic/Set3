package hashing

import (
	"hash/maphash"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// SeedToMaphashSeed writes eight bytes into a maphash.Seed through unsafe,
// which relies on the concrete layout of that type. The layout is not
// guaranteed by the spec, so these tests pin the assumptions: if a future Go
// release changes maphash.Seed, they fail loudly instead of letting the
// fallback hasher run on a silently corrupted seed.

func TestMaphashSeedLayoutAssumptions(t *testing.T) {
	require.Equal(t, uintptr(8), unsafe.Sizeof(maphash.Seed{}),
		"SeedToMaphashSeed writes 8 bytes into maphash.Seed")
}

// TestSeedToMaphashSeedIsDeterministic verifies the property the unsafe write
// exists for: unlike maphash.MakeSeed, the mapping is reproducible.
func TestSeedToMaphashSeedIsDeterministic(t *testing.T) {
	require.Equal(t, SeedToMaphashSeed(0x1234), SeedToMaphashSeed(0x1234))
	require.NotEqual(t, SeedToMaphashSeed(0x1234), SeedToMaphashSeed(0x5678))
}

// TestSeedToMaphashSeedIsUsable verifies the produced Seed is accepted by the
// stdlib. maphash panics on a zero Seed, so this also covers the seed == 0
// special case.
func TestSeedToMaphashSeedIsUsable(t *testing.T) {
	for _, seed := range []uint64{0, 1, 0x1234, ^uint64(0)} {
		require.NotPanics(t, func() {
			_ = maphash.Comparable(SeedToMaphashSeed(seed), 42)
		}, "seed %#x must produce a usable maphash.Seed", seed)
	}
}

// TestSeedToMaphashSeedDistinguishesSeeds verifies the seed actually reaches
// the hash, i.e. the unsafe write landed in the field maphash reads.
func TestSeedToMaphashSeedDistinguishesSeeds(t *testing.T) {
	a := maphash.Comparable(SeedToMaphashSeed(0x1111), 42)
	b := maphash.Comparable(SeedToMaphashSeed(0x2222), 42)
	require.NotEqual(t, a, b, "different seeds must produce different hashes")
}
