package set3

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestRandomSeedIsNonZero verifies the invariant the helper exists for.
func TestRandomSeedIsNonZero(t *testing.T) {
	for range 10000 {
		require.NotZero(t, randomSeed())
	}
}

// TestRandomSeedUsesFullWidth guards against a repeat of the truncated-mask
// bug, where a 15-digit hex mask (0xFFFFFFFFFFFFFFE) silently pinned the top
// four bits of every seed to zero. Every bit position must be observed set at
// least once across a reasonable number of draws.
func TestRandomSeedUsesFullWidth(t *testing.T) {
	var seen uint64
	for range 10000 {
		seen |= randomSeed()
	}
	require.Equal(t, ^uint64(0), seen, "every one of the 64 seed bits must be reachable")
}
