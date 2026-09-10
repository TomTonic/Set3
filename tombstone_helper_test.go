package set3

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestFindHashForGroupAndH2HitsEveryGroup verifies the helper for every group
// of a range of table sizes. The previous implementation scanned upwards in
// steps of 0x80 and could only ever reach group 0, because getGroupIndex
// reduces over the high bits of the product: with two groups, group 1 starts
// at hash 2^63. Any target group other than 0 made it spin forever.
func TestFindHashForGroupAndH2HitsEveryGroup(t *testing.T) {
	for _, groupCount := range []uint64{2, 3, 5, 7, 11, 23, 97, 1021, 65537} {
		for targetGroup := uint64(0); targetGroup < groupCount; targetGroup++ {
			for _, h2 := range []uint64{1, 2, 0x3f, 0x7f} {
				h := findHashForGroupAndH2(groupCount, targetGroup, h2)
				require.Equal(t, targetGroup, getGroupIndex(h, groupCount),
					"groupCount=%d targetGroup=%d h2=%#x -> h=%#x", groupCount, targetGroup, h2, h)
				require.Equal(t, h2, h&0x7f,
					"groupCount=%d targetGroup=%d h2=%#x -> h=%#x", groupCount, targetGroup, h2, h)
			}
		}
	}
}

// TestFindHashForGroupAndH2IsDeterministic keeps the helper usable as a stand-in
// hash function: the same inputs must always yield the same hash.
func TestFindHashForGroupAndH2IsDeterministic(t *testing.T) {
	a := findHashForGroupAndH2(97, 42, 0x13)
	b := findHashForGroupAndH2(97, 42, 0x13)
	require.Equal(t, a, b)
}

// TestFindHashForGroupAndH2DistinctH2 verifies that different H2 values yield
// different hashes in the same group, which the callers rely on to place
// several distinct keys into one start bucket.
func TestFindHashForGroupAndH2DistinctH2(t *testing.T) {
	seen := make(map[uint64]bool)
	for h2 := uint64(1); h2 <= 0x7f; h2++ {
		h := findHashForGroupAndH2(11, 5, h2)
		require.False(t, seen[h], "duplicate hash for h2=%#x", h2)
		seen[h] = true
		require.Equal(t, uint64(5), getGroupIndex(h, 11))
	}
}

func TestFindHashForGroupAndH2RejectsBadInput(t *testing.T) {
	require.Panics(t, func() { findHashForGroupAndH2(11, 0, 0) })
	require.Panics(t, func() { findHashForGroupAndH2(11, 0, 0x80) })
	require.Panics(t, func() { findHashForGroupAndH2(11, 11, 1) })
}
