package set3

import (
	"math/bits"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
	"github.com/stretchr/testify/require"
)

// testRuntimeHasher mirrors hashing.RuntimeHasher layout to inject a custom
// hash function from tests.
type testRuntimeHasher[K comparable] struct {
	Seed uint64
	Fn   hashing.HashFunction
}

// setTestHashFunction overrides the runtime hasher function for deterministic
// collision/probing scenarios in tests.
func setTestHashFunction[K comparable](set *Set3[K], fn hashing.HashFunction) {
	(*testRuntimeHasher[K])(unsafe.Pointer(&set.hashFunction)).Fn = fn
}

// findHashForGroupAndH2 finds a hash value that maps to targetGroup while
// keeping the lower 7 bits equal to h2.
func findHashForGroupAndH2(groupCount, targetGroup, h2 uint64) uint64 {
	if h2 == 0 || h2 > 0x7f {
		panic("h2 must be in range 1..127")
	}
	for h := h2; ; h += 0x80 {
		if getGroupIndex(h, groupCount) == targetGroup {
			return h
		}
	}
}

// findElementSlot scans the table and returns the group/slot for key.
func findElementSlot(set *Set3[int], key int) (uint64, int, bool) {
	for grp := range len(set.groupCtrl) {
		ctrl := set.groupCtrl[grp]
		elemMask := ^ctrl & set3hiBits
		for elemMask != 0 {
			slot := bits.TrailingZeros64(elemMask) >> 3
			elemMask &= elemMask - 1
			if set.groupSlot[grp][slot] == key {
				return uint64(grp), slot, true
			}
		}
	}
	return 0, 0, false
}

// TestAddReusesTombstoneWithoutBreakingOverflowProbe verifies that users can
// still find elements that overflowed into the next bucket after deleting from
// a full bucket and inserting again.
//
// This test covers Set3's probing/tombstone behavior around Add and Remove in
// a forced-collision scenario.
//
// It forces many keys into one start bucket, removes one key to create a
// tombstone, inserts a new key, and asserts that the insertion reuses the
// tombstone while an overflowed key in the next bucket remains reachable.
func TestAddReusesTombstoneWithoutBreakingOverflowProbe(t *testing.T) {
	set := EmptyWithCapacity[int](1)
	groupCount := uint64(len(set.groupCtrl))
	require.GreaterOrEqual(t, groupCount, uint64(2), "test requires at least two groups")

	const targetGroup uint64 = 0
	collidingKeys := []int{101, 102, 103, 104, 105, 106, 107, 108, 109}
	keyToRemove := 103
	keyToInsert := 203

	hashByKey := make(map[int]uint64, len(collidingKeys)+1)
	for i, k := range collidingKeys {
		h2 := uint64(i + 1) // keep H2 deterministic and non-zero
		hashByKey[k] = findHashForGroupAndH2(groupCount, targetGroup, h2)
	}
	hashByKey[keyToInsert] = findHashForGroupAndH2(groupCount, targetGroup, uint64(len(collidingKeys)+1))

	setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
		k := *(*int)(ptr)
		h, ok := hashByKey[k]
		if !ok {
			return findHashForGroupAndH2(groupCount, targetGroup, 0x7f)
		}
		return h
	})

	for _, k := range collidingKeys {
		set.Add(k)
	}

	spillKey := collidingKeys[len(collidingKeys)-1]
	spillGroup, _, found := findElementSlot(set, spillKey)
	require.True(t, found, "spill key must be present")
	require.NotEqual(t, targetGroup, spillGroup, "last colliding key should have overflowed to next group")

	removed := set.Remove(keyToRemove)
	require.True(t, removed)
	require.Equal(t, uint32(1), set.dead, "remove from full bucket must create tombstone")

	deletedMask := set3ctlrMatchDeleted(set.groupCtrl[targetGroup])
	require.NotZero(t, deletedMask, "target group should contain a tombstone")
	deletedSlot := bits.TrailingZeros64(deletedMask) >> 3

	set.Add(keyToInsert)

	require.True(t, set.Contains(keyToInsert), "reinserted key must be found")
	require.True(t, set.Contains(spillKey), "overflowed key must still be found after tombstone reuse")
	require.False(t, set.Contains(keyToRemove), "removed key must stay absent")
	require.Equal(t, uint32(0), set.dead, "reusing the tombstone should clear dead counter")

	insertGroup, insertSlot, insertFound := findElementSlot(set, keyToInsert)
	require.True(t, insertFound)
	require.Equal(t, targetGroup, insertGroup, "new key should be inserted into original group tombstone")
	require.Equal(t, deletedSlot, insertSlot, "new key should reuse the tombstone slot")
}

// FuzzTombstoneReuseProbeChain verifies from a user perspective that repeated
// remove/add churn under heavy collisions does not make existing keys
// undiscoverable.
//
// This fuzz test targets Set3 probing and tombstone handling in a forced
// single-start-bucket workload.
//
// It repeatedly deletes and reinserts colliding keys while asserting that at
// least one overflowed key in a following bucket stays findable and that all
// currently present keys are still returned by Contains.
func FuzzTombstoneReuseProbeChain(f *testing.F) {
	// seed: varying start-bucket positions
	f.Add(uint64(0), uint8(8))
	f.Add(uint64(1), uint8(16))
	f.Add(uint64(7), uint8(32))
	f.Add(uint64(17), uint8(64))
	// edge: wraparound bucket (last group → group 0)
	f.Add(uint64(0xFFFF_FFFF_FFFF_FFFF), uint8(1))
	// edge: minimum cycles
	f.Add(uint64(3), uint8(1))
	// edge: maximum cycles cap
	f.Add(uint64(13), uint8(255))
	// prime-count table sizes tend to land at specific group counts
	f.Add(uint64(11), uint8(8))
	f.Add(uint64(23), uint8(48))
	f.Add(uint64(97), uint8(96))

	f.Fuzz(func(t *testing.T, seed uint64, cycles uint8) {
		set := EmptyWithCapacity[int](1)
		groupCount := uint64(len(set.groupCtrl))
		if groupCount < 2 {
			t.Skip("test requires at least two groups")
		}

		targetGroup := seed % groupCount
		if cycles == 0 {
			cycles = 1
		}
		if cycles > 96 {
			cycles = 96
		}

		base := int(seed&0x00FF_FFFF) * 1000
		baseCount := 10 // 8 fill target group, additional keys force overflow
		baseKeys := make([]int, 0, baseCount)
		for i := range baseCount {
			baseKeys = append(baseKeys, base+i+1)
		}

		hashByKey := make(map[int]uint64, baseCount+int(cycles)+16)
		nextH2 := uint64(1)
		for _, k := range baseKeys {
			hashByKey[k] = findHashForGroupAndH2(groupCount, targetGroup, nextH2)
			nextH2++
		}

		setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
			k := *(*int)(ptr)
			h, ok := hashByKey[k]
			if !ok {
				h = findHashForGroupAndH2(groupCount, targetGroup, nextH2)
				hashByKey[k] = h
				nextH2++
				if nextH2 > 0x7f {
					nextH2 = 1
				}
			}
			return h
		})

		present := make(map[int]struct{}, len(baseKeys)+int(cycles))
		for _, k := range baseKeys {
			set.Add(k)
			present[k] = struct{}{}
		}

		overflowKey := 0
		for _, k := range baseKeys {
			grp, _, found := findElementSlot(set, k)
			if found && grp != targetGroup {
				overflowKey = k
				break
			}
		}
		require.NotZero(t, overflowKey, "at least one key must overflow to the next group")

		removePool := make([]int, 0, len(baseKeys)-1)
		for _, k := range baseKeys {
			if k != overflowKey {
				removePool = append(removePool, k)
			}
		}

		for i := 0; i < int(cycles); i++ {
			removeKey := removePool[i%len(removePool)]
			removed := set.Remove(removeKey)
			require.True(t, removed, "remove must succeed")
			delete(present, removeKey)

			newKey := base + 10_000 + i
			set.Add(newKey)
			present[newKey] = struct{}{}

			require.True(t, set.Contains(overflowKey), "overflow key must remain reachable")
			require.False(t, set.Contains(removeKey), "removed key must stay absent")
			for k := range present {
				require.True(t, set.Contains(k), "present key must be discoverable")
			}
		}
	})
}

// FuzzTombstoneReuseMultiBucket verifies from a user perspective that
// delete/reinsert churn across multiple colliding bucket chains does not lose
// any key that is supposed to still be present.
//
// This fuzz test targets Set3's probing correctness under simultaneous
// tombstone pressure in several bucket chains within the same table.
//
// It fills numBuckets different start buckets with overflowing key sets and
// then runs cycles of remove/reinsert on each chain in sequence, checking
// after every operation that every currently-present key remains reachable
// and every removed key is absent.
func FuzzTombstoneReuseMultiBucket(f *testing.F) {
	// (seed, numBuckets, cycles)
	f.Add(uint64(0), uint8(2), uint8(4))
	f.Add(uint64(1), uint8(3), uint8(8))
	f.Add(uint64(7), uint8(2), uint8(32))
	f.Add(uint64(13), uint8(4), uint8(16))
	// edge: single extra bucket (minimum multi case)
	f.Add(uint64(5), uint8(1), uint8(8))
	// edge: many buckets, few cycles
	f.Add(uint64(99), uint8(6), uint8(1))
	// edge: wraparound
	f.Add(uint64(0xFFFF_FFFF_FFFF_FFFF), uint8(2), uint8(8))
	// edge: max cycles cap
	f.Add(uint64(3), uint8(2), uint8(255))

	f.Fuzz(func(t *testing.T, seed uint64, numBuckets, cycles uint8) {
		if numBuckets == 0 {
			numBuckets = 1
		}
		if numBuckets > 6 {
			numBuckets = 6
		}
		if cycles == 0 {
			cycles = 1
		}
		if cycles > 64 {
			cycles = 64
		}

		// size the set so it has enough groups for all buckets plus spacing
		capacity := uint32(numBuckets)*10 + 4
		set := EmptyWithCapacity[int](capacity)
		groupCount := uint64(len(set.groupCtrl))
		if groupCount < uint64(numBuckets)+1 {
			t.Skip("not enough groups for requested bucket count")
		}

		hashByKey := make(map[int]uint64, int(numBuckets)*12+int(cycles)*int(numBuckets)+16)
		h2Counters := make([]uint64, numBuckets)
		for i := range h2Counters {
			h2Counters[i] = 1
		}

		alloc := func(bucket uint8, key int) {
			if _, ok := hashByKey[key]; ok {
				return
			}
			tg := (seed + uint64(bucket)) % groupCount
			h2 := h2Counters[bucket]
			hashByKey[key] = findHashForGroupAndH2(groupCount, tg, h2)
			h2Counters[bucket]++
			if h2Counters[bucket] > 0x7f {
				h2Counters[bucket] = 1
			}
		}

		setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
			k := *(*int)(ptr)
			if h, ok := hashByKey[k]; ok {
				return h
			}
			// unknown key → bucket 0 as fallback; assign deterministically
			alloc(0, k)
			return hashByKey[k]
		})

		// build base keys per bucket: 10 keys each (fills group + overflows)
		type bucketMeta struct {
			baseKeys    []int
			overflowKey int
			removePool  []int
		}
		buckets := make([]bucketMeta, numBuckets)
		base := int(seed&0x00FF_FFFF)*10_000 + 1
		present := make(map[int]struct{}, int(numBuckets)*12)

		for b := range numBuckets {
			bm := &buckets[b]
			tg := (seed + uint64(b)) % groupCount
			for i := range 10 {
				k := base + int(b)*1000 + i
				alloc(b, k)
				bm.baseKeys = append(bm.baseKeys, k)
				set.Add(k)
				present[k] = struct{}{}
			}
			// find the overflow key (any key that landed outside tg)
			for _, k := range bm.baseKeys {
				grp, _, found := findElementSlot(set, k)
				if found && grp != tg {
					bm.overflowKey = k
					break
				}
			}
			require.NotZero(t, bm.overflowKey, "bucket %d must produce an overflow key", b)
			for _, k := range bm.baseKeys {
				if k != bm.overflowKey {
					bm.removePool = append(bm.removePool, k)
				}
			}
		}

		// churn: cycle through all buckets
		for i := range int(cycles) {
			for b := range numBuckets {
				bm := &buckets[b]
				removeKey := bm.removePool[i%len(bm.removePool)]

				if _, stillPresent := present[removeKey]; stillPresent {
					removed := set.Remove(removeKey)
					require.True(t, removed, "remove must succeed for key %d", removeKey)
					delete(present, removeKey)
				}

				newKey := base + int(b)*1000 + 10_000 + i
				alloc(uint8(b), newKey) //nolint:gosec
				set.Add(newKey)
				present[newKey] = struct{}{}

				// verify invariants after each churn step
				require.True(t, set.Contains(bm.overflowKey),
					"overflow key %d for bucket %d must stay reachable", bm.overflowKey, b)
				require.False(t, set.Contains(removeKey),
					"removed key %d must stay absent", removeKey)
			}
		}

		// final sweep: every key in present must be findable
		for k := range present {
			require.True(t, set.Contains(k), "key %d must be reachable in final sweep", k)
		}
	})
}
