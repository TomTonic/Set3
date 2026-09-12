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
	// gosec: deliberate reinterpret of an identical struct layout, guarded by
	// TestMaphashSeedLayoutAssumptions in the hashing package.
	(*testRuntimeHasher[K])(unsafe.Pointer(&set.hashFunction)).Fn = fn //nolint:gosec
}

// findHashForGroupAndH2 returns a hash value that maps to targetGroup while
// keeping the lower 7 bits equal to h2.
//
// getGroupIndex reduces a hash with bits.Mul64(hash, groupCount) and keeps the
// high 64 bits, i.e. floor(hash*groupCount / 2^64). The hashes mapping to
// group g therefore form the contiguous range
//
//	ceil(g*2^64/groupCount) <= hash < ceil((g+1)*2^64/groupCount)
//
// so the value can be computed directly. Scanning upwards from a small hash in
// steps of 0x80 does not work: with two groups, reaching group 1 requires a
// hash of at least 2^63, which is roughly 2^56 iterations away.
func findHashForGroupAndH2(groupCount, targetGroup, h2 uint64) uint64 {
	if h2 == 0 || h2 > 0x7f {
		panic("h2 must be in range 1..127")
	}
	if targetGroup >= groupCount {
		panic("targetGroup must be less than groupCount")
	}

	// Lowest hash that maps to targetGroup.
	lowest := ceilShift64(targetGroup, groupCount)

	// Round up to the next hash whose lower 7 bits are h2.
	h := (lowest &^ 0x7f) | h2
	if h < lowest {
		h += 0x80
	}

	if getGroupIndex(h, groupCount) != targetGroup {
		// Only reachable if a group spans fewer than 128 hash values, i.e.
		// groupCount > 2^57. Set3 never allocates anywhere near that.
		panic("no hash with the requested H2 maps to the requested group")
	}
	return h
}

// ceilShift64 returns ceil(num * 2^64 / den). num must be less than den, which
// is what keeps the quotient inside 64 bits.
func ceilShift64(num, den uint64) uint64 {
	quo, rem := bits.Div64(num, 0, den)
	if rem != 0 {
		quo++
	}
	return quo
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
	// Presized well above the handful of keys this uses, so that no rehash can
	// fire in the middle and scramble the arrangement being tested. A tight
	// capacity made this depend on set3maxAvgGroupLoad: at 4.8 the tenth key
	// tripped the element limit and the test failed for a reason that had
	// nothing to do with tombstones.
	set := EmptyWithCapacity[int](64)
	groupCount := uint64(len(set.groupCtrl))
	require.GreaterOrEqual(t, groupCount, uint64(2), "test requires at least two groups")
	defer func() {
		require.Equal(t, groupCount, uint64(len(set.groupCtrl)),
			"the table rehashed during the test, so what it asserts about slot arrangement is meaningless")
	}()

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

			// The freshly inserted key takes the slot of the one just removed,
			// so the pool keeps referring to keys that are actually present.
			// Without this, cycles > len(removePool) would wrap around and try
			// to remove a key that was deleted earlier and never re-added.
			removePool[i%len(removePool)] = newKey

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
				alloc(b, newKey)
				set.Add(newKey)
				present[newKey] = struct{}{}

				// The freshly inserted key takes the slot of the one just
				// removed, so the pool keeps referring to keys that are
				// actually present. Without this, cycles beyond
				// len(removePool) would select already-removed keys and the
				// stillPresent guard above would skip the removal entirely,
				// leaving most of the churn loop doing no work at all.
				bm.removePool[i%len(bm.removePool)] = newKey

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

// --- Tests for the test helpers above --------------------------------------

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

// TestChurnDoesNotGrowTheTableWithoutBound pins the fix for a set that grew
// while its element count stayed put.
//
// resident counts every slot that is not empty, tombstones included, and the
// insert path used to grow whenever resident reached the limit. Remove can clear
// a slot outright only when its group has an empty slot to terminate probes
// with; in a full table it usually does not and leaves a tombstone. Add reuses
// one only when it happens to lie on the probe path of the element being
// inserted.
//
// The keys have to be ones the set has never seen. A window that cycles a
// bounded ring gives every arriving key the home group it had before, so its
// tombstones are reused almost perfectly and the drift never appears — which is
// why the comparison suite's sliding-window scenario could not see this and a
// churn-fresh scenario had to be added next to it. Measured at 262 144 uint64
// keys over twenty window turns, before the fix: 24.94 bytes per element at 36%
// occupancy against 16.62 at 54% after.
//
// One growth is allowed and expected: EmptyWithCapacity leaves the table at
// about 98% of its limit, and a window needs free slots to keep probing short.
// What must not happen is a second one, or a third.
func TestChurnDoesNotGrowTheTableWithoutBound(t *testing.T) {
	const window = 4096
	const turns = 40
	key := func(i uint64) uint64 { return i * 0x9e3779b97f4a7c15 }

	set := EmptyWithCapacity[uint64](window)
	for i := range uint64(window) {
		set.Add(key(i))
	}

	// One window turn to let it settle out of the presized capacity.
	for i := uint64(window); i < 2*window; i++ {
		set.Remove(key(i - window))
		set.Add(key(i))
	}
	settled := len(set.groupCtrl)

	for i := uint64(2 * window); i < turns*window; i++ {
		set.Remove(key(i - window))
		set.Add(key(i))
		require.Equal(t, window, int(set.Size()), "the window changed size, so this is no longer the workload under test")
		require.Equal(t, settled, len(set.groupCtrl),
			"the table grew to %d groups while holding a constant %d elements; tombstone pressure is being "+
				"answered by growing instead of by rehashing in place", len(set.groupCtrl), window)
	}

	// Without tombstones the assertion above would hold for the wrong reason:
	// the workload has to actually produce the pressure it claims to. This also
	// keeps the test meaningful if set3maxAvgGroupLoad is retuned, where a
	// roomier table may never need to rehash at all.
	require.Positive(t, set.dead, "no tombstone survived, so the pressure this test is about never arose")
	t.Logf("%d groups throughout %d window turns, %d tombstones against %d elements",
		settled, turns-1, set.dead, set.Size())
}

// TestMakeRoomGrowsOnlyWhenTheElementsNeedIt checks the decision itself, at both
// sides of the boundary it draws.
//
// Growing when the table is full of live elements is right; growing when it is
// full of tombstones is the bug. Both states are built directly and makeRoom is
// called on them, rather than waiting for a workload to produce them — so the
// test says what the rule is, not merely that some workload comes out well.
func TestMakeRoomGrowsOnlyWhenTheElementsNeedIt(t *testing.T) {
	// fillToLimit adds consecutive elements until the table has no free slots
	// left, which is the state that calls makeRoom.
	fillToLimit := func(set *Set3[uint64]) uint64 {
		var i uint64
		for set.resident < set.elementLimit {
			set.Add(i)
			i++
		}
		return i
	}

	t.Run("tombstones rehash in place", func(t *testing.T) {
		set := EmptyWithCapacity[uint64](1000)
		added := fillToLimit(set)

		// Remove three quarters of the elements. Every removal whose group has
		// no empty slot leaves a tombstone behind.
		for i := range added / 4 * 3 {
			set.Remove(i)
		}
		require.Positive(t, set.dead, "the removals left no tombstones, so this case is not being tested")
		require.LessOrEqual(t, uint64(set.Size())*growthDenominator, uint64(set.elementLimit)*growthNumerator,
			"the set is not below the threshold makeRoom decides on, so this case is not being tested")

		groups := len(set.groupCtrl)
		survivors := set.ToArray()
		set.makeRoom()

		require.Equal(t, groups, len(set.groupCtrl),
			"the table grew although three quarters of its slots were tombstones")
		require.Zero(t, set.dead, "the in-place rehash did not drop the tombstones")
		require.Equal(t, set.Size(), set.resident, "every remaining slot should now hold a live element")
		require.Len(t, survivors, int(set.Size()), "the rehash changed the element count")
		for _, e := range survivors {
			require.True(t, set.Contains(e), "element %d was lost by the in-place rehash", e)
		}
	})

	t.Run("live elements grow the table", func(t *testing.T) {
		set := EmptyWithCapacity[uint64](1000)
		fillToLimit(set)
		require.Zero(t, set.dead, "no element was removed, so there should be no tombstones")

		groups := len(set.groupCtrl)
		size := set.Size()
		set.makeRoom()

		require.Greater(t, len(set.groupCtrl), groups,
			"the table was full of live elements and did not grow")
		require.Equal(t, size, set.Size(), "growing changed the element count")
	})
}
