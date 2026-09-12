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
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// compactInPlace rearranges a live table through unsafe-looking index
// arithmetic and has three paths that are easy to get subtly wrong: leaving an
// element where it is, moving it into an empty slot, and swapping it with an
// element that has not been placed yet. A mistake in any of them does not
// crash; it loses an element, or leaves one where a probe can no longer reach
// it, which is a silent wrong answer from Contains much later.
//
// So the tests here do not sample and hope. They check the invariants after
// every shape of table this can produce, they force the overflow chains that
// the rare paths need, and they compare against the full rehash that has always
// been there.

// checkCompacted asserts everything that must hold of a table that has just
// been compacted, and is called from every test below.
func checkCompacted[T comparable](t *testing.T, set *Set3[T], want []T, groupsBefore int) {
	t.Helper()

	require.Equal(t, groupsBefore, len(set.groupCtrl), "compaction must not change the table's size")
	require.Zero(t, set.dead, "compaction must leave no tombstone behind")
	require.Equal(t, uint32(len(want)), set.Size(), "compaction changed the element count") //nolint:gosec
	require.Equal(t, set.Size(), set.resident, "after compaction every non-empty slot must hold a live element")

	// No control byte may still say DELETED, whatever the accounting claims.
	for g, ctrl := range set.groupCtrl {
		for s := range set3groupSize {
			require.NotEqual(t, set3Deleted, ctrlByteAt(ctrl, s),
				"group %d slot %d is still marked deleted", g, s)
		}
	}

	// The occupied slots must be exactly the elements, counted independently of
	// the accounting fields.
	occupied := 0
	for _, ctrl := range set.groupCtrl {
		for s := range set3groupSize {
			if ctrlByteAt(ctrl, s) != set3Empty {
				occupied++
			}
		}
	}
	require.Equal(t, len(want), occupied, "the number of occupied slots does not match the number of elements")

	for _, e := range want {
		require.True(t, set.Contains(e), "element %v cannot be found after compaction", e)
	}
	got := set.ToArray()
	require.Len(t, got, len(want), "iteration returns a different number of elements than were put in")
	seen := make(map[T]struct{}, len(got))
	for _, e := range got {
		seen[e] = struct{}{}
	}
	for _, e := range want {
		_, ok := seen[e]
		require.True(t, ok, "element %v is not reachable by iteration after compaction", e)
	}
}

// TestCompactInPlaceKeepsEveryElement walks a wide range of table shapes and
// removal patterns.
//
// The patterns matter as much as the sizes. Removing a contiguous run, every
// other element, or a random scatter produce quite different tombstone layouts,
// and which of compaction's three paths get exercised depends on that layout.
func TestCompactInPlaceKeepsEveryElement(t *testing.T) {
	patterns := []struct {
		name string
		keep func(i, n int, r *rand.Rand) bool
	}{
		{"remove the first half", func(i, n int, _ *rand.Rand) bool { return i >= n/2 }},
		{"remove the second half", func(i, n int, _ *rand.Rand) bool { return i < n/2 }},
		{"remove every other", func(i, _ int, _ *rand.Rand) bool { return i%2 == 0 }},
		{"remove three of every four", func(i, _ int, _ *rand.Rand) bool { return i%4 == 0 }},
		{"remove at random", func(_, _ int, r *rand.Rand) bool { return r.IntN(2) == 0 }},
		{"remove all but one", func(i, _ int, _ *rand.Rand) bool { return i == 0 }},
		{"remove all", func(_, _ int, _ *rand.Rand) bool { return false }},
		{"remove none", func(_, _ int, _ *rand.Rand) bool { return true }},
	}

	for _, n := range []int{1, 2, 7, 8, 9, 17, 64, 100, 511, 1000, 5000} {
		for _, p := range patterns {
			t.Run(fmt.Sprintf("n=%d/%s", n, p.name), func(t *testing.T) {
				r := rand.New(rand.NewPCG(uint64(n), 42)) //nolint:gosec
				// Presized clear of its limit so that nothing rehashes behind
				// the test's back and the compaction under test is the only
				// rearrangement that happens.
				set := EmptyWithCapacity[uint64](uint32(4 * n)) //nolint:gosec
				groups := len(set.groupCtrl)

				var want []uint64
				for i := range n {
					key := uint64(i) * 0x9e3779b97f4a7c15
					set.Add(key)
					if p.keep(i, n, r) {
						want = append(want, key)
					}
				}
				for i := range n {
					key := uint64(i) * 0x9e3779b97f4a7c15
					if !slices.Contains(want, key) {
						require.True(t, set.Remove(key))
					}
				}
				require.Equal(t, groups, len(set.groupCtrl), "the table resized while the test was setting up")

				set.compactInPlace()
				checkCompacted(t, set, want, groups)

				// And it must still behave like a set afterwards: re-adding the
				// removed keys has to work and has to find the reclaimed slots.
				for i := range n {
					set.Add(uint64(i) * 0x9e3779b97f4a7c15)
				}
				require.Equal(t, uint32(n), set.Size(), "re-adding after compaction produced the wrong size") //nolint:gosec
				for i := range n {
					require.True(t, set.Contains(uint64(i)*0x9e3779b97f4a7c15))
				}
			})
		}
	}
}

// TestCompactInPlaceWithForcedOverflowChains is the test the rare paths need.
//
// Compaction only has to move an element when its home group is full, and it
// only has to swap when the slot it wants is held by another element that has
// not been placed yet. Random keys spread over a roomy table almost never
// produce either. Here every key is given a hash that lands on the same home
// group, so the group overflows into the ones after it and every path is taken.
func TestCompactInPlaceWithForcedOverflowChains(t *testing.T) {
	for _, homeGroup := range []uint64{0, 1, 3} {
		for _, keyCount := range []int{9, 16, 17, 24, 33} {
			for _, removeEvery := range []int{2, 3, 5} {
				name := fmt.Sprintf("home=%d/keys=%d/removeEvery=%d", homeGroup, keyCount, removeEvery)
				t.Run(name, func(t *testing.T) {
					set := EmptyWithCapacity[int](uint32(8 * keyCount)) //nolint:gosec
					groupCount := uint64(len(set.groupCtrl))
					require.Greater(t, groupCount, homeGroup, "test needs more groups than the home group index")
					groups := len(set.groupCtrl)

					// Every key hashes into the same group, with H2 values that
					// repeat on purpose: identical tags force the probe to
					// compare elements rather than short-circuit on the tag.
					hashByKey := make(map[int]uint64, keyCount)
					for i := range keyCount {
						h2 := uint64(i%0x7f) + 1
						hashByKey[i] = findHashForGroupAndH2(groupCount, homeGroup, h2)
					}
					setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
						return hashByKey[*(*int)(ptr)]
					})

					for i := range keyCount {
						set.Add(i)
					}
					require.Equal(t, uint32(keyCount), set.Size()) //nolint:gosec

					var want []int
					for i := range keyCount {
						if i%removeEvery == 0 {
							require.True(t, set.Remove(i), "key %d should have been present", i)
						} else {
							want = append(want, i)
						}
					}
					require.Positive(t, set.dead, "this arrangement was supposed to produce tombstones")
					require.Equal(t, groups, len(set.groupCtrl), "the table resized while the test was setting up")

					set.compactInPlace()
					checkCompacted(t, set, want, groups)
				})
			}
		}
	}
}

// TestCompactInPlaceMatchesAFullRehash pins the new implementation to the old
// one that has always been there.
//
// rehashToNumGroups at the current size does the same job by allocating a
// replacement table and replaying every element into it. That is the reference:
// whatever compaction does to the arrangement, the set it represents has to be
// identical.
func TestCompactInPlaceMatchesAFullRehash(t *testing.T) {
	for _, n := range []int{1, 8, 33, 257, 2000} {
		for _, removeEvery := range []int{2, 3, 7} {
			t.Run(fmt.Sprintf("n=%d/removeEvery=%d", n, removeEvery), func(t *testing.T) {
				build := func() *Set3[uint64] {
					s := EmptyWithCapacity[uint64](uint32(4 * n)) //nolint:gosec
					for i := range n {
						s.Add(uint64(i) * 0x9e3779b97f4a7c15)
					}
					for i := range n {
						if i%removeEvery == 0 {
							s.Remove(uint64(i) * 0x9e3779b97f4a7c15)
						}
					}
					return s
				}
				compacted, rehashed := build(), build()
				groups := len(compacted.groupCtrl)

				compacted.compactInPlace()
				rehashed.rehashToNumGroups(uint32(len(rehashed.groupCtrl))) //nolint:gosec

				require.True(t, compacted.Equals(rehashed), "compaction and a full rehash disagree about the set's contents")
				require.Equal(t, rehashed.Size(), compacted.Size())
				require.Equal(t, groups, len(compacted.groupCtrl), "compaction changed the table's size")
				require.Equal(t, rehashed.resident, compacted.resident, "the two disagree about how many slots are occupied")
			})
		}
	}
}

// TestCompactInPlaceAllocatesNothing is the reason this function exists at all.
//
// A full rehash at the same size allocates the whole backing store again. If
// compaction ever starts doing that, the only thing it still has over the older
// path is a longer implementation, so this failing means the function should be
// deleted rather than fixed.
func TestCompactInPlaceAllocatesNothing(t *testing.T) {
	// Measured over several independently prepared tables, and judged by the
	// smallest result rather than by one measurement.
	//
	// testing.AllocsPerRun counts allocations process-wide, and one run of a
	// function this fast can pick up a stray allocation from another package's
	// tests running in parallel — which is how this test failed once while the
	// code was correct. A function that really allocates does so on every run,
	// so the minimum is the honest statistic and noise can only push it up.
	const measurements = 5
	minAllocs := -1.0
	for range measurements {
		set, want, groups := crowdedSet(t, 3, 64)
		require.Positive(t, set.dead, "the setup produced no tombstones, so this measures nothing")
		// One run per table: compaction is idempotent, so a second call on the
		// same table has nothing to do and would measure an early return.
		got := testing.AllocsPerRun(1, func() { set.compactInPlace() })
		if minAllocs < 0 || got < minAllocs {
			minAllocs = got
		}
		// Allocating nothing is only interesting if the table survived, and a
		// compaction that silently dropped everything would allocate nothing
		// too.
		checkCompacted(t, set, want, groups)
	}
	require.Zero(t, minAllocs, "compactInPlace allocated on every one of %d measurements; the whole point of it is that it does not", measurements)

	// And the measurement has to be able to see an allocation at all, or the
	// assertion above would pass for a function that was never called.
	control, _, _ := crowdedSet(t, 3, 64)    //nolint:dogsled
	groups := uint32(len(control.groupCtrl)) //nolint:gosec
	rehashAllocs := testing.AllocsPerRun(1, func() { control.rehashToNumGroups(groups) })
	require.Positive(t, rehashAllocs,
		"a full rehash at the same size reported no allocation either, so this test cannot tell the two apart")
}

// crowdedSet builds a table whose groups are genuinely full, removes every
// removeEvery-th element, and returns what should be left.
//
// Removals only leave tombstones when their group has no empty slot, so a
// roomy table produces none at all and a test built on one measures nothing.
// Forcing every key onto one home group guarantees the overflow chains and the
// full groups that make tombstones unavoidable.
func crowdedSet(t *testing.T, removeEvery, keyCount int) (*Set3[int], []int, int) {
	t.Helper()
	set := EmptyWithCapacity[int](uint32(8 * keyCount)) //nolint:gosec
	groupCount := uint64(len(set.groupCtrl))
	groups := len(set.groupCtrl)

	hashByKey := make(map[int]uint64, keyCount)
	for i := range keyCount {
		hashByKey[i] = findHashForGroupAndH2(groupCount, 0, uint64(i%0x7f)+1)
	}
	setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
		return hashByKey[*(*int)(ptr)]
	})

	for i := range keyCount {
		set.Add(i)
	}
	var want []int
	for i := range keyCount {
		if i%removeEvery == 0 {
			require.True(t, set.Remove(i), "key %d should have been present", i)
		} else {
			want = append(want, i)
		}
	}
	require.Equal(t, groups, len(set.groupCtrl), "the table resized while the test was setting up")
	return set, want, groups
}

// TestCompactInPlaceKeepsTheSeed pins the one way this deliberately differs
// from rehashToNumGroups.
//
// A rehash that grows draws a fresh seed, because a rehash triggered by a
// collision pattern wants that pattern broken up. Compaction must not: it moves
// elements within the home groups they already have, and a new seed would send
// them somewhere else entirely while the control bytes still said otherwise.
func TestCompactInPlaceKeepsTheSeed(t *testing.T) {
	set := EmptyWithCapacity[uint64](4096)
	for i := range uint64(1000) {
		set.Add(i)
	}
	for i := range uint64(1000) {
		if i%2 == 0 {
			set.Remove(i)
		}
	}
	seed := set.hashFunction.Seed
	set.compactInPlace()
	require.Equal(t, seed, set.hashFunction.Seed, "compaction reseeded the hash, which invalidates every control byte")
}

// TestCompactInPlaceIsANoOpWithoutTombstones checks the early return, including
// that it does not quietly disturb a table that is already clean.
func TestCompactInPlaceIsANoOpWithoutTombstones(t *testing.T) {
	for _, n := range []int{0, 1, 100} {
		set := EmptyWithCapacity[uint64](uint32(4*n + 1)) //nolint:gosec
		for i := range uint64(n) {                        //nolint:gosec
			set.Add(i)
		}
		before := slices.Clone(set.groupCtrl)
		resident, dead := set.resident, set.dead
		set.compactInPlace()
		require.Equal(t, before, set.groupCtrl, "n=%d: compaction rearranged a table that had no tombstones", n)
		require.Equal(t, resident, set.resident, "n=%d", n)
		require.Equal(t, dead, set.dead, "n=%d", n)
	}
}

// TestCompactInPlaceWithASingleTombstoneInACrowdedTable is the worst case for
// the placement loop: almost every slot is occupied, so the probe has hardly
// anywhere to put anything and the swap path runs again and again.
func TestCompactInPlaceWithASingleTombstoneInACrowdedTable(t *testing.T) {
	for _, keyCount := range []int{16, 40, 100} {
		t.Run(fmt.Sprintf("keys=%d", keyCount), func(t *testing.T) {
			set := EmptyWithCapacity[int](uint32(8 * keyCount)) //nolint:gosec
			groupCount := uint64(len(set.groupCtrl))
			groups := len(set.groupCtrl)

			hashByKey := make(map[int]uint64, keyCount)
			for i := range keyCount {
				hashByKey[i] = findHashForGroupAndH2(groupCount, 0, uint64(i%0x7f)+1)
			}
			setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
				return hashByKey[*(*int)(ptr)]
			})
			for i := range keyCount {
				set.Add(i)
			}

			// One element from the middle of the overflow chain: the removal
			// most likely to leave a tombstone that other elements must probe
			// straight through.
			victim := keyCount / 2
			require.True(t, set.Remove(victim))
			require.Equal(t, uint32(1), set.dead, "a removal from the middle of a full chain must leave exactly one tombstone")

			var want []int
			for i := range keyCount {
				if i != victim {
					want = append(want, i)
				}
			}
			set.compactInPlace()
			checkCompacted(t, set, want, groups)
		})
	}
}

// TestConvertDeletedToEmptyAndFullToDeleted checks the word-at-a-time first
// pass against a byte-at-a-time reference, over every byte value.
//
// It is bit arithmetic on a packed control word, which is exactly the kind of
// code that works for the cases someone thought of.
func TestConvertDeletedToEmptyAndFullToDeleted(t *testing.T) {
	reference := func(b uint64) uint64 {
		if b&0x80 != 0 { // EMPTY or DELETED
			return set3Empty
		}
		return set3Deleted
	}
	// Every byte value in every position, then a spread of mixed words.
	for v := range uint64(256) {
		for pos := range set3groupSize {
			ctrl := setCTRLat(set3AllEmpty, v, pos)
			got := convertDeletedToEmptyAndFullToDeleted(ctrl)
			for s := range set3groupSize {
				want := reference(ctrlByteAt(ctrl, s))
				require.Equal(t, want, ctrlByteAt(got, s),
					"byte %#x at position %d: slot %d converted wrongly", v, pos, s)
			}
		}
	}
	r := rand.New(rand.NewPCG(7, 9)) //nolint:gosec
	for range 10000 {
		ctrl := r.Uint64()
		got := convertDeletedToEmptyAndFullToDeleted(ctrl)
		for s := range set3groupSize {
			require.Equal(t, reference(ctrlByteAt(ctrl, s)), ctrlByteAt(got, s), "word %#x slot %d", ctrl, s)
		}
	}
}

// TestFirstNonFullProbesLikeAdd pins the probe order to the one the lookup path
// uses. An element placed by a different sequence would simply not be found.
func TestFirstNonFullProbesLikeAdd(t *testing.T) {
	set := EmptyWithCapacity[uint64](64)
	groupCount := uint64(len(set.groupCtrl))
	require.GreaterOrEqual(t, groupCount, uint64(3))

	// Fill group 1 completely, leave the others untouched.
	for s := range set3groupSize {
		set.groupCtrl[1] = setCTRLat(set.groupCtrl[1], uint64(s)+1, s)
	}
	// A hash whose home group is 1 must skip to group 2.
	h := findHashForGroupAndH2(groupCount, 1, 5)
	g, slot := firstNonFull(set.groupCtrl, groupCount, h)
	require.Equal(t, uint64(2), g, "a full home group must be skipped, in ascending order")
	require.Equal(t, 0, slot, "the first free slot of the target group is the one to use")

	// A hash whose home group is free must stay there.
	h0 := findHashForGroupAndH2(groupCount, 0, 5)
	g0, slot0 := firstNonFull(set.groupCtrl, groupCount, h0)
	require.Equal(t, uint64(0), g0)
	require.Equal(t, 0, slot0)

	// Wraparound: fill the last group and probe from it.
	last := groupCount - 1
	for s := range set3groupSize {
		set.groupCtrl[last] = setCTRLat(set.groupCtrl[last], uint64(s)+1, s)
	}
	hl := findHashForGroupAndH2(groupCount, last, 5)
	gl, _ := firstNonFull(set.groupCtrl, groupCount, hl)
	require.Equal(t, uint64(0), gl, "probing past the last group must wrap to the first")
}

// FuzzCompactInPlace drives arbitrary sequences of insertion, removal and
// compaction against a plain map and requires the two to agree at every step.
func FuzzCompactInPlace(f *testing.F) {
	f.Add([]byte{1, 2, 3, 0, 4, 5}, uint16(16))
	f.Add([]byte{}, uint16(1))
	f.Add([]byte{7, 7, 7, 7, 7, 7, 7, 7, 7}, uint16(2))

	f.Fuzz(func(t *testing.T, ops []byte, capacity uint16) {
		set := EmptyWithCapacity[uint64](uint32(capacity) % 4096)
		reference := map[uint64]struct{}{}

		for i, op := range ops {
			key := uint64(op) % 64
			switch i % 3 {
			case 0, 1:
				set.Add(key)
				reference[key] = struct{}{}
			case 2:
				set.Remove(key)
				delete(reference, key)
			}
			if op%16 == 0 {
				set.compactInPlace()
				require.Zero(t, set.dead, "compaction left a tombstone")
			}
		}
		set.compactInPlace()

		require.Equal(t, uint32(len(reference)), set.Size(), "size diverged from the reference map") //nolint:gosec
		for k := range reference {
			require.True(t, set.Contains(k), "element %d lost", k)
		}
		for _, k := range set.ToArray() {
			_, ok := reference[k]
			require.True(t, ok, "element %d appeared from nowhere", k)
		}
	})
}

// TestCompactInPlaceSwapsWhenTheTargetIsStillOccupied covers the third and
// hardest path, which none of the tests above reach.
//
// Compaction walks groups in ascending order, so by the time it processes group
// g the groups before it are settled and the groups after it are untouched. An
// element it wants to move therefore lands in an empty slot almost always — the
// only way the target can be a slot that still holds an unplaced element is if
// the element's home group lies *after* g, which means it got to g by wrapping
// past the end of the table.
//
// So this forces every key onto the last group. They fill it, spill past the
// end and wrap around to group 0. Compaction then starts at group 0 holding
// elements whose home is the last group, finds that group full of unplaced
// elements, and has to swap.
//
// This test is the only cover for that path. If it is ever changed, check that
// compactInPlace still reports full statement coverage before believing it.
func TestCompactInPlaceSwapsWhenTheTargetIsStillOccupied(t *testing.T) {
	for _, keyCount := range []int{12, 17, 25, 40} {
		for _, removeEvery := range []int{3, 4, 7} {
			t.Run(fmt.Sprintf("keys=%d/removeEvery=%d", keyCount, removeEvery), func(t *testing.T) {
				set := EmptyWithCapacity[int](uint32(8 * keyCount)) //nolint:gosec
				groupCount := uint64(len(set.groupCtrl))
				groups := len(set.groupCtrl)
				homeGroup := groupCount - 1

				hashByKey := make(map[int]uint64, keyCount)
				for i := range keyCount {
					hashByKey[i] = findHashForGroupAndH2(groupCount, homeGroup, uint64(i%0x7f)+1)
				}
				setTestHashFunction(set, func(ptr unsafe.Pointer, _ uint64) uint64 {
					return hashByKey[*(*int)(ptr)]
				})

				for i := range keyCount {
					set.Add(i)
				}
				require.Equal(t, uint32(keyCount), set.Size()) //nolint:gosec

				// The keys have to have wrapped, or the swap path is not
				// reachable and this test proves nothing.
				wrapped := false
				for s := range set3groupSize {
					if ctrlByteAt(set.groupCtrl[0], s) != set3Empty {
						wrapped = true
						break
					}
				}
				require.True(t, wrapped, "the overflow did not reach group 0, so the wraparound this test needs did not happen")

				var want []int
				for i := range keyCount {
					if i%removeEvery == 0 {
						require.True(t, set.Remove(i))
					} else {
						want = append(want, i)
					}
				}
				require.Positive(t, set.dead, "this arrangement was supposed to produce tombstones")
				require.Equal(t, groups, len(set.groupCtrl), "the table resized while the test was setting up")

				set.compactInPlace()
				checkCompacted(t, set, want, groups)
			})
		}
	}
}
