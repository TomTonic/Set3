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

import "math/bits"

// In-place compaction: reclaiming tombstoned slots without allocating.
//
// # Why this exists next to rehashToNumGroups
//
// rehashToNumGroups allocates a fresh control slice and a fresh slot slice and
// replays every element into them. That is the right thing when the table has
// to change size, and the wrong thing when it does not: a set churning at a
// constant size would allocate its whole backing store again every time it
// needed its tombstones back. Measured before this existed, a window of 262 144
// uint64 keys allocated about 4.4 MB on each of those rehashes.
//
// compactInPlace does the same job by rearranging what is already there. It
// allocates nothing, and TestCompactInPlaceAllocatesNothing keeps it that way.
//
// # The algorithm is Abseil's, not an invention
//
// This is drop_deletes_without_resize from absl::raw_hash_set, transcribed to
// this layout. The steps are:
//
//	mark every DELETED slot EMPTY and every FULL slot DELETED
//	for each slot still marked DELETED:
//	    probe from the element's home group for the first group with a free slot
//	    if that is the group the element already sits in, mark it FULL and stop
//	    if the free slot is EMPTY, move the element there and empty the source
//	    otherwise it is DELETED and holds an element that also needs a home:
//	        swap the two, mark the target FULL, and redo this slot with what
//	        came back
//
// # Why emptying the source slot cannot break anyone else's probe
//
// This is the part that is not obvious, and it rests on one invariant: a slot
// that becomes FULL in the second pass never changes again. The probe skips a
// group only when all eight of its slots are FULL, which means every element in
// it has already been placed and will stay. So when an element is placed in
// group t, the groups between its home group and t are permanently free of
// EMPTY slots, and a later lookup walking that same path cannot terminate early.
// Emptying a source slot is safe because the source group still holds a slot
// the probe would have had to stop at anyway — it was not one of the all-FULL
// groups the probe skipped.
//
// # What it deliberately does not do
//
// It does not draw a new seed. rehashToNumGroups does, because a rehash that
// grows is usually a rehash triggered by a collision pattern and a fresh seed
// breaks that pattern up. Here every element must keep the home group it
// already has — that is what makes the rearrangement local and allocation-free —
// so the seed has to stay. TestCompactInPlaceKeepsTheSeed pins it.

// compactInPlace reclaims every tombstoned slot without allocating. It is a
// no-op on a table that has none.
func (thisSet *Set3[T]) compactInPlace() {
	if thisSet.dead == 0 {
		return
	}
	groupCtrl := thisSet.groupCtrl
	groupSlot := thisSet.groupSlot
	groupCount := uint64(len(groupCtrl))

	for i := range groupCtrl {
		groupCtrl[i] = convertDeletedToEmptyAndFullToDeleted(groupCtrl[i])
	}

	var zero T
	for g := range groupCount {
		for s := range set3groupSize {
			// Not "if": the swap case puts a different element into this same
			// slot and it has to be placed too.
			for ctrlByteAt(groupCtrl[g], s) == set3Deleted {
				element := groupSlot[g][s]
				hash := thisSet.hashFunction.Hash(element)
				h2 := hash & 0x0000_0000_0000_007f
				targetGroup, targetSlot := firstNonFull(groupCtrl, groupCount, hash)

				if targetGroup == g {
					// Any free slot in this group is as good as any other, and
					// the element is already in one of them.
					groupCtrl[g] = setCTRLat(groupCtrl[g], h2, s)
					break
				}
				if ctrlByteAt(groupCtrl[targetGroup], targetSlot) == set3Empty {
					groupCtrl[targetGroup] = setCTRLat(groupCtrl[targetGroup], h2, targetSlot)
					groupSlot[targetGroup][targetSlot] = element
					groupCtrl[g] = setCTRLat(groupCtrl[g], set3Empty, s)
					// Drop the reference so that a key holding a pointer does
					// not keep its target alive, exactly as Remove does.
					groupSlot[g][s] = zero
					break
				}
				// The target holds an element that has not been placed yet.
				// Swap, settle the target, and run this slot again with the
				// element that came back.
				groupSlot[g][s] = groupSlot[targetGroup][targetSlot]
				groupSlot[targetGroup][targetSlot] = element
				groupCtrl[targetGroup] = setCTRLat(groupCtrl[targetGroup], h2, targetSlot)
			}
		}
	}

	live := thisSet.resident - thisSet.dead
	thisSet.resident = live
	thisSet.dead = 0
}

// firstNonFull returns the first group in hash's probe sequence that has a slot
// which is not FULL, together with the index of that slot.
//
// It probes exactly as Add and Contains do — the home group from getGroupIndex,
// then linearly with wraparound — because an element placed anywhere else would
// not be found again.
//
// It always terminates. The second pass only runs while at least one element is
// still unplaced, and an unplaced element occupies a slot that is not FULL.
func firstNonFull(groupCtrl []uint64, groupCount, hash uint64) (uint64, int) {
	g := getGroupIndex(hash, groupCount)
	for {
		// EMPTY is 0x80 and DELETED is 0xFE; both have the high bit set, and a
		// FULL slot holds an H2 in 0x00..0x7f, which does not.
		if free := groupCtrl[g] & set3hiBits; free != 0 {
			return g, bits.TrailingZeros64(free) >> 3
		}
		g++
		if g == groupCount {
			g = 0
		}
	}
}

// convertDeletedToEmptyAndFullToDeleted is the first pass, done a control word
// at a time.
//
// EMPTY (0x80) and DELETED (0xfe) both carry the high bit; an H2 never does. So
// the special bytes become EMPTY and the rest become DELETED:
//
//	special -> 0x80 | 0x00 = 0x80  (EMPTY)
//	full    -> 0x80 | 0x7e = 0xfe  (DELETED)
func convertDeletedToEmptyAndFullToDeleted(ctrl uint64) uint64 {
	// 0x01 in every byte that is currently FULL, 0x00 in every special one.
	full := ((ctrl & set3hiBits) >> 7) ^ set3loBits
	// Each 0x01 becomes 0x7e; 0x7e is below 0x100 so no byte carries into
	// the next.
	return set3hiBits | (full * 0x7e)
}

// ctrlByteAt returns the control byte for one slot of a group.
func ctrlByteAt(ctrl uint64, pos int) uint64 {
	return (ctrl >> (pos << 3)) & 0xFF
}
