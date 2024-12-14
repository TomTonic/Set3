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
	"iter"
	"math/bits"
	"strings"

	"github.com/dolthub/maphash"
)

const (
	set3groupSize       = 8
	set3maxAvgGroupLoad = 6.5

	set3loBits uint64 = 0x0101010101010101
	set3hiBits uint64 = 0x8080808080808080

	set3AllEmpty   uint64 = 0x8080808080808080
	set3AllDeleted uint64 = 0xFEFEFEFEFEFEFEFE
	set3Empty      uint64 = 0b0000_1000_0000
	set3Deleted    uint64 = 0b0000_1111_1110
	set3Sentinel   uint64 = 0b0000_1111_1111
)

func set3ctlrMatchH2(m uint64, h uint64) uint64 {
	return set3hasZeroByte(m ^ (set3loBits * h))
}

func set3ctlrMatchEmpty(m uint64) uint64 {
	return set3hasZeroByte(m ^ set3hiBits)
}

func set3nextMatch(b *uint64) int {
	s := bits.TrailingZeros64(*b)
	*b &= ^(1 << s) // clear bit |s|
	return s >> 3   // div by 8
}

func set3hasZeroByte(x uint64) uint64 {
	return ((x - set3loBits) & ^(x)) & set3hiBits
}

// Set3 is a hash set of type K.
type Set3[T comparable] struct {
	hashFunction maphash.Hasher[T]
	resident     uint32
	dead         uint32
	elementLimit uint32
	groupCtrl    []uint64
	groupSlot    [][set3groupSize]T
}

/*
Returns a string representation of the elements of thisSet in Roster notation (https://en.wikipedia.org/wiki/Set_(mathematics)#Roster_notation).
The order of the elements in the result is arbitrarily.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	fmt.Println(set) // will print "{2,3,1}" with the numbers in arbitrary order
*/
func (thisSet *Set3[T]) String() string {
	if thisSet == nil {
		return "{nil}"
	}
	var builder strings.Builder
	builder.WriteString("{")
	total := thisSet.Size()
	cnt := uint32(0)
	for e := range thisSet.MutableRange() {
		builder.WriteString(fmt.Sprintf("%v", e))
		if cnt < total-1 {
			builder.WriteString(",")
		}
		cnt++
	}
	builder.WriteString("}")
	return builder.String()
}

/*
Empty creates a new and empty Set3 with a reasonable default initial capacity. Choose this constructor if you have no idea on how big your set will be.
You can add as many elements to this set as you like, the backing data structure will automatically be reorganized to fit your needs.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
*/
func Empty[T comparable]() *Set3[T] {
	return EmptyWithCapacity[T](21)
}

/*
EmptyWithCapacity creates a new and empty Set3 with a given initial capacity. Choose this constructor if you have a pretty good idea on how big your set will be.
Nonetheless, you can add as many elements to this set as you like, the backing data structure will automatically be reorganized to fit your needs.

Example:

	set1 := Empty[int]() // you can put 1 mio. ints in set1. set1 will rehash itself several times while adding them
	set2 := EmptyWithCapacity[int](2_000_000) // you can put 1 mio. ints in set2. set2 does not need to rehash itself while adding them
*/
func EmptyWithCapacity[T comparable](initialCapacity uint32) *Set3[T] {
	reqNrOfGroups := calcReqNrOfGroups(initialCapacity)
	result := &Set3[T]{
		hashFunction: maphash.NewHasher[T](),
		elementLimit: uint32(float64(reqNrOfGroups) * set3maxAvgGroupLoad),
		groupCtrl:    make([]uint64, reqNrOfGroups),
		groupSlot:    make([][set3groupSize]T, reqNrOfGroups),
	}
	for i := range reqNrOfGroups {
		result.groupCtrl[i] = set3AllEmpty
	}
	return result
}

func calcReqNrOfGroups(reqCapa uint32) uint32 {
	reqNrOfGroups := uint32((float64(reqCapa) + set3maxAvgGroupLoad - 1) / set3maxAvgGroupLoad)
	if reqNrOfGroups == 0 {
		reqNrOfGroups = 1
	}
	return reqNrOfGroups
}

/*
From is a convenience constructor to directly create a Set3 from given arguments. It creates a new Set3 and adds all (unique) elements to this set.

If the arguments contain duplicates, the duplicates are omitted. If no arguments are provided, an empty Set3 is returned.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := From(1, 2, 3) // set1 and set2 are equal
*/
func From[T comparable](args ...T) *Set3[T] {
	if args == nil {
		return Empty[T]()
	}
	result := EmptyWithCapacity[T](uint32(len(args) * 7 / 5)) //nolint:gosec
	for _, e := range args {
		result.Add(e)
	}
	return result
}

/*
FromArray is a convenience constructor to directly create a Set3 from given values. It creates a new Set3 and adds all (unique) elements to this set.

If the array contains duplicates, the duplicates are omitted. If data is nil, an empty Set3 is returned.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := FromArray([]int{1, 2, 3}) // set1 and set2 are equal
*/
func FromArray[T comparable](data []T) *Set3[T] {
	if data == nil {
		return Empty[T]()
	}
	result := EmptyWithCapacity[T](uint32(len(data) * 7 / 5)) //nolint:gosec
	for _, e := range data {
		result.Add(e)
	}
	return result
}

/*
Clone creates an exact clone of thisSet. You can manipulate both clones independently.

Cloning is 'cheap' in comparison with creating a new set and adding all elements from this set, as only the backing data structures are copied (no rehashing is applied).

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := set1.Clone() // set2 will be an exact but independent clone of set1
*/
func (thisSet *Set3[T]) Clone() *Set3[T] {
	result := &Set3[T]{
		hashFunction: thisSet.hashFunction,
		elementLimit: thisSet.elementLimit,
		resident:     thisSet.resident,
		dead:         thisSet.dead,
		groupCtrl:    make([]uint64, len(thisSet.groupCtrl)),
		groupSlot:    make([][set3groupSize]T, len(thisSet.groupSlot)),
	}
	copy(result.groupCtrl, thisSet.groupCtrl)
	copy(result.groupSlot, thisSet.groupSlot)
	return result
}

/*
Contains returns true if the element is contained in thisSet.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	b1 := set.Contains(2) // b1 will be true
	b2 := set.Contains(4) // b2 will be false
*/
func (thisSet *Set3[T]) Contains(element T) bool {
	hash := thisSet.hashFunction.Hash(element)
	H2 := (hash & 0x0000_0000_0000_007f)
	groupCount := uint64(len(thisSet.groupCtrl))
	currentGroupIndex := getGroupIndex(hash, groupCount)
	for {
		ctrl := thisSet.groupCtrl[currentGroupIndex]
		H2matches := set3ctlrMatchH2(ctrl, H2)
		if H2matches != 0 {
			slot := &(thisSet.groupSlot[currentGroupIndex])
			for H2matches != 0 {
				s := set3nextMatch(&H2matches)
				if element == slot[s] {
					return true
				}
			}
		}
		// |key| is not in group |g|,
		// stop probing if we see an empty slot
		emptyMatches := set3ctlrMatchEmpty(ctrl)
		if emptyMatches != 0 {
			// there is an empty slot - the element, if it had been added, hat either
			// been found until now or it had been added in the next empty spot -
			// well, this is the next empty spot...
			return false
		}
		currentGroupIndex++ // carousel through all groups
		if currentGroupIndex >= groupCount {
			currentGroupIndex = 0
		}
	}
}

func getGroupIndex(hash, groupCount uint64) uint64 {
	// no. 1: Original approach
	// H1 := (hash & 0xffff_ffff_ffff_ff80) >> 7
	// no. 2: adapt constant to use Lemire's fast alternative to the modulo reduction (see https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/)
	// H1 := (hash & 0x0000_007f_ffff_ff80) >> 7
	// no. 3: make sure not to loose entropy from the top 25 bits (in case of weak hash function)
	// H1 := (((hash >> 25) ^ hash) & 0x0000_007f_ffff_ff80) >> 7

	// Original approach (no. 1)
	// return H1 % groupCount

	// Lemire's fast alternative to the modulo reduction (no. 2)
	// return (H1 * groupCount) >> 32

	// no. 3b: same as no. 3
	a := hash >> 32
	b := hash >> 7
	c := a ^ b
	d := c & 0x0000_0000_ffff_ffff
	e := d * groupCount
	f := e >> 32
	return f
}

/*
Returns true if thisSet contains all elements from thatSet.

If thatSet is empty, ContainsAll returns true. If thatSet is nil, ContainsAll returns true.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	Empty := Empty[int]()
	b := set.ContainsAll(Empty) // b will be true
*/
func (thisSet *Set3[T]) ContainsAll(thatSet *Set3[T]) bool {
	if thisSet == thatSet {
		return true
	}
	if thatSet == nil {
		// nil is interpreted as empty set
		return true
	}
	if thisSet.Size() < thatSet.Size() {
		return false
	}
	for e := range thatSet.MutableRange() {
		if !thisSet.Contains(e) {
			return false
		}
	}
	return true
}

/*
Returns true if thisSet contains all of the given argument values.

If the number of arguments is zero, ContainsAllOf returns true.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	b := set.ContainsAllOf(2,3,4) // b will be false
*/
func (thisSet *Set3[T]) ContainsAllOf(args ...T) bool {
	if args == nil {
		// nil is interpreted as empty set
		return true
	}
	for _, e := range args {
		if !thisSet.Contains(e) {
			return false
		}
	}
	return true
}

/*
Returns true if thisSet contains all elements from the given data array.

If the length of data is zero, ContainsAllFromArray returns true. If data is nil, ContainsAllFromArray returns true.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	b := set.ContainsAllFromArray([]int{2,3,4}) // b will be false
*/
func (thisSet *Set3[T]) ContainsAllFromArray(data []T) bool {
	if data == nil {
		// nil is interpreted as empty set
		return true
	}
	for _, e := range data {
		if !thisSet.Contains(e) {
			return false
		}
	}
	return true
}

/*
Returns true if thisSet and thatSet have the same size and contain the same elements.

If thatSet is nil, Equals returns true if and only if thisSet is empty.

Example:

	set1 := Empty[int]()
	set2 := Empty[int]()
	b1 := set1.Equals(set2) // b1 will be true
	set1.Add(7)
	set2.Add(31)
	b2 := set1.Equals(set2) // b2 will be false
*/
func (thisSet *Set3[T]) Equals(thatSet *Set3[T]) bool {
	if thisSet == thatSet {
		return true
	}
	if thatSet == nil {
		// nil is interpreted as empty set
		return thisSet.Size() == 0
	}
	if thisSet.Size() != thatSet.Size() {
		return false
	}
	for elem := range thatSet.MutableRange() {
		if !thisSet.Contains(elem) {
			return false
		}
	}
	return true
}

/*
Iterates over all elements in thisSet.

Caution: If thisSet is changed during the iteration, the result is unpredictable. So if you want to add or remove elements to or from thisSet during the itration, choose [ImmutableRange].

Example:

	for elem := range set.MutableRange() {
		// do something with elem...
	}
*/
func (thisSet *Set3[T]) MutableRange() iter.Seq[T] {
	return func(yield func(T) bool) {
		for i, ctrl := range thisSet.groupCtrl {
			if ctrl&set3hiBits != set3hiBits { // not all empty or deleted
				slot := &(thisSet.groupSlot[i])
				for i := 0; i < set3groupSize; i++ {
					if isAnElementAt(ctrl, i) {
						if !yield(slot[i]) {
							return
						}
					}
				}
			}
		}
	}
}

/*
Iterates over all elements in thisSet.

Makes an internal copy of the stored elements first, so you can add or remove elements to or from thisSet during the itration, for example. To avoid this extra copy, e.g., for performance reasons, choose [MutableRange].

Example:

	for elem := range set.ImmutableRange() {
		// do something with elem...
	}
*/
func (thisSet *Set3[T]) ImmutableRange() iter.Seq[T] {
	return func(yield func(T) bool) {
		localCtrl := make([]uint64, len(thisSet.groupCtrl))
		localSlot := make([][set3groupSize]T, len(thisSet.groupSlot))
		copy(localCtrl, thisSet.groupCtrl)
		copy(localSlot, thisSet.groupSlot)
		for i, ctrl := range localCtrl {
			if ctrl&set3hiBits != set3hiBits { // not all empty or deleted
				slot := &(localSlot[i])
				for i := 0; i < set3groupSize; i++ {
					if isAnElementAt(ctrl, i) {
						if !yield(slot[i]) {
							return
						}
					}
				}
			}
		}
	}
}

/*
ToArray allocates an array of type T and adds all elements of thisSet to it. The order of the elements in the resulting array is arbitrary.

Example:

	set := Empty[int]()
	set.Add(7)
	set.Add(31)
	intArray := set.ToArray() // will be an []int of length 2 containing 7 and 31 in arbitrary order
*/
func (thisSet *Set3[T]) ToArray() []T {
	result := make([]T, thisSet.Size())
	i := 0
	for e := range thisSet.MutableRange() {
		result[i] = e
		i++
	}
	return result
}

/*
Inserts the element into thisSet if it is not yet in thisSet.

Example:

	set := Empty[int]()
	set.Add(7)
*/
func (thisSet *Set3[T]) Add(element T) {
	if thisSet.resident >= thisSet.elementLimit {
		thisSet.rehashToNumGroups(thisSet.calcNextGroupCount())
	}
	hash := thisSet.hashFunction.Hash(element)
	H2 := (hash & 0x0000_0000_0000_007f)
	groupCount := uint64(len(thisSet.groupCtrl))
	currentGroupIndex := getGroupIndex(hash, groupCount)
	for {
		ctrl := thisSet.groupCtrl[currentGroupIndex]
		H2matches := set3ctlrMatchH2(ctrl, H2)
		if H2matches != 0 {
			slot := &(thisSet.groupSlot[currentGroupIndex])
			for H2matches != 0 {
				s := set3nextMatch(&H2matches)
				if element == slot[s] {
					// found - already in Set3, just return
					return
				}
			}
		}

		// element is not in group,
		// stop probing if we see an empty slot
		emptyMatches := set3ctlrMatchEmpty(ctrl)

		if emptyMatches != 0 {
			// empty spot -> element can't be in Set3 (see Contains) -> insert
			s := set3nextMatch(&emptyMatches)
			thisSet.groupCtrl[currentGroupIndex] = setCTRLat(ctrl, H2, s)
			thisSet.groupSlot[currentGroupIndex][s] = element
			thisSet.resident++
			return

		}
		currentGroupIndex++ // carousel through all groups
		if currentGroupIndex >= groupCount {
			currentGroupIndex = 0
		}
	}
}

/*
Inserts all elements from thatSet that are not yet in thisSet into thisSet.

If thatSet is nil, nothing is added to thisSet.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set2 := Empty[int]()
	set2.Add(2)
	set2.Add(3)
	set1.AddAll(set2) // set1 will now contain 1, 2, 3
*/
func (thisSet *Set3[T]) AddAll(thatSet *Set3[T]) {
	if thatSet == nil {
		return
	}
	for e := range thatSet.MutableRange() {
		thisSet.Add(e)
	}
}

/*
Inserts all parameter values that are not yet in thisSet into thisSet.

If the number of parameters is zero, nothing is added to thisSet.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.AddAllOf(2,3) // set will now contain 1, 2, 3
*/
func (thisSet *Set3[T]) AddAllOf(args ...T) {
	if args == nil {
		return
	}
	for _, e := range args {
		thisSet.Add(e)
	}
}

/*
Inserts all elements from the given data array that are not yet in thisSet into thisSet.

If data is nil, nothing is added to thisSet.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.AddAllFromArray([]int{2,3}]) // set will now contain 1, 2, 3
*/
func (thisSet *Set3[T]) AddAllFromArray(data []T) {
	if data == nil {
		return
	}
	for _, e := range data {
		thisSet.Add(e)
	}
}

/*
Creates a new Set3 as a mathematical union of the elements from thisSet and thatSet.

If thatSet is nil, Unite returns a clone of thisSet.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set2 := Empty[int]()
	set2.Add(2)
	set2.Add(3)

	u := set1.Unite(set2) // set1 and set2 remain unchanged, u will contain 1, 2, 3
*/
func (thisSet *Set3[T]) Unite(thatSet *Set3[T]) *Set3[T] {
	if thatSet == nil {
		return thisSet.Clone()
	}
	potentialSize := thisSet.Size() + thatSet.Size()
	result := EmptyWithCapacity[T](potentialSize)
	for e := range thisSet.MutableRange() {
		result.Add(e)
	}
	for e := range thatSet.MutableRange() {
		result.Add(e)
	}
	return result
}

func setCTRLat(ctrl, val uint64, pos int) uint64 {
	shift := pos << 3                // *8
	ctrl &= ^(uint64(0xFF) << shift) // clear byte
	ctrl |= val << shift             // set byte to given value
	return ctrl
}

func isAnElementAt(ctrl uint64, pos int) bool {
	shift := pos << 3               // *8
	ctrl &= (uint64(0x80) << shift) // clear all other bits
	// if a bit is set, the according byte represented either set3Empty or set3Deleted
	// -> if ctlr is 0 now, the according position stores a value
	return ctrl == 0
}

/*
Removes the given element from thisSet if it is in thisSet, returns whether or not the element was in thisSet.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Remove(2)	// nothing happens to set
	set.Remove(1)	// set will be empty
	set.Remove(0)	// set will still be empty
*/
func (thisSet *Set3[T]) Remove(element T) bool {
	hash := thisSet.hashFunction.Hash(element)
	H2 := (hash & 0x0000_0000_0000_007f)
	groupCount := uint64(len(thisSet.groupCtrl))
	currentGroupIndex := getGroupIndex(hash, groupCount)
	for {
		ctrl := thisSet.groupCtrl[currentGroupIndex]
		H2matches := set3ctlrMatchH2(ctrl, H2)
		if H2matches != 0 {
			slot := &(thisSet.groupSlot[currentGroupIndex])
			for H2matches != 0 {
				s := set3nextMatch(&H2matches)
				if element == slot[s] {
					// found - already in Set3, just return
					// optimization: if |m.ctrl[g]| contains any empty
					// metadata bytes, we can physically delete |element|
					// rather than placing a tombstone.
					// The observation is that any probes into group |g|
					// would already be terminated by the existing empty
					// slot, and therefore reclaiming slot |s| will not
					// cause premature termination of probes into |g|.
					if set3ctlrMatchEmpty(ctrl) != 0 {
						thisSet.groupCtrl[currentGroupIndex] = setCTRLat(ctrl, set3Empty, s)
						thisSet.resident--
					} else {
						thisSet.groupCtrl[currentGroupIndex] = setCTRLat(ctrl, set3Deleted, s)
						thisSet.dead++
						/*
							// unfortunately, this is an invalid optimization, as the algorithm might stop searching for elements to early.
							// if they spilled over in the next group, we unfortunately need all the tumbstones...
							if group.ctrl == set3AllDeleted {
								group.ctrl = set3AllEmpty
								thisSet.dead -= set3groupSize
								thisSet.resident -= set3groupSize
							}
						*/
					}
					var k T
					thisSet.groupSlot[currentGroupIndex][s] = k
					return true
				}
			}
		}

		// |element| is not in group |g|,
		// stop probing if we see an empty slot
		emptyMatches := set3ctlrMatchEmpty(ctrl)
		if emptyMatches != 0 {
			// |element| absent
			return false
		}
		currentGroupIndex++ // linear probing
		if currentGroupIndex >= groupCount {
			currentGroupIndex = 0
		}
	}
}

/*
Removes all elements from thisSet that are in thatSet.

If thatSet is nil, nothing happens.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := Empty[int]()
	set2.Add(3)
	set2.Add(4)
	set1.RemoveAll(set2) // set1 will now contain 1, 2
*/
func (thisSet *Set3[T]) RemoveAll(thatSet *Set3[T]) {
	if thatSet == nil {
		return
	}
	for e := range thatSet.MutableRange() {
		thisSet.Remove(e)
	}
}

/*
Removes all elements from thisSet that are passed as arguments.

If no arguments are passed, nothing happens.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	set.RemoveAllOf(3,4) // set will now contain 1, 2
*/
func (thisSet *Set3[T]) RemoveAllOf(args ...T) {
	if args == nil {
		return
	}
	for _, e := range args {
		thisSet.Remove(e)
	}
}

/*
Removes all elements from thisSet that are in the data array.

If data is nil, nothing happens.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	set.RemoveAllFromArray([]int{3,4}]) // set will now contain 1, 2
*/
func (thisSet *Set3[T]) RemoveAllFromArray(data []T) {
	if data == nil {
		return
	}
	for _, e := range data {
		thisSet.Remove(e)
	}
}

/*
Creates a new Set3 as a mathematical difference between thisSet and thatSet. The result is a new Set3 that contains elements that are in thisSet but not in thatSet.

If thatSet is nil, Subtract returns a clone of thisSet.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := Empty[int]()
	set2.Add(3)
	set2.Add(4)
	d := set1.Subtract(set2) // set1 and set2 are not altered, d will contain 1, 2
*/
func (thisSet *Set3[T]) Subtract(thatSet *Set3[T]) *Set3[T] {
	if thatSet == nil {
		return thisSet.Clone()
	}
	potentialSize := thisSet.Size()
	result := EmptyWithCapacity[T](potentialSize)
	for e := range thisSet.MutableRange() {
		if !thatSet.Contains(e) {
			result.Add(e)
		}
	}
	return result
}

/*
Clear removes all elements from thisSet.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Clear()  // set will be empty, Count() will return 0
*/
func (thisSet *Set3[T]) Clear() {
	// var k T
	for grpidx := range len(thisSet.groupCtrl) {
		thisSet.groupCtrl[grpidx] = set3AllEmpty
		/*
			for j := range set3groupSize {
				thisSet.groupSlot[grpidx][j] = k
			}
		*/
	}
	thisSet.resident, thisSet.dead = 0, 0
}

/*
Creates a new Set3 as a mathematical intersection between this and that. The result is a new Set3 that contains elements that are in both sets.

If thatSet is nil, Intersect returns an empty Set3.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := Empty[int]()
	set2.Add(3)
	set2.Add(4)
	intersect := set1.Intersect(set2) // set1 and set2 are not altered, intersect will contain 3
*/
func (thisSet *Set3[T]) Intersect(thatSet *Set3[T]) *Set3[T] {
	if thatSet == nil {
		return Empty[T]()
	}

	var smallerSet *Set3[T]
	var biggerSet *Set3[T]

	if thisSet.Size() < thatSet.Size() {
		smallerSet = thisSet
		biggerSet = thatSet
	} else {
		smallerSet = thatSet
		biggerSet = thisSet
	}

	potentialSize := smallerSet.Size()
	result := EmptyWithCapacity[T](potentialSize)
	for e := range smallerSet.ImmutableRange() {
		if biggerSet.Contains(e) {
			result.Add(e)
		}
	}
	return result
}

/*
Creates a new Set3 as a mathematical intersection between thisSet and the elements of the data array. The result is a new Set3.

If data is nil, IntersectWithArray returns an empty Set3.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	intersect := set.IntersectWithArray([]int{3,4}]) // set1 and set2 are not altered, intersect will contain 3
*/
func (thisSet *Set3[T]) IntersectWithArray(data []T) *Set3[T] {
	if data == nil {
		return Empty[T]()
	}

	var potentialSize uint32

	if thisSet.Size() < uint32(len(data)) { //nolint:gosec
		potentialSize = thisSet.Size()
	} else {
		potentialSize = uint32(len(data)) //nolint:gosec
	}

	result := EmptyWithCapacity[T](potentialSize)
	for _, e := range data {
		if thisSet.Contains(e) {
			result.Add(e)
		}
	}
	return result
}

/*
Checks if thisSet contains any element that is also present in thatSet. This function also provides a quick way to check if two Set3 are disjoint (i.e. !ContainsAny).

Returns false if thatSet is nil.

Example:

	set1 := Empty[int]()
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	set2 := Empty[int]()
	set2.Add(0)
	set2.Add(1)
	b := set1.ContainsAny(set2) // b will be true
*/
func (thisSet *Set3[T]) ContainsAny(thatSet *Set3[T]) bool {
	if thatSet == nil {
		return false
	}

	var smallerSet *Set3[T]
	var biggerSet *Set3[T]

	if thisSet.Size() < thatSet.Size() {
		smallerSet = thisSet
		biggerSet = thatSet
	} else {
		smallerSet = thatSet
		biggerSet = thisSet
	}

	for e := range smallerSet.ImmutableRange() {
		if biggerSet.Contains(e) {
			return true
		}
	}
	return false
}

/*
Checks if thisSet contains any of the given argument values.

Returns false if the number of arguments is zero.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	b := set1.ContainsAnyOf(4, 5, 6) // b will be false
*/
func (thisSet *Set3[T]) ContainsAnyOf(args ...T) bool {
	if args == nil {
		return false
	}
	for _, d := range args {
		if thisSet.Contains(d) {
			return true
		}
	}
	return false
}

/*
Checks if thisSet contains any element fromthe given data array.

Returns false if data is nil.

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	b := set1.ContainsAnyFromArray([]int{4, 5, 6}) // b will be false
*/
func (thisSet *Set3[T]) ContainsAnyFromArray(data []T) bool {
	if data == nil {
		return false
	}
	for _, d := range data {
		if thisSet.Contains(d) {
			return true
		}
	}
	return false
}

/*
Size returns the number of elements in thisSet.

Example:

	set := Empty[int]()
	set.Add(7)
	set.Add(8)
	set.Add(9)
	c := set.Size()   // c will be 3
*/
func (thisSet *Set3[T]) Size() uint32 {
	return thisSet.resident - thisSet.dead
}

func (thisSet *Set3[T]) calcNextGroupCount() uint32 {
	n := len(thisSet.groupCtrl) * 2
	if thisSet.dead >= (thisSet.resident / 2) {
		n = len(thisSet.groupCtrl)
	}
	return uint32(n) //nolint:gosec
}

/*
Rorganizes the backend of thisSet for optimal space efficiency: This call rehashes thisSet to a size matching its current element count.

Example:

	set := Empty[int](1_000_000) // allocates a big hashset
	set.Add(1)
	set.Add(2)
	set.Add(3)
	set.Rehash() // saves memory consumed by set
*/
func (thisSet *Set3[T]) Rehash() {
	numGroups := calcReqNrOfGroups(thisSet.Size()) //nolint:gosec
	thisSet.rehashToNumGroups(numGroups)
}

/*
Rorganizes the backend of thisSet: RehashToCapacity redistributs the elements of thisSet onto a new hashset in its backend, e.g., to ensure faster element access.

If newSize is smaller than the current number of elements in thisSet, this function does nothing. If newSize is equal to the current number of elements in thisSet, this function does the same as [Rehash].

Example:

	set := Empty[int]()
	set.Add(1)
	set.Add(2)
	set.Add(3)
	set.RehashToCapacity(1000) // ensures that you can add at least 997 more elements to set without rehashing
*/
func (thisSet *Set3[T]) RehashToCapacity(newCapacity uint32) {
	if newCapacity < thisSet.Size() {
		return
	}
	newNumGroups := calcReqNrOfGroups(newCapacity)
	thisSet.rehashToNumGroups(newNumGroups)
}

func (thisSet *Set3[T]) rehashToNumGroups(newNumGroups uint32) {
	oldNumGroups := len(thisSet.groupCtrl)
	oldGroupCtrl := make([]uint64, oldNumGroups)
	oldGroupSlot := make([][set3groupSize]T, oldNumGroups)
	copy(oldGroupCtrl, thisSet.groupCtrl)
	copy(oldGroupSlot, thisSet.groupSlot)

	thisSet.hashFunction = maphash.NewSeed(thisSet.hashFunction)
	thisSet.elementLimit = uint32(float64(newNumGroups) * set3maxAvgGroupLoad)
	thisSet.resident, thisSet.dead = 0, 0
	thisSet.groupCtrl = make([]uint64, newNumGroups)
	thisSet.groupSlot = make([][set3groupSize]T, newNumGroups)
	for i := range newNumGroups {
		thisSet.groupCtrl[i] = set3AllEmpty
	}
	grpCnt := uint64(newNumGroups)
	for oldGroupIndex := 0; oldGroupIndex < oldNumGroups; oldGroupIndex++ {
		ctrl := oldGroupCtrl[oldGroupIndex]
		if ctrl&set3hiBits != set3hiBits { // not all positions empty or deleted
			for s := range set3groupSize {
				if isAnElementAt(ctrl, s) {
					elementToAdd := oldGroupSlot[oldGroupIndex][s]

					// inlined and optimized Add implementation instead of Set3.Add(oldGroup.slot[s])
					hash := thisSet.hashFunction.Hash(elementToAdd)
					H2 := (hash & 0x0000_0000_0000_007f)
					grpIdx := getGroupIndex(hash, uint64(newNumGroups))
					stillSearchingSpace := true
					for stillSearchingSpace {
						// optimization: we know it cannot be in thisSet yet so skip
						// searching for the hashcode and start searching for an empty slot
						// immediately
						matches := set3ctlrMatchEmpty(thisSet.groupCtrl[grpIdx])

						if matches != 0 {
							// empty spot -> element can't be in Set3 (see Contains) -> insert
							s := set3nextMatch(&matches)
							thisSet.groupCtrl[grpIdx] = setCTRLat(thisSet.groupCtrl[grpIdx], H2, s)
							thisSet.groupSlot[grpIdx][s] = elementToAdd
							thisSet.resident++
							stillSearchingSpace = false
						}
						grpIdx++ // carousel through all groups
						if grpIdx >= grpCnt {
							grpIdx = 0
						}
					}
				}
			}
		}
	}
}
