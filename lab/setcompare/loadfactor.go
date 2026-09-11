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

package setcompare

import (
	"unsafe"

	set3 "github.com/TomTonic/Set3"
)

// Load factor, and why the suite measures two operating points.
//
// Set3 fills to 6.67 of every 8 slots, about 83%. Go's map, measured rather
// than assumed, runs near 47%. That is not two implementations disagreeing
// about a detail — it is the same design decision traded in opposite
// directions. A lower load factor buys shorter probe sequences and pays for
// them in memory; a higher one does the reverse. Comparing the two as shipped
// therefore compares two *operating points*, and reports the sum of a layout
// difference and a space/time choice without separating them.
//
// Both questions are worth answering and they are different questions:
//
//   - "Which should I use?" is answered by comparing them as shipped. Memory
//     and speed are both outcomes and the reader weighs them. Every scenario
//     in the catalogue except the -eqload ones does this.
//   - "Is the layout faster?" needs the space/time choice held fixed. Set3 can
//     be moved to any operating point with RehashToCapacity, so the suite moves
//     it to the one the native map chose and asks again. The -eqload scenarios
//     do this.
//
// Note what equal load factor does *not* equalise. A slot costs Set3 one
// control byte plus the key; it costs the native map one control byte plus a
// padded key-and-element struct, and a struct{} element is not free because Go
// pads a struct whose last field is zero-sized. For a uint64 key that is 9
// bytes against 17. So even at an identical load factor Set3 holds about 53% of
// the bytes — the memory advantage does not come only from the load factor, and
// the -eqload rows are where you can see which part does.

// mapSlot mirrors the layout of one slot of a map[T]struct{}: the key followed
// by the zero-sized element. Its size is what the runtime allocates per slot,
// padding included, and measuring it here is what lets the suite turn a
// container's retained bytes back into a slot count.
type mapSlot[T comparable] struct {
	// These fields are never read, and that is the point: the type exists for
	// its layout, and unsafe.Sizeof is the only thing that consults it. The
	// zero-sized elem is the field that matters — Go pads a struct whose last
	// field has no size, which is why a map[K]struct{} costs the same per slot
	// as a map[K]uint64.
	key  T        //nolint:unused
	elem struct{} //nolint:unused
}

// bytesPerSlotNativeMap returns what one slot of a map[T]struct{} costs: the
// padded slot plus its share of the group's control word, which is one byte
// per slot in a group of eight.
func bytesPerSlotNativeMap[T comparable]() float64 {
	return 1 + float64(unsafe.Sizeof(mapSlot[T]{}))
}

// bytesPerSlotSet3 returns what one slot of a Set3[T] costs: the key itself
// plus one control byte, Set3 storing keys in a plain array beside a 64-bit
// control word per group of eight.
func bytesPerSlotSet3[T comparable]() float64 {
	var zero T
	return 1 + float64(unsafe.Sizeof(zero))
}

// loadFactor is the realized occupancy of a container holding elements, derived
// from the heap it actually retained.
//
// Deriving it from a measurement rather than from the implementation's own
// arithmetic is deliberate: Go's map has no documented final capacity, it
// depends on the history of table splits, and the only honest way to learn it
// is to build one and weigh it. The same function is used on both containers so
// that neither gets the benefit of a more flattering method.
//
// Returns zero when the measurement came out non-positive, which happens if the
// heap moved under it.
func loadFactor(retainedBytes, bytesPerSlot float64, elements int) float64 {
	if retainedBytes <= 0 || bytesPerSlot <= 0 || elements <= 0 {
		return 0
	}
	slots := retainedBytes / bytesPerSlot
	if slots <= 0 {
		return 0
	}
	return float64(elements) / slots
}

// measureNativeMapSlots builds a map[T]struct{} of the given size and reports
// how many slots it ended up with, along with its load factor.
//
// It exists so that a Set3 can be rehashed to the same operating point. The
// cost is one extra container build per cell, which is negligible next to the
// measurement that follows it.
func measureNativeMapSlots[T comparable](keys []T, size int) (slots float64, load float64) {
	bytes, _, _ := measureOnce(func() any {
		m := make(map[T]struct{}, size)
		for i := range size {
			m[keys[i]] = struct{}{}
		}
		return m
	}, keys)
	perSlot := bytesPerSlotNativeMap[T]()
	if bytes <= 0 || perSlot <= 0 {
		return 0, 0
	}
	return bytes / perSlot, loadFactor(bytes, perSlot, size)
}

// set3CapacityForSlots converts a wanted slot count into the capacity argument
// RehashToCapacity takes.
//
// Set3 sizes itself in groups of eight slots and admits 6.67 elements per
// group, so a capacity of c asks for ceil(c/6.67) groups, rounded up to a
// prime. Inverting that gives the capacity which produces the wanted number of
// slots; the prime rounding means the result is approximate, which is why the
// achieved load factor is measured afterwards rather than assumed.
func set3CapacityForSlots(slots float64) uint32 {
	const maxAvgGroupLoad = 6.666666666666667
	groups := slots / 8
	if groups < 1 {
		groups = 1
	}
	return uint32(groups * maxAvgGroupLoad) //nolint:gosec
}

// measureSet3Load builds a Set3 at the given capacity, fills it with size
// keys, and reports the load factor it ended up at.
//
// The container is built inside the measured region because that is the only
// way the retained heap says anything: an object allocated before the first
// reading appears in both readings and cancels. Since the capacity always
// exceeds the element count here, no rehash happens during the fill, so the
// table measured is the same one RehashToCapacity would produce.
func measureSet3Load[T comparable](keys []T, size int, capacity uint32) float64 {
	bytes, _, _ := measureOnce(func() any {
		s := set3.EmptyWithCapacity[T](capacity)
		for i := range size {
			s.Add(keys[i])
		}
		return s
	}, keys)
	return loadFactor(bytes, bytesPerSlotSet3[T](), size)
}

// set3CapacityMatchingMap returns the capacity at which a Set3 holding size
// elements occupies as many slots as a map[T]struct{} holding the same
// elements, together with the two load factors that result.
//
// This is the whole of the -eqload setup. Everything about it is measured
// rather than derived: the map's slot count comes from weighing a real map,
// and the Set3's achieved load factor comes from weighing a real Set3 at the
// capacity this returns, because Set3 rounds its group count up to a prime and
// so lands near the target rather than on it.
func set3CapacityMatchingMap[T comparable](keys []T, size int) (capacity uint32, set3Load, mapLoad float64) {
	slots, mapLoad := measureNativeMapSlots(keys, size)
	if slots <= 0 {
		return uint32(size), 0, 0 //nolint:gosec
	}
	capacity = set3CapacityForSlots(slots)
	return capacity, measureSet3Load(keys, size, capacity), mapLoad
}
