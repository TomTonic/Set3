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
	"fmt"

	set3 "github.com/TomTonic/Set3"
)

// This file holds the whole-pass workloads: one batch operation builds a fresh
// container, runs a complete pass over it, and throws it away. They allocate,
// so they run with a collection between batches and the collector left on; see
// Workload.Allocating.
//
// The unit for all of them is per logical item, not per pass, which is what
// Workload.ItemsPerOp converts.

// buildFillWorkload measures filling a container with a known set of distinct
// keys, either into a container created at the right capacity (presized) or
// into an empty one that has to grow (presized == false).
//
// The two variants answer different questions. Presized is the steady cost of
// insertion with no rehashing anywhere in it, which is the number to quote for
// a bulk load. Growing includes the whole rehash cascade from the default
// capacity of 21 up to the final size, amortised over the elements — that is
// the number the README's fill chart shows, and the one most programs actually
// pay because most programs do not know the size in advance.
func buildFillWorkload[T comparable](w *Workload, mk keyMaker[T], size int, presized bool) *Workload {
	keys := buildKeys(mk, memberDomain, size)
	w.ItemsPerOp = float64(size)
	w.Allocating = true

	capacity := uint32(0)
	if presized {
		capacity = uint32(size) //nolint:gosec
	}

	w.Set3Batch = func(n uint64) {
		var acc uint64
		for range n {
			s := set3.EmptyWithCapacity[T](capacity)
			for _, k := range keys {
				s.Add(k)
			}
			acc += uint64(s.Size())
		}
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		var acc uint64
		for range n {
			m := make(map[T]struct{}, capacity)
			for _, k := range keys {
				m[k] = struct{}{}
			}
			acc += uint64(len(m))
		}
		sink += acc
	}
	w.Verify = func() error {
		s := set3.EmptyWithCapacity[T](capacity)
		m := make(map[T]struct{}, capacity)
		for _, k := range keys {
			s.Add(k)
			m[k] = struct{}{}
		}
		return sameSize("fill", int(s.Size()), len(m), size)
	}
	w.Release = func() { keys = nil }
	return w
}

// buildDedupWorkload measures deduplicating a skewed stream, in the shape a
// deduplicator that has to emit the new keys must use: ask first, insert only
// on a miss.
//
// The stream is twice the key space long and heavily skewed, so a small head
// of keys recurs constantly and stays cache-resident while the tail is seen
// once. That skew is the whole point: on a uniform stream every probe is a
// cache miss and the comparison degenerates into a memory latency measurement
// that says nothing about either implementation.
func buildDedupWorkload[T comparable](w *Workload, mk keyMaker[T], size int) *Workload {
	members := buildKeys(mk, memberDomain, size)
	stream := zipfStream(members, 2*size)
	w.ItemsPerOp = float64(len(stream))
	w.Allocating = true

	w.Set3Batch = func(n uint64) {
		var acc uint64
		for range n {
			s := set3.Empty[T]()
			var fresh uint64
			for _, e := range stream {
				if !s.Contains(e) {
					s.Add(e)
					fresh++
				}
			}
			acc += fresh
		}
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		var acc uint64
		for range n {
			m := make(map[T]struct{})
			var fresh uint64
			for _, e := range stream {
				if _, ok := m[e]; !ok {
					m[e] = struct{}{}
					fresh++
				}
			}
			acc += fresh
		}
		sink += acc
	}
	w.Verify = func() error {
		s := set3.Empty[T]()
		m := make(map[T]struct{})
		var freshSet, freshMap int
		for _, e := range stream {
			if !s.Contains(e) {
				s.Add(e)
				freshSet++
			}
			if _, ok := m[e]; !ok {
				m[e] = struct{}{}
				freshMap++
			}
		}
		if freshSet != freshMap {
			return fmt.Errorf("dedup: Set3 emitted %d new keys, map emitted %d", freshSet, freshMap)
		}
		return sameSize("dedup", int(s.Size()), len(m), freshSet)
	}
	w.Release = func() { members, stream = nil, nil }
	return w
}

// buildIntersectWorkload measures intersecting two equally sized sets that
// overlap in a fifth of their elements, which is roughly what an inverted-index
// AND query looks like when neither term is rare.
//
// This is the one workload that compares an API against hand-written code
// rather than an operation against an operation. Set3 offers Intersect; a map
// user writes the loop, and the loop written here is the one they would write:
// iterate the smaller side, probe the larger, collect into a result sized from
// the smaller side. That is also exactly what Set3.Intersect does internally,
// with one difference that the measurement will show — Set3 iterates through
// ImmutableRange, which copies the smaller table before walking it.
func buildIntersectWorkload[T comparable](w *Workload, mk keyMaker[T], size int) *Workload {
	left := buildKeys(mk, memberDomain, size)
	right := make([]T, size)
	disjoint := buildKeys(mk, secondDomain, size)
	for i := range right {
		if i%5 == 0 { // 20% overlap
			right[i] = left[i]
		} else {
			right[i] = disjoint[i]
		}
	}

	setA := set3.FromArray(left)
	setB := set3.FromArray(right)
	mapA := make(map[T]struct{}, size)
	mapB := make(map[T]struct{}, size)
	for _, k := range left {
		mapA[k] = struct{}{}
	}
	for _, k := range right {
		mapB[k] = struct{}{}
	}

	w.ItemsPerOp = float64(size)
	w.Allocating = true

	w.Set3Batch = func(n uint64) {
		var acc uint64
		for range n {
			acc += uint64(setA.Intersect(setB).Size())
		}
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		var acc uint64
		for range n {
			small, big := mapA, mapB
			if len(mapB) <= len(mapA) {
				small, big = mapB, mapA
			}
			result := make(map[T]struct{}, len(small))
			for k := range small {
				if _, ok := big[k]; ok {
					result[k] = struct{}{}
				}
			}
			acc += uint64(len(result))
		}
		sink += acc
	}
	w.Verify = func() error {
		got := int(setA.Intersect(setB).Size())
		small, big := mapA, mapB
		if len(mapB) <= len(mapA) {
			small, big = mapB, mapA
		}
		want := 0
		for k := range small {
			if _, ok := big[k]; ok {
				want++
			}
		}
		if got != want {
			return fmt.Errorf("intersect: Set3 found %d common elements, the map loop found %d", got, want)
		}
		if want == 0 {
			return fmt.Errorf("intersect: the two operands do not overlap at all, so the workload measures nothing")
		}
		return nil
	}
	w.Release = func() {
		left, right, disjoint = nil, nil, nil
		setA, setB, mapA, mapB = nil, nil, nil, nil
	}
	return w
}

// buildIterateWorkload measures walking every element of a populated container
// once, as a flush, an export or a checkpoint does.
//
// Set3 is compared through MutableRange, the iterator that walks the table in
// place. Its sibling ImmutableRange copies the whole table before yielding
// anything, which is a different operation and would not be a fair comparison
// against ranging over a map — that cost shows up in the intersect workload
// instead, where the library uses it internally.
//
// Both sides fold each element through the same folder, so that the loop body
// is not empty and the comparison is between the two traversals rather than
// between two ways of doing nothing.
func buildIterateWorkload[T comparable](w *Workload, mk keyMaker[T], fold folder[T], size int) *Workload {
	keys := buildKeys(mk, memberDomain, size)
	s := set3.FromArray(keys)
	m := make(map[T]struct{}, size)
	for _, k := range keys {
		m[k] = struct{}{}
	}

	w.ItemsPerOp = float64(size)
	// Not allocating, despite appearances. MutableRange returns a closure per
	// pass, which looks like an allocation and is not: the compiler keeps it on
	// the stack because range-over-func consumes it immediately, and the suite
	// measures zero bytes per operation for this workload. The distinction is
	// not cosmetic — see collectOptionsFor for what declaring it allocating
	// costs, and auditAllocationFlag for what catches the mistake.
	w.Allocating = false

	// The accumulator lives in a one-element slice rather than in a local, on
	// both sides, and that is not a style choice.
	//
	// A local captured by the range-over-func closure is normally kept in a
	// register: the compiler proves the closure does not escape because
	// range-over-func consumes it immediately. Under a profile that made the
	// batch an indirect call, it stopped proving that — the closure escaped,
	// the local moved to the heap with it, and every accumulation became a
	// heap write. The map side ranges the language construct and has no
	// closure, so it was unaffected, and the measured difference moved by up to
	// 24 percentage points for a reason that lived entirely in this file.
	//
	// A slice element is already heap memory in every build, so both sides do
	// the same store to the same cache line whatever the compiler decides. It
	// costs one L1 write per element on each side and it does not move.
	accSet := make([]uint64, 1)
	accMap := make([]uint64, 1)
	w.Set3Batch = func(n uint64) {
		for range n {
			for e := range s.MutableRange() {
				accSet[0] ^= fold(e)
			}
		}
		sink += accSet[0]
	}
	w.MapBatch = func(n uint64) {
		for range n {
			for e := range m {
				accMap[0] ^= fold(e)
			}
		}
		sink += accMap[0]
	}
	w.Verify = func() error {
		var fromSet, fromMap uint64
		var seenSet, seenMap int
		for e := range s.MutableRange() {
			fromSet ^= fold(e)
			seenSet++
		}
		for e := range m {
			fromMap ^= fold(e)
			seenMap++
		}
		if fromSet != fromMap || seenSet != seenMap {
			return fmt.Errorf("iterate: Set3 yielded %d elements folding to %#x, the map yielded %d folding to %#x",
				seenSet, fromSet, seenMap, fromMap)
		}
		return sameSize("iterate", seenSet, seenMap, size)
	}
	w.Release = func() {
		keys = nil
		s = nil
		m = nil
	}
	return w
}

// buildGraphWorkload measures a breadth-first traversal of a random 4-regular
// graph that uses the container as its visited set, with a fresh set per
// traversal.
//
// It is the most realistic workload here and the least isolated one, in the
// same breath. Realistic, because a visited set is the single most common use
// of a set in ordinary code, and because the keys are dense small integers —
// node ids, not the well-spread random values every other workload uses, which
// is a genuinely harder case for a hash function. Least isolated, because the
// queue, the successor lookups and the loop are shared overhead that both
// candidates carry equally: they cannot flip the sign of the result but they
// do shrink it. Read this row as "how much faster does the traversal get",
// not "how much faster is the set".
//
// Node ids are uint64, so this scenario is registered for that key type only.
func buildGraphWorkload(w *Workload, size int) *Workload {
	const degree = 4
	rng := newSplitmix(0x5eed_5ca1_ab1e_0003)
	succ := make([]uint32, size*degree)
	for i := range succ {
		succ[i] = rng.uint32n(uint32(size)) //nolint:gosec
	}
	queue := make([]uint32, 0, size)

	// One dry run over a bitmap establishes how many edges a traversal
	// examines, which is what the per-edge cost is divided by. The graph is
	// fixed, so this is deterministic and measuring it once is enough.
	seen := make([]bool, size)
	edges := traverse(succ, queue, degree, func(u uint64) bool {
		if seen[u] {
			return false
		}
		seen[u] = true
		return true
	})
	seen = nil

	w.ItemsPerOp = float64(edges)
	w.Allocating = true

	w.Set3Batch = func(n uint64) {
		var acc uint64
		for range n {
			visited := set3.EmptyWithCapacity[uint64](uint32(size)) //nolint:gosec
			acc += uint64(traverse(succ, queue, degree, func(u uint64) bool {
				if visited.Contains(u) {
					return false
				}
				visited.Add(u)
				return true
			}))
		}
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		var acc uint64
		for range n {
			visited := make(map[uint64]struct{}, size)
			acc += uint64(traverse(succ, queue, degree, func(u uint64) bool {
				if _, ok := visited[u]; ok {
					return false
				}
				visited[u] = struct{}{}
				return true
			}))
		}
		sink += acc
	}
	w.Verify = func() error {
		visited := set3.EmptyWithCapacity[uint64](uint32(size)) //nolint:gosec
		fromSet := traverse(succ, queue, degree, func(u uint64) bool {
			if visited.Contains(u) {
				return false
			}
			visited.Add(u)
			return true
		})
		seen := make(map[uint64]struct{}, size)
		fromMap := traverse(succ, queue, degree, func(u uint64) bool {
			if _, ok := seen[u]; ok {
				return false
			}
			seen[u] = struct{}{}
			return true
		})
		if fromSet != fromMap || int(visited.Size()) != len(seen) {
			return fmt.Errorf("graph: Set3 walked %d edges reaching %d nodes, the map walked %d reaching %d",
				fromSet, visited.Size(), fromMap, len(seen))
		}
		if fromSet != edges {
			return fmt.Errorf("graph: traversal is not deterministic, %d edges now against %d at setup", fromSet, edges)
		}
		return nil
	}
	w.Release = func() {
		succ = nil
		queue = nil
	}
	return w
}

// traverse runs the breadth-first walk from node zero and returns the number
// of edges it examined. visit reports whether the node was newly discovered;
// it is where the container under test lives, and it is what stops the walk
// from revisiting.
//
// The queue buffer is passed in and re-sliced rather than allocated here, so
// that the traversal itself does not allocate and the only allocation inside a
// batch is the visited container, which is the thing being compared. It is
// large enough for every node, so the append never reallocates.
func traverse(succ []uint32, queue []uint32, degree int, visit func(uint64) bool) int {
	queue = queue[:0]
	if !visit(0) {
		return 0
	}
	queue = append(queue, 0)
	edges := 0
	for head := 0; head < len(queue); head++ {
		base := int(queue[head]) * degree
		for _, next := range succ[base : base+degree] {
			edges++
			if visit(uint64(next)) {
				queue = append(queue, next)
			}
		}
	}
	return edges
}

// sameSize is the check every whole-pass workload ends with: both containers
// hold the same number of elements, and that number is the one the workload
// was built for. A workload that quietly degenerated — a key generator that
// collided, a stream that turned out to hold one distinct value — would
// otherwise be measured happily and reported as a result.
func sameSize(what string, got, want, expected int) error {
	if got != want {
		return fmt.Errorf("%s: Set3 holds %d elements, the map holds %d", what, got, want)
	}
	if got != expected {
		return fmt.Errorf("%s: both hold %d elements, expected %d", what, got, expected)
	}
	return nil
}
