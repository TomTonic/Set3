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

// This file holds the steady-state workloads. Their containers are built once,
// before any measurement, and one batch operation is one operation on the
// living container. Nothing here allocates inside the timed region, so these
// run with the collector disabled and a collection between batches, which is
// the quietest configuration rtcompare offers.
//
// The ones that mutate are stationary by construction: every insert is paired
// with a remove, so the container is the same size after a batch as before it.
// A workload that grew during the run would show up as drift and would be
// unusable, because the bootstrap cannot see a trend across a series.

// queryStreamLen picks the length of a precomputed query stream: long enough
// that the branch predictor cannot memorise it, short enough that the stream
// itself does not evict the container from cache. It scales with the container
// so that a thousand-element set is not measured through a half-megabyte
// stream, and it is a power of two so the batch loop indexes it with a mask.
func queryStreamLen(size int) int {
	n := nextPow2(4 * size)
	if n < 1024 {
		n = 1024
	}
	if n > 1<<16 {
		n = 1 << 16
	}
	return n
}

// buildLookupWorkload measures membership queries against a populated
// container at a fixed hit ratio.
//
// This is the workload most set-shaped code actually runs: build once, query
// forever. The hit ratio matters more than it looks, which is why the suite
// runs two of them. A miss in a Swiss table usually ends at the first group,
// because the control bytes rule out all eight slots at once and an empty slot
// stops the probe; a hit has to compare the key. The native map's two paths
// differ in the same way but not by the same amount, so a single hit ratio
// would describe one half of the truth.
//
// equalLoad selects the second operating point. With it false, Set3 is built
// the way a caller gets it — filled to about 83% — and the comparison includes
// the fact that the two containers chose different points on the space/time
// curve. With it true, Set3 is given the capacity at which it occupies as many
// slots as the native map does, so the probe sequences are comparable and what
// is left is the layout. Neither is the "correct" comparison; they answer
// different questions. See loadfactor.go.
func buildLookupWorkload[T comparable](w *Workload, mk keyMaker[T], size int, hitRatio float64, equalLoad bool) *Workload {
	members := buildKeys(mk, memberDomain, size)
	misses := buildKeys(mk, missDomain, size)
	queries := mixedQueries(members, misses, hitRatio, queryStreamLen(size))
	mask := uint64(len(queries) - 1)

	capacity := uint32(size) //nolint:gosec
	if equalLoad {
		capacity, w.Set3LoadFactor, w.MapLoadFactor = set3CapacityMatchingMap(members, size)
	} else {
		_, w.MapLoadFactor = measureNativeMapSlots(members, size)
		w.Set3LoadFactor = measureSet3Load(members, size, capacity)
	}

	s := set3.EmptyWithCapacity[T](capacity)
	for _, k := range members {
		s.Add(k)
	}
	m := make(map[T]struct{}, size)
	for _, k := range members {
		m[k] = struct{}{}
	}

	w.ItemsPerOp = 1
	w.Allocating = false

	var posSet, posMap uint64
	w.Set3Batch = func(n uint64) {
		p, acc := posSet, uint64(0)
		for range n {
			if s.Contains(queries[p&mask]) {
				acc++
			}
			p++
		}
		posSet = p
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		p, acc := posMap, uint64(0)
		for range n {
			if _, ok := m[queries[p&mask]]; ok {
				acc++
			}
			p++
		}
		posMap = p
		sink += acc
	}
	w.Verify = func() error {
		var setHits, mapHits int
		for _, q := range queries {
			if s.Contains(q) {
				setHits++
			}
			if _, ok := m[q]; ok {
				mapHits++
			}
		}
		if setHits != mapHits {
			return fmt.Errorf("lookup: Set3 answered %d of %d queries with yes, the map answered %d",
				setHits, len(queries), mapHits)
		}
		observed := float64(setHits) / float64(len(queries))
		if observed < hitRatio-0.05 || observed > hitRatio+0.05 {
			return fmt.Errorf("lookup: hit ratio came out at %.3f, expected about %.2f", observed, hitRatio)
		}
		return sameContents(s, m)
	}
	w.Release = func() {
		members, misses, queries = nil, nil, nil
		s = nil
		m = nil
	}
	return w
}

// buildSlidingWindowWorkload measures a fixed-size window over an endless key
// stream: every operation removes the oldest key and inserts a new one, so the
// container holds exactly size elements forever.
//
// This is a dedup window with an expiry, a rate limiter's recent-request set, a
// bounded "have I seen this" cache — and for a Swiss table it is the adversarial
// case, because a removal leaves a tombstone rather than an empty slot and a
// tombstone does not terminate a probe. Set3 reuses tombstones on insert, so
// the table should reach a steady state rather than degrade; whether it does is
// exactly what this workload is here to find out. A run that degrades shows up
// as drift in the report, not as a quietly worse average.
//
// The key ring is twice the window, so the key being inserted was last seen
// long enough ago to have left the window entirely.
func buildSlidingWindowWorkload[T comparable](w *Workload, mk keyMaker[T], size int) *Workload {
	ringLen := nextPow2(2 * size)
	ring := buildKeys(mk, memberDomain, ringLen)
	ringMask := uint64(ringLen - 1)
	window := uint64(size)

	s := set3.EmptyWithCapacity[T](uint32(size)) //nolint:gosec
	m := make(map[T]struct{}, size)
	for i := range size {
		s.Add(ring[i])
		m[ring[i]] = struct{}{}
	}

	w.ItemsPerOp = 1
	w.Allocating = false

	var curSet, curMap uint64
	w.Set3Batch = func(n uint64) {
		c := curSet
		for range n {
			s.Remove(ring[c&ringMask])
			s.Add(ring[(c+window)&ringMask])
			c++
		}
		curSet = c
		sink += uint64(s.Size())
	}
	w.MapBatch = func(n uint64) {
		c := curMap
		for range n {
			delete(m, ring[c&ringMask])
			m[ring[(c+window)&ringMask]] = struct{}{}
			c++
		}
		curMap = c
		sink += uint64(len(m))
	}
	w.Verify = func() error {
		const steps = 1_000
		w.Set3Batch(steps)
		w.MapBatch(steps)
		if curSet != curMap {
			return fmt.Errorf("sliding-window: cursors diverged, %d against %d", curSet, curMap)
		}
		if int(s.Size()) != size || len(m) != size {
			return fmt.Errorf("sliding-window: window drifted off its size, Set3 holds %d and the map holds %d, expected %d",
				s.Size(), len(m), size)
		}
		return sameContents(s, m)
	}
	w.Release = func() {
		ring = nil
		s = nil
		m = nil
	}
	return w
}

// Operation kinds in the mixed-index script, packed two bits above a 30-bit
// argument. One packed slice rather than two parallel ones, because the script
// is read inside the timed region and a second stream of cache misses would be
// charged to both candidates for no benefit.
const (
	opLookupMember = uint32(0) << 30
	opLookupMiss   = uint32(1) << 30
	opAdvance      = uint32(2) << 30
	opKindShift    = 30
	opArgMask      = uint32(1)<<30 - 1
)

// buildMixedIndexWorkload measures a live membership index: mostly lookups, a
// steady trickle of churn, constant size.
//
// Nine operations in ten are lookups, half of which hit; the tenth retires the
// oldest key and admits a new one. That is the shape of a service that keeps a
// set of currently-valid tokens, active sessions or recently-seen ids and
// answers questions about it far more often than it changes it. None of the
// single-operation workloads predicts this one, because the interesting part is
// what the churn does to the lookups: on a Swiss table the removals leave
// tombstones that every later probe has to walk past.
//
// The operation script is precomputed and shared by both candidates, so the
// two see the same sequence of decisions in the same order.
func buildMixedIndexWorkload[T comparable](w *Workload, mk keyMaker[T], size int) *Workload {
	ringLen := nextPow2(2 * size)
	ring := buildKeys(mk, memberDomain, ringLen)
	ringMask := uint64(ringLen - 1)
	window := uint64(size)

	missLen := nextPow2(min(size, 1<<16))
	misses := buildKeys(mk, missDomain, missLen)
	missMask := uint32(missLen - 1) //nolint:gosec

	script := buildMixedScript(1<<16, size)
	scriptMask := uint64(len(script) - 1)

	s := set3.EmptyWithCapacity[T](uint32(size)) //nolint:gosec
	m := make(map[T]struct{}, size)
	for i := range size {
		s.Add(ring[i])
		m[ring[i]] = struct{}{}
	}

	w.ItemsPerOp = 1
	w.Allocating = false

	var posSet, posMap, curSet, curMap uint64
	w.Set3Batch = func(n uint64) {
		p, c, acc := posSet, curSet, uint64(0)
		for range n {
			e := script[p&scriptMask]
			p++
			switch e >> opKindShift {
			case 0:
				if s.Contains(ring[(c+uint64(e&opArgMask))&ringMask]) {
					acc++
				}
			case 1:
				if s.Contains(misses[e&opArgMask&missMask]) {
					acc++
				}
			default:
				s.Remove(ring[c&ringMask])
				s.Add(ring[(c+window)&ringMask])
				c++
			}
		}
		posSet, curSet = p, c
		sink += acc
	}
	w.MapBatch = func(n uint64) {
		p, c, acc := posMap, curMap, uint64(0)
		for range n {
			e := script[p&scriptMask]
			p++
			switch e >> opKindShift {
			case 0:
				if _, ok := m[ring[(c+uint64(e&opArgMask))&ringMask]]; ok {
					acc++
				}
			case 1:
				if _, ok := m[misses[e&opArgMask&missMask]]; ok {
					acc++
				}
			default:
				delete(m, ring[c&ringMask])
				m[ring[(c+window)&ringMask]] = struct{}{}
				c++
			}
		}
		posMap, curMap = p, c
		sink += acc
	}
	w.Verify = func() error {
		const steps = 10_000
		w.Set3Batch(steps)
		w.MapBatch(steps)
		if posSet != posMap || curSet != curMap {
			return fmt.Errorf("mixed-index: the two candidates read different scripts, script pos %d/%d cursor %d/%d",
				posSet, posMap, curSet, curMap)
		}
		if int(s.Size()) != len(m) {
			return fmt.Errorf("mixed-index: Set3 holds %d elements, the map holds %d", s.Size(), len(m))
		}
		return sameContents(s, m)
	}
	w.Release = func() {
		ring, misses, script = nil, nil, nil
		s = nil
		m = nil
	}
	return w
}

// buildMixedScript lays out the operation mix: 45% member lookups, 45% miss
// lookups, 10% churn, in a shuffled order rather than a repeating pattern.
//
// The member argument is an offset from the churn cursor rather than an
// absolute index, so it names a key that is currently in the window no matter
// how far the window has slid by the time the entry is reached.
func buildMixedScript(length, size int) []uint32 {
	rng := newSplitmix(0x5eed_5ca1_ab1e_0004)
	script := make([]uint32, length)
	for i := range script {
		switch r := rng.float64(); {
		case r < 0.45:
			script[i] = opLookupMember | (rng.uint32n(uint32(size)) & opArgMask) //nolint:gosec
		case r < 0.90:
			script[i] = opLookupMiss | (uint32(rng.next()) & opArgMask)
		default:
			script[i] = opAdvance
		}
	}
	return script
}

// sameContents reports whether a Set3 and a native map hold exactly the same
// elements.
//
// The size check alone would not catch the failure mode that matters: two
// containers can hold the same number of different things, which is precisely
// what a workload whose two sides have drifted out of step produces.
func sameContents[T comparable](s *set3.Set3[T], m map[T]struct{}) error {
	if int(s.Size()) != len(m) {
		return fmt.Errorf("Set3 holds %d elements, the map holds %d", s.Size(), len(m))
	}
	for e := range s.MutableRange() {
		if _, ok := m[e]; !ok {
			return fmt.Errorf("Set3 holds an element the map does not")
		}
	}
	return nil
}
