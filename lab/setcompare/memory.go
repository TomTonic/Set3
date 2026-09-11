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
	"runtime"
	"sort"

	set3 "github.com/TomTonic/Set3"
)

// The four fill histories the memory pass measures. A hash table's footprint
// is not a function of how many elements it holds; it is a function of how it
// came to hold them, and these are the four histories that a program actually
// produces.
const (
	// shapePresized is a container created at the right capacity and filled.
	// The best case, and the one a size hint buys you.
	shapePresized = "presized"

	// shapeGrown is a container created empty and grown to the same size. It
	// ends up at whatever capacity the growth sequence landed on, which is
	// somewhere between the element count and twice it.
	shapeGrown = "grown"

	// shapeHalfRemoved is presized, filled, then emptied of every second
	// element. Neither implementation gives the memory back — the question is
	// what the remaining half costs.
	shapeHalfRemoved = "half-removed"

	// shapeWindow is a sliding window that has already churned through four
	// times its own size. This is where a Swiss table's tombstones would show
	// up if they accumulated.
	shapeWindow = "window-steady"
)

// MemoryShapes is the list in the order the CSV and the charts use.
var MemoryShapes = []string{shapePresized, shapeGrown, shapeHalfRemoved, shapeWindow}

// Where the native map's bytes go, since the measured figure surprises people
// and a surprising number that cannot be explained is a number that should not
// be quoted. Measured on Go 1.26/amd64, map[uint64]struct{} costs about 36
// bytes per element, against Set3's 11. Two mechanisms account for all of it,
// and both are properties of the runtime rather than of anyone's benchmark:
//
//  1. A struct{} value is not free. The map's slot is a struct of key and
//     element, and Go pads a struct whose last field is zero-sized so that a
//     pointer one past the end cannot escape the object. So
//     struct{uint64; struct{}} occupies 16 bytes, exactly as much as
//     struct{uint64; uint64} — which is why a map[uint64]struct{} and a
//     map[uint64]uint64 measure identically here, to within sixteen bytes over
//     a million elements. With eight control bytes per group of eight slots,
//     the native map pays 17 bytes per slot. Set3 stores the keys themselves in
//     a [8]T array with one 64-bit control word beside it, so it pays 9.
//
//  2. The occupancy differs. At 17 bytes per slot the measured 36 bytes per
//     element works out to 2.12 slots per element, so the map is running at
//     about 47% full — a consequence of how a Swiss table splits and how its
//     directory grows, not of the keys. Set3 runs to its limit of 6.67 slots
//     in 8, which is 83%.
//
// Nine bytes per slot at 83% against seventeen at 47% is the whole of the
// difference. Note what it is not: it is not a claim that one implementation
// wastes memory and the other does not. It is the price of a lower load factor
// and a padded slot, and a lower load factor buys shorter probe sequences.

// MemoryResult is one row of the memory CSV: what a populated container costs,
// measured as retained heap after a full collection rather than as anything
// the container reports about itself.
type MemoryResult struct {
	Shape    string
	KeyType  string
	Size     int
	Elements int // elements actually held, which is Size/2 for half-removed

	Set3Bytes float64
	MapBytes  float64

	Set3BytesPerElement float64
	MapBytesPerElement  float64

	// RatioSet3OverMap is below one when Set3 is the smaller of the two. It is
	// the headline number of this pass.
	RatioSet3OverMap float64

	// Set3BuildBytes and MapBuildBytes are everything allocated on the way to
	// the final container, including the intermediate tables a growing
	// container throws away. Reading them next to the retained figures is the
	// difference between "how much does it cost to hold" and "how much did it
	// cost to get there".
	Set3BuildBytes   float64
	MapBuildBytes    float64
	Set3BuildMallocs float64
	MapBuildMallocs  float64

	Note string
}

// MeasureMemory runs the memory pass over every shape, key type and size in
// the configuration.
//
// Each cell is measured Config.MemRepeats times and the median retained heap
// is reported. Repetition is not there to beat down noise — a container of a
// given size allocates a deterministic number of bytes — but to survive the one
// run in a handful where the collector had not finished with something else,
// which shows up as a single implausible reading and which a median discards.
//
// logf receives one line per cell.
func MeasureMemory(cfg Config, logf func(format string, args ...any)) []MemoryResult {
	var out []MemoryResult
	for _, shape := range MemoryShapes {
		for _, keyType := range cfg.KeyTypes {
			for _, size := range cfg.Sizes {
				res := measureMemoryCell(shape, keyType, size, cfg.MemRepeats)
				out = append(out, res)
				logf("%-14s %-11s n=%-9d Set3 %9.2f B/elem  map %9.2f B/elem  ratio %.3f  (build: %.1f vs %.1f B/elem)%s",
					res.Shape, res.KeyType, res.Size,
					res.Set3BytesPerElement, res.MapBytesPerElement, res.RatioSet3OverMap,
					res.Set3BuildBytes, res.MapBuildBytes, noteSuffix(res.Note))
			}
		}
	}
	return out
}

// measureMemoryCell dispatches on the key type, the same way newWorkload does.
func measureMemoryCell(shape, keyType string, size, repeats int) MemoryResult {
	switch keyType {
	case keyTypeUint64:
		return memoryFor(shape, keyType, size, repeats, makeUint64Key)
	case keyTypeString:
		return memoryFor(shape, keyType, size, repeats, makeStringKey)
	case keyTypeStruct:
		return memoryFor(shape, keyType, size, repeats, makeTenantKey)
	case keyTypeMixed:
		return memoryFor(shape, keyType, size, repeats, makeEventKey)
	}
	return MemoryResult{Shape: shape, KeyType: keyType, Size: size, Note: "unknown key type"}
}

// memoryFor measures both containers for one shape.
//
// The key material is allocated before the first MemStats reading and stays
// alive throughout, so it appears in both readings and cancels. That matters
// most for the string key type: the strings themselves belong to the key slice,
// and neither container copies them, so what is measured is the table and only
// the table — which is the honest comparison, since both containers store the
// same 16-byte string headers.
func memoryFor[T comparable](shape, keyType string, size, repeats int, mk keyMaker[T]) MemoryResult {
	res := MemoryResult{Shape: shape, KeyType: keyType, Size: size}

	ringLen := nextPow2(2 * size)
	keys := buildKeys(mk, memberDomain, ringLen)
	defer func() { keys = nil }()

	set3Bytes := make([]float64, 0, repeats)
	mapBytes := make([]float64, 0, repeats)
	var elements int

	for range repeats {
		retained, built, mallocs := measureOnce(func() any {
			s, n := buildSet3Shape(shape, keys, size)
			elements = n
			return s
		}, keys)
		set3Bytes = append(set3Bytes, retained)
		res.Set3BuildBytes, res.Set3BuildMallocs = built, mallocs

		retained, built, mallocs = measureOnce(func() any {
			m, n := buildMapShape(shape, keys, size)
			elements = n
			return m
		}, keys)
		mapBytes = append(mapBytes, retained)
		res.MapBuildBytes, res.MapBuildMallocs = built, mallocs
	}

	res.Elements = elements
	res.Set3Bytes = median(set3Bytes)
	res.MapBytes = median(mapBytes)
	if elements > 0 {
		res.Set3BytesPerElement = res.Set3Bytes / float64(elements)
		res.MapBytesPerElement = res.MapBytes / float64(elements)
		res.Set3BuildBytes /= float64(elements)
		res.MapBuildBytes /= float64(elements)
		res.Set3BuildMallocs /= float64(elements)
		res.MapBuildMallocs /= float64(elements)
	}
	if res.MapBytes > 0 {
		res.RatioSet3OverMap = res.Set3Bytes / res.MapBytes
	}
	if res.Set3Bytes <= 0 || res.MapBytes <= 0 {
		res.Note = "a reading came out non-positive; the heap moved under the measurement"
	}
	return res
}

// measureOnce builds one container and reports what it retained, what it
// allocated on the way, and how many objects that took.
//
// Retained heap is read after two collections with the container still
// reachable, which is the only way to ask "what does holding this cost" rather
// than "what did building it cost". The two collections are not superstition:
// the first can leave objects finalisable rather than freed, and a single one
// has been observed to leave several percent of a large table uncounted.
//
// keepAlive is the trap this signature exists to close, and it is worth
// stating plainly because it silently corrupted an earlier version of this
// pass. The key material is allocated before the first reading so that it
// appears in both and cancels — but only if it is still alive at the second
// reading. Go's liveness analysis ends a variable's life at its last use, so
// the keys slice handed to build was routinely dead by the time the collections
// ran, which freed it inside the measured window and subtracted exactly eight
// bytes per element from the answer. The same map then measured 36 bytes per
// element or 28 depending on whether the caller happened to use its keys again
// afterwards. Anything the measurement assumes is constant must be named here.
func measureOnce(build func() any, keepAlive ...any) (retained, allocated, mallocs float64) {
	runtime.GC()
	runtime.GC()

	var before, mid, after runtime.MemStats
	runtime.ReadMemStats(&before)

	container := build()

	runtime.ReadMemStats(&mid)
	runtime.GC()
	runtime.GC()
	runtime.ReadMemStats(&after)
	runtime.KeepAlive(container)
	runtime.KeepAlive(keepAlive)

	return diffBytes(before.HeapAlloc, after.HeapAlloc),
		diffBytes(before.TotalAlloc, mid.TotalAlloc),
		diffBytes(before.Mallocs, mid.Mallocs)
}

// diffBytes subtracts two counters as signed values, because HeapAlloc can go
// down between two readings when something unrelated was freed, and an
// unsigned subtraction would turn that into a number the size of the address
// space.
func diffBytes(before, after uint64) float64 {
	return float64(int64(after) - int64(before)) //nolint:gosec
}

// buildSet3Shape builds a Set3 with the given fill history and returns it with
// the number of elements it ends up holding.
func buildSet3Shape[T comparable](shape string, keys []T, size int) (*set3.Set3[T], int) {
	switch shape {
	case shapeGrown:
		s := set3.Empty[T]()
		for i := range size {
			s.Add(keys[i])
		}
		return s, size

	case shapeHalfRemoved:
		s := set3.EmptyWithCapacity[T](uint32(size)) //nolint:gosec
		for i := range size {
			s.Add(keys[i])
		}
		for i := 0; i < size; i += 2 {
			s.Remove(keys[i])
		}
		return s, int(s.Size())

	case shapeWindow:
		s := set3.EmptyWithCapacity[T](uint32(size)) //nolint:gosec
		for i := range size {
			s.Add(keys[i])
		}
		mask := len(keys) - 1
		for i := range 4 * size {
			s.Remove(keys[i&mask])
			s.Add(keys[(i+size)&mask])
		}
		return s, int(s.Size())

	default: // shapePresized
		s := set3.EmptyWithCapacity[T](uint32(size)) //nolint:gosec
		for i := range size {
			s.Add(keys[i])
		}
		return s, size
	}
}

// buildMapShape is buildSet3Shape for the native map, step for step, so that
// the two are measured after the same history and not merely at the same size.
func buildMapShape[T comparable](shape string, keys []T, size int) (map[T]struct{}, int) {
	switch shape {
	case shapeGrown:
		m := make(map[T]struct{})
		for i := range size {
			m[keys[i]] = struct{}{}
		}
		return m, size

	case shapeHalfRemoved:
		m := make(map[T]struct{}, size)
		for i := range size {
			m[keys[i]] = struct{}{}
		}
		for i := 0; i < size; i += 2 {
			delete(m, keys[i])
		}
		return m, len(m)

	case shapeWindow:
		m := make(map[T]struct{}, size)
		for i := range size {
			m[keys[i]] = struct{}{}
		}
		mask := len(keys) - 1
		for i := range 4 * size {
			delete(m, keys[i&mask])
			m[keys[(i+size)&mask]] = struct{}{}
		}
		return m, len(m)

	default: // shapePresized
		m := make(map[T]struct{}, size)
		for i := range size {
			m[keys[i]] = struct{}{}
		}
		return m, size
	}
}

// median returns the middle value of a copy of the input, or zero for an empty
// input. It is here rather than borrowed from rtcompare because the memory
// pass has no other reason to depend on that package.
func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	c := append([]float64(nil), v...)
	sort.Float64s(c)
	return c[len(c)/2]
}
