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

	"github.com/TomTonic/rtcompare"
)

// sink exists so the compiler cannot fold a batch body to a constant. See the
// note on Batch in the rtcompare HOWTO: Go does not delete a loop merely
// because nothing reads its result, but a loop whose result is knowable at
// compile time is a different matter, and this is cheap insurance against it.
var sink uint64

// SinkValue returns whatever the batch functions have accumulated so far.
//
// Its only purpose is to be read: a package-level variable that is written and
// never read is dead by every definition the compiler and the linter use, and
// the whole point of the sink is that it must not be. Callers log it and
// ignore the value.
func SinkValue() uint64 { return sink }

// Workload is one comparable pair of batch functions, ready to hand to
// rtcompare, together with everything the report needs to describe it.
//
// The two batch functions must do the same work on the two containers and must
// be equally burdened by everything that is not the container: the same loop,
// the same indexing, the same accumulator. Whatever they share is attenuation
// — it shrinks the measured difference without ever flipping its sign — and
// ItemsPerOp is what converts the measured per-operation cost back into the
// unit a reader cares about.
type Workload struct {
	// Scenario, KeyType and Size identify the cell.
	Scenario string
	KeyType  string
	Size     int

	// Unit names what one reported item is, e.g. "ns/lookup".
	Unit string

	// ItemsPerOp is how many logical items one batch operation covers. A
	// whole-pass workload that builds a set of a million elements per operation
	// reports a million here, so the CSV carries nanoseconds per element rather
	// than per pass.
	ItemsPerOp float64

	// Allocating says whether the batch body allocates. It selects the garbage
	// collection strategy: an allocating workload gets a collection between
	// batches but keeps the collector on, because turning it off would let the
	// heap grow without bound over sixteen thousand batches and would hide the
	// assist work that a real program pays. A non-allocating workload gets both
	// flags and the steadier measurement that comes with them.
	Allocating bool

	// RawBlockHash records whether Set3 hashed this key type as one raw byte
	// block or fell back to hash/maphash. It is the single most useful piece of
	// context for reading a row.
	RawBlockHash bool

	// Set3LoadFactor and MapLoadFactor are the occupancies the two containers
	// actually ended up at, measured rather than assumed. They are zero for
	// workloads that build their container per operation and so have none to
	// weigh.
	//
	// They are the context a speed comparison is meaningless without: a lower
	// load factor buys shorter probe sequences and pays in memory, so two
	// containers at different occupancies are two operating points and not just
	// two layouts. See loadfactor.go.
	Set3LoadFactor float64
	MapLoadFactor  float64

	// Set3Batch and MapBatch are the candidates.
	Set3Batch rtcompare.Batch
	MapBatch  rtcompare.Batch

	// Verify checks that the two batch functions did the same work, not merely
	// the same amount of it. A comparison between two candidates that compute
	// different things is worse than no comparison, because it still produces a
	// confident number. Every builder populates this, and TestWorkloadsAgree
	// runs all of them at a small size before any measurement is trusted.
	Verify func() error

	// Release drops the workload's references to its containers and key
	// material. Cells at eight million string keys hold gigabytes; without
	// this the next cell starts against a heap that has no room left.
	Release func()
}

// candidates wraps the two batch functions as rtcompare candidates, named so
// that an error message from the harness says which side failed.
func (w *Workload) candidates() (set3, native rtcompare.Candidate) {
	// Written as two statements rather than one multi-value return with inline
	// composite literals: gofmt double-indents that form, and golangci-lint's
	// own formatter disagrees with gofmt about it, so the shape is a standing
	// CI failure waiting for a linter upgrade. This one formats identically
	// under both.
	set3 = rtcompare.Candidate{
		Name:  "Set3[" + w.KeyType + "]",
		Batch: w.Set3Batch,
	}
	native = rtcompare.Candidate{
		Name:  "map[" + w.KeyType + "]struct{}",
		Batch: w.MapBatch,
	}
	return set3, native
}

// scenarioInfo is the static description of one workload family: what it
// stands for, what it reports, and where it is applicable.
type scenarioInfo struct {
	// name is the identifier used in the CSV and in SET3_CMP_SCENARIOS.
	name string

	// doc is one sentence on what real thing this stands for. It is printed in
	// the run log and copied into the chart captions.
	doc string

	// unit names one reported item.
	unit string

	// keyTypes lists the key types this scenario runs on. Not every workload
	// makes sense for every key type: a graph traversal has integer node ids
	// and nothing else.
	keyTypes []string

	// maxSize caps the scenario, for workloads that rebuild a whole container
	// per operation and would otherwise spend the entire run budget in one
	// cell. Zero means no cap.
	maxSize int
}

// Scenario names. They are constants because they appear in three places —
// the catalogue, the builder switch, and the chart tool — and a typo in any of
// them would silently drop a row.
const (
	scBuildPresized = "build-presized"
	scBuildGrowing  = "build-growing"
	scDedupStream   = "dedup-stream"
	scLookupHit30   = "lookup-hit30"
	scLookupHit95   = "lookup-hit95"
	scLookupHit30Eq = "lookup-hit30-eqload"
	scLookupHit95Eq = "lookup-hit95-eqload"
	scSlidingWindow = "sliding-window"
	scChurnFresh    = "churn-fresh"
	scMixedIndex    = "mixed-index"
	scGraphVisited  = "graph-visited"
	scIntersect     = "intersect"
	scIterate       = "iterate"
)

var (
	allKeyTypes    = []string{keyTypeUint64, keyTypeString, keyTypeStruct, keyTypeMixed}
	scalarKeyTypes = []string{keyTypeUint64, keyTypeString}
)

// Workloads is the catalogue. Six of the ten are shaped after something a
// program actually does with a set; the other four isolate one cost each,
// because a realistic workload that mixes four costs cannot tell you which of
// them moved.
var Workloads = []scenarioInfo{
	{
		name:     scBuildPresized,
		doc:      "Load a known-size id list into a container created with the right capacity — the case where neither side ever rehashes.",
		unit:     "ns/element",
		keyTypes: allKeyTypes,
	},
	{
		name:     scBuildGrowing,
		doc:      "The same load without a size hint, so the container grows and rehashes its way up. This is what the fill charts in the README measure.",
		unit:     "ns/element",
		keyTypes: scalarKeyTypes,
	},
	{
		name:     scDedupStream,
		doc:      "Deduplicate a skewed event stream: a few keys recur constantly, most appear once. Contains-then-Add, as a deduplicator that has to emit the new ones must do.",
		unit:     "ns/event",
		keyTypes: allKeyTypes,
	},
	{
		name:     scLookupHit30,
		doc:      "Membership filter against a populated set, 30% of queries present — a blocklist or an allowlist check on the request path. Same hit rate as the README's search chart.",
		unit:     "ns/lookup",
		keyTypes: allKeyTypes,
	},
	{
		name:     scLookupHit95,
		doc:      "The same filter where almost everything is present, which exercises the found path instead of the probe-until-empty path.",
		unit:     "ns/lookup",
		keyTypes: []string{keyTypeUint64, keyTypeStruct},
	},
	{
		name: scLookupHit30Eq,
		doc: "The same membership filter with Set3 rehashed to the native map's occupancy, which holds the space/time trade fixed " +
			"and asks what is left: the layout, not the operating point.",
		unit:     "ns/lookup",
		keyTypes: allKeyTypes,
	},
	{
		name:     scLookupHit95Eq,
		doc:      "The hit-heavy filter at the native map's occupancy, for the same reason.",
		unit:     "ns/lookup",
		keyTypes: []string{keyTypeUint64, keyTypeStruct},
	},
	{
		name:     scSlidingWindow,
		doc:      "A fixed-size dedup window: remove the oldest key, insert the newest, forever. The workload that fills a Swiss table with tombstones.",
		unit:     "ns/(remove+insert)",
		keyTypes: scalarKeyTypes,
	},
	{
		name: scChurnFresh,
		doc: "The same fixed-size window over keys the container has never seen — a deduplicator over a live stream, a cache, a work queue. " +
			"sliding-window cycles a bounded ring, so a key returning to the table finds the home group it had before; here every insert probes from a fresh one, " +
			"which is the case that decides whether tombstones are reused or merely accumulate.",
		unit:     "ns/(remove+insert)",
		keyTypes: []string{keyTypeUint64, keyTypeStruct},
	},
	{
		name:     scMixedIndex,
		doc:      "A live membership index: 90% lookups at a 50% hit rate, 10% churn, constant size. The closest thing here to a service under load.",
		unit:     "ns/operation",
		keyTypes: scalarKeyTypes,
	},
	{
		name:     scGraphVisited,
		doc:      "Breadth-first traversal of a random 4-regular graph, using the container as the visited set. Dense integer keys and a fresh set per traversal.",
		unit:     "ns/edge",
		keyTypes: []string{keyTypeUint64},
		maxSize:  SizeL3,
	},
	{
		name:     scIntersect,
		doc:      "Intersect two sets with 20% overlap, as an inverted-index AND query does. Set3.Intersect against the loop a map user writes by hand.",
		unit:     "ns/probe",
		keyTypes: scalarKeyTypes,
	},
	{
		name:     scIterate,
		doc:      "Walk every element once, as a flush or an export does. Set3 scans its groups linearly; the native map walks its buckets in a randomised order.",
		unit:     "ns/element",
		keyTypes: allKeyTypes,
	},
}

// scenarioByName returns the catalogue entry for a scenario name.
func scenarioByName(name string) (scenarioInfo, bool) {
	for _, s := range Workloads {
		if s.name == name {
			return s, true
		}
	}
	return scenarioInfo{}, false
}

// appliesTo reports whether this scenario runs for the given key type and size.
func (s scenarioInfo) appliesTo(keyType string, size int) bool {
	if s.maxSize > 0 && size > s.maxSize {
		return false
	}
	for _, k := range s.keyTypes {
		if k == keyType {
			return true
		}
	}
	return false
}

// newWorkload builds the workload for one cell.
//
// It is the single place where a key type name turns into a concrete type
// parameter. Everything downstream of here is generic over T, and everything
// upstream of here is a string, which is what lets the catalogue, the
// configuration and the CSV all talk about key types without the package
// growing four copies of every workload.
//
// Returns an error for an unknown scenario or key type, which can only happen
// through SET3_CMP_SCENARIOS or SET3_CMP_KEYS.
func newWorkload(scenario, keyType string, size int) (*Workload, error) {
	switch keyType {
	case keyTypeUint64:
		return buildWorkload(scenario, keyType, size, makeUint64Key, func(v uint64) uint64 { return v })
	case keyTypeString:
		return buildWorkload(scenario, keyType, size, makeStringKey, func(v string) uint64 { return uint64(len(v)) + uint64(v[4]) })
	case keyTypeStruct:
		return buildWorkload(scenario, keyType, size, makeTenantKey, func(v tenantKey) uint64 { return v.Object })
	case keyTypeMixed:
		return buildWorkload(scenario, keyType, size, makeEventKey, func(v eventKey) uint64 { return v.ID })
	}
	return nil, fmt.Errorf("unknown key type %q", keyType)
}

// folder reduces a key to a uint64 so that a workload which only needs to
// touch every element can do so without knowing what T is. Both candidates
// call the same folder, so its cost cancels out of the comparison.
type folder[T comparable] func(T) uint64

// buildWorkload dispatches on the scenario name, now that T is known.
func buildWorkload[T comparable](scenario, keyType string, size int, mk keyMaker[T], fold folder[T]) (*Workload, error) {
	info, ok := scenarioByName(scenario)
	if !ok {
		return nil, fmt.Errorf("unknown scenario %q", scenario)
	}
	base := &Workload{
		Scenario:     scenario,
		KeyType:      keyType,
		Size:         size,
		Unit:         info.unit,
		RawBlockHash: rawBlockEligible[T](),
		Release:      func() {},
		Verify:       func() error { return nil },
	}

	switch scenario {
	case scBuildPresized:
		return buildFillWorkload(base, mk, size, true), nil
	case scBuildGrowing:
		return buildFillWorkload(base, mk, size, false), nil
	case scDedupStream:
		return buildDedupWorkload(base, mk, size), nil
	case scLookupHit30:
		return buildLookupWorkload(base, mk, size, 0.30, false), nil
	case scLookupHit95:
		return buildLookupWorkload(base, mk, size, 0.95, false), nil
	case scLookupHit30Eq:
		return buildLookupWorkload(base, mk, size, 0.30, true), nil
	case scLookupHit95Eq:
		return buildLookupWorkload(base, mk, size, 0.95, true), nil
	case scSlidingWindow:
		return buildSlidingWindowWorkload(base, mk, size), nil
	case scChurnFresh:
		return buildChurnFreshWorkload(base, mk, size), nil
	case scMixedIndex:
		return buildMixedIndexWorkload(base, mk, size), nil
	case scGraphVisited:
		return buildGraphWorkload(base, size), nil
	case scIntersect:
		return buildIntersectWorkload(base, mk, size), nil
	case scIterate:
		return buildIterateWorkload(base, mk, fold, size), nil
	}
	return nil, fmt.Errorf("scenario %q has no builder", scenario)
}

// nextPow2 rounds up to a power of two, so a stream can be indexed with a mask
// instead of a modulo. The masking is inside the timed region and identical on
// both sides; a division would be too, but it would be a larger constant share
// of a two-nanosecond operation.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}
