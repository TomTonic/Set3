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
	"math"

	set3 "github.com/TomTonic/Set3"
	"github.com/TomTonic/rtcompare"
)

// The load curve is the measurement this whole comparison was missing.
//
// Every other scenario compares two containers at whatever occupancy each one
// happens to choose, which conflates two different things: how the table is
// laid out, and where on the space/time curve it is being operated. Set3 fills
// to about 83%, Go's map to about 47%, and a lower occupancy buys shorter
// probe sequences at the cost of memory. A single number comparing them is the
// sum of a layout difference and a tuning difference, and it cannot be read as
// either.
//
// Set3 has RehashToCapacity, so its operating point is a parameter rather than
// a property. The native map has no such knob: it is one point. This pass
// therefore measures Set3 at a range of occupancies and the map once, and
// reports each as a (bytes per element, nanoseconds per lookup) pair. The
// result is a curve and a point, and the question "which is better" turns into
// the question it should always have been: better at what price.
//
// The occupancies are chosen to span from Set3's own default down to the point
// where it occupies as many bytes as the native map does, so the curve covers
// the whole range a caller could reasonably ask for.

// loadCurveTargets are the occupancies Set3 is measured at.
//
// 0.80 is what EmptyWithCapacity(n) filled with n elements produces, which is
// Set3 as shipped. 0.25 is where a uint64 Set3 costs the same 36 bytes per
// element as the native map, so the last point is a like-for-like memory
// comparison. The rest fill in the curve between them.
var loadCurveTargets = []float64{0.80, 0.65, 0.50, 0.40, 0.30, 0.25}

// LoadCurvePoint is one measured operating point.
//
// The native map contributes exactly one of these per cell, with TargetLoad
// zero and DeltaPct zero, because it is the reference the Set3 points are
// compared against rather than a point on any curve.
type LoadCurvePoint struct {
	KeyType string
	Size    int

	// Impl is "Set3" or "map".
	Impl string

	// TargetLoad is the occupancy that was asked for; Load is what the
	// container actually ended up at, which differs because Set3 rounds its
	// group count up to a prime.
	TargetLoad float64
	Load       float64

	// BytesPerElement is retained heap, measured the same way the memory pass
	// measures it.
	BytesPerElement float64

	// NsPerLookup is the median cost of one membership query at this point.
	NsPerLookup float64

	// DeltaPct and its interval compare this Set3 point against the native map,
	// positive meaning Set3 is faster. Zero for the map's own row.
	DeltaPct      float64
	CILowPct      float64
	CIHighPct     float64
	NoiseFloorPct float64
	Resolved      bool

	Note string
}

// MeasureLoadCurve runs the load-curve pass over the configured key types, at
// the sizes in cfg.LoadCurveSizes.
//
// One cell is one (key type, size): the native map is measured once, then Set3
// is measured at each occupancy in loadCurveTargets and compared against it.
// logf receives one line per point.
func MeasureLoadCurve(cfg Config, logf func(format string, args ...any)) []LoadCurvePoint {
	var out []LoadCurvePoint
	for _, keyType := range cfg.KeyTypes {
		for _, size := range cfg.LoadCurveSizes {
			out = append(out, loadCurveCell(cfg, keyType, size, logf)...)
		}
	}
	return out
}

// loadCurveCell dispatches on the key type, as the other passes do.
func loadCurveCell(cfg Config, keyType string, size int, logf func(string, ...any)) []LoadCurvePoint {
	switch keyType {
	case keyTypeUint64:
		return loadCurveFor(cfg, keyType, size, makeUint64Key, logf)
	case keyTypeString:
		return loadCurveFor(cfg, keyType, size, makeStringKey, logf)
	case keyTypeStruct:
		return loadCurveFor(cfg, keyType, size, makeTenantKey, logf)
	case keyTypeMixed:
		return loadCurveFor(cfg, keyType, size, makeEventKey, logf)
	}
	return nil
}

// loadCurveFor measures one cell: the map once, then Set3 at every target.
func loadCurveFor[T comparable](cfg Config, keyType string, size int, mk keyMaker[T], logf func(string, ...any)) []LoadCurvePoint {
	members := buildKeys(mk, memberDomain, size)
	misses := buildKeys(mk, missDomain, size)
	queries := mixedQueries(members, misses, 0.30, queryStreamLen(size))
	defer func() { members, misses, queries = nil, nil, nil }()

	nativeMap := make(map[T]struct{}, size)
	for _, k := range members {
		nativeMap[k] = struct{}{}
	}
	mapBytes, _, _ := measureOnce(func() any {
		m := make(map[T]struct{}, size)
		for _, k := range members {
			m[k] = struct{}{}
		}
		return m
	}, members)

	var points []LoadCurvePoint
	var mapCosts []float64
	for _, target := range loadCurveTargets {
		p, mapNs := loadCurvePoint(cfg, keyType, size, target, members, queries, nativeMap)
		points = append(points, p)
		if mapNs > 0 {
			mapCosts = append(mapCosts, mapNs)
		}
		logf("%-11s n=%-9d load %.2f (asked %.2f)  %7.2f B/elem  %6.2f ns/lookup  vs map %+6.1f%% [%+.1f,%+.1f] floor %.2f%% %v%s",
			p.KeyType, p.Size, p.Load, p.TargetLoad, p.BytesPerElement, p.NsPerLookup,
			p.DeltaPct, p.CILowPct, p.CIHighPct, p.NoiseFloorPct, p.Resolved, noteSuffix(p.Note))
	}

	// The map's cost is the median of what the interleaved runs measured for it,
	// one per Set3 point. Taking it from those runs rather than from a separate
	// measurement is what makes the two columns of a row comparable: they were
	// measured against each other, alternating, on the same machine state.
	reference := LoadCurvePoint{
		KeyType:         keyType,
		Size:            size,
		Impl:            "map",
		Load:            loadFactor(mapBytes, bytesPerSlotNativeMap[T](), size),
		BytesPerElement: mapBytes / float64(size),
		NsPerLookup:     median(mapCosts),
	}
	logf("%-11s n=%-9d load %.2f              %7.2f B/elem  %6.2f ns/lookup  (the reference, median of %d interleaved runs)",
		"map", size, reference.Load, reference.BytesPerElement, reference.NsPerLookup, len(mapCosts))

	// The reference goes first so the CSV reads as "here is the point, here is
	// the curve against it".
	return append([]LoadCurvePoint{reference}, points...)
}

// loadCurvePoint measures Set3 at one occupancy against the native map.
func loadCurvePoint[T comparable](
	cfg Config, keyType string, size int, target float64,
	members, queries []T, nativeMap map[T]struct{},
) (LoadCurvePoint, float64) {
	capacity := set3CapacityForSlots(float64(size) / target)
	s := set3.EmptyWithCapacity[T](capacity)
	for _, k := range members {
		s.Add(k)
	}
	set3Bytes, _, _ := measureOnce(func() any {
		fresh := set3.EmptyWithCapacity[T](capacity)
		for _, k := range members {
			fresh.Add(k)
		}
		return fresh
	}, members)

	p := LoadCurvePoint{
		KeyType:         keyType,
		Size:            size,
		Impl:            "Set3",
		TargetLoad:      target,
		Load:            loadFactor(set3Bytes, bytesPerSlotSet3[T](), size),
		BytesPerElement: set3Bytes / float64(size),
	}

	mask := uint64(len(queries) - 1)
	var posSet, posMap uint64
	set3Candidate := rtcompare.Candidate{
		Name: "Set3[" + keyType + "]",
		Batch: func(n uint64) {
			pos, acc := posSet, uint64(0)
			for range n {
				if s.Contains(queries[pos&mask]) {
					acc++
				}
				pos++
			}
			posSet = pos
			sink += acc
		},
	}
	mapCandidate := rtcompare.Candidate{
		Name: "map[" + keyType + "]struct{}",
		Batch: func(n uint64) {
			pos, acc := posMap, uint64(0)
			for range n {
				if _, ok := nativeMap[queries[pos&mask]]; ok {
					acc++
				}
				pos++
			}
			posMap = pos
			sink += acc
		},
	}

	// The same options every steady-state workload gets: a three-millisecond
	// batch and no forced collections. See collectOptionsFor.
	opt := rtcompare.CollectOptions{
		Order:                rtcompare.OrderABBA,
		MaxQuantizationError: quantizationTargetFor(targetBatchDuration),
	}
	step, _, _ := planBudget(float64(targetBatchDuration.Nanoseconds()), cfg.Budget, cfg.Resamples)
	opt.Repeats = step.repeats

	report, err := rtcompare.Compare(set3Candidate, mapCandidate, rtcompare.CompareOptions{
		Collect:        opt,
		ValidationRuns: step.validationRuns,
		Level:          cfg.Level,
		Resamples:      cfg.Resamples,
	})
	if err != nil {
		p.Note = "rtcompare: " + err.Error()
		return p, 0
	}
	p.NsPerLookup = report.NsPerOpA
	p.DeltaPct = report.Estimate.Delta * 100
	p.CILowPct = report.Estimate.Low * 100
	p.CIHighPct = report.Estimate.High * 100
	p.NoiseFloorPct = report.NoiseFloor * 100
	p.Resolved = report.Resolved
	p.Note = joinNotes(p.Note, joinWarnings(report.Warnings))

	// The map's cost from this same interleaved run, returned so the caller can
	// build the reference point out of measurements that alternated with the
	// Set3 ones rather than a separate run at a different moment.
	return p, report.NsPerOpB
}

// joinWarnings renders rtcompare's warnings as one field.
func joinWarnings(warnings []string) string {
	out := ""
	for _, w := range warnings {
		out = joinNotes(out, w)
	}
	return out
}

// SummarizeLoadCurve renders the headline the curve exists to produce: where on
// it Set3 overtakes the native map, and what that point costs in memory.
func SummarizeLoadCurve(points []LoadCurvePoint) string {
	type cell struct {
		keyType string
		size    int
	}
	mapPoint := map[cell]LoadCurvePoint{}
	for _, p := range points {
		if p.Impl == "map" {
			mapPoint[cell{p.KeyType, p.Size}] = p
		}
	}

	out := "Set3's space/time curve against the native map's single point:\n"
	for c, m := range mapPoint {
		best := LoadCurvePoint{DeltaPct: math.Inf(-1)}
		var crossover *LoadCurvePoint
		for i, p := range points {
			if p.Impl != "Set3" || p.KeyType != c.keyType || p.Size != c.size {
				continue
			}
			if p.DeltaPct > best.DeltaPct {
				best = p
			}
			// The cheapest point (highest load) that is still faster.
			if p.Resolved && p.DeltaPct > 0 && (crossover == nil || p.BytesPerElement < crossover.BytesPerElement) {
				crossover = &points[i]
			}
		}
		out += fmt.Sprintf("  %s n=%d: map is %.1f B/elem at %.2f ns.\n",
			c.keyType, c.size, m.BytesPerElement, m.NsPerLookup)
		if crossover != nil {
			out += fmt.Sprintf("      Set3 first wins at %.2f load, %.1f B/elem (%.0f%% of the map's) and %+.1f%%.\n",
				crossover.Load, crossover.BytesPerElement, crossover.BytesPerElement/m.BytesPerElement*100, crossover.DeltaPct)
		} else {
			out += "      Set3 did not resolve a win at any occupancy measured.\n"
		}
		if best.Impl != "" {
			out += fmt.Sprintf("      best is %.2f load, %.1f B/elem, %+.1f%%.\n",
				best.Load, best.BytesPerElement, best.DeltaPct)
		}
	}
	return out
}

// loadCurveHeader names the columns of loadcurve.csv.
var loadCurveHeader = []string{
	"keytype", "size", "impl", "target_load", "load", "bytes_per_element", "ns_per_lookup",
	"delta_pct", "ci_low_pct", "ci_high_pct", "noise_floor_pct", "resolved", "note",
}

// loadCurveRecords renders the points for the CSV writer.
func loadCurveRecords(points []LoadCurvePoint) [][]string {
	out := make([][]string, 0, len(points))
	for _, p := range points {
		out = append(out, []string{
			p.KeyType, itoa(p.Size), p.Impl, f(p.TargetLoad, 4), f(p.Load, 4),
			f(p.BytesPerElement, 3), f(p.NsPerLookup, 4),
			f(p.DeltaPct, 3), f(p.CILowPct, 3), f(p.CIHighPct, 3), f(p.NoiseFloorPct, 4),
			b(p.Resolved), clean(p.Note),
		})
	}
	return out
}
