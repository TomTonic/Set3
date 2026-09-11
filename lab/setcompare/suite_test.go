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
	"os"
	"testing"
	"time"
)

// TestWorkloadsAgree verifies that for every workload in the suite, the Set3
// side and the native map side compute the same answer — not merely a
// similar-looking amount of work.
//
// This is the guard that stands in front of every number this package
// produces. A benchmark whose two candidates quietly diverge (a key generator
// that collided, a query stream that turned out to be all misses, a sliding
// window that fell out of step) does not fail: it reports a confident,
// meaningless speedup. Every workload builder therefore ships a Verify closure
// that compares the two containers element by element, and this test runs all
// of them at a size small enough to finish in a second.
func TestWorkloadsAgree(t *testing.T) {
	const size = 4096
	for _, info := range Workloads {
		for _, keyType := range info.keyTypes {
			t.Run(info.name+"/"+keyType, func(t *testing.T) {
				w, err := newWorkload(info.name, keyType, size)
				if err != nil {
					t.Fatalf("building the workload failed: %v", err)
				}
				defer w.Release()

				if w.Set3Batch == nil || w.MapBatch == nil {
					t.Fatal("a candidate has no batch function")
				}
				if w.ItemsPerOp <= 0 {
					t.Fatalf("items per operation is %v, so the reported unit would be meaningless", w.ItemsPerOp)
				}
				if err := w.Verify(); err != nil {
					t.Fatalf("the two candidates do not agree: %v", err)
				}
			})
		}
	}
}

// TestBatchesHonourTheirLoopCount verifies that a batch of n operations really
// performs n operations, which is the one assumption rtcompare's whole
// measurement rests on.
//
// If a batch ignores its n, calibration cannot make batches longer and fails
// with "could not reach target batch duration" — or worse, succeeds against a
// constant and reports the time it takes to do nothing. The check is a coarse
// one on purpose: batches at two very different sizes, asserting only that the
// larger one takes materially longer, because anything tighter would be a
// timing assertion in a unit test.
func TestBatchesHonourTheirLoopCount(t *testing.T) {
	for _, info := range Workloads {
		t.Run(info.name, func(t *testing.T) {
			w, err := newWorkload(info.name, info.keyTypes[0], 1024)
			if err != nil {
				t.Fatalf("building the workload failed: %v", err)
			}
			defer w.Release()

			w.Set3Batch(1) // warm
			small := timeBatch(w.Set3Batch, 1)
			large := timeBatch(w.Set3Batch, 64)
			if large <= small {
				t.Fatalf("64 operations took %v and one took %v, so the batch is not looping over n", large, small)
			}
		})
	}
}

func timeBatch(b func(uint64), n uint64) time.Duration {
	start := time.Now()
	b(n)
	return time.Since(start)
}

// TestHashRouteIsWhatTheSuiteClaims verifies that the four key types really do
// take the four different routes through the hashing package that the suite
// says they do.
//
// It matters for reading the results, not for correctness: a row where Set3
// lost on a struct key means something quite different depending on whether it
// hashed the struct as one raw byte block or handed it to hash/maphash. The
// CSV records that per row, and this test is what keeps the recorded value
// honest if the hashing package's eligibility rules ever change.
func TestHashRouteIsWhatTheSuiteClaims(t *testing.T) {
	if !rawBlockEligible[uint64]() {
		t.Error("uint64 should be hashed as a raw byte block")
	}
	if !rawBlockEligible[tenantKey]() {
		t.Error("a struct of three uint64 fields should be hashed as a raw byte block")
	}
	if rawBlockEligible[eventKey]() {
		t.Error("a struct holding a string cannot be hashed as a raw byte block")
	}
	if rawBlockEligible[string]() {
		t.Error("a string is not a fixed-size block")
	}
}

// TestBudgetPlannerStaysInsideItsBudget verifies that the scheduler never
// promises a cell more time than the run allows, and that it degrades in the
// documented order when it cannot.
//
// The suite runs for tens of minutes across ninety-odd cells, and a planner
// that mis-sizes one expensive cell turns that into hours. The planner is
// therefore checked at three regimes: an operation far below the clock's
// resolution, one at the calibration target, and one that costs a tenth of a
// second all by itself.
func TestBudgetPlannerStaysInsideItsBudget(t *testing.T) {
	cases := []struct {
		name    string
		batchNs float64
		budget  time.Duration
		wantFit bool
	}{
		{"a two-nanosecond lookup", 60_000, 15 * time.Second, true},
		{"a one-millisecond pass", 1e6, 15 * time.Second, true},
		{"a hundred-millisecond pass", 1e8, 15 * time.Second, true},
		{"a one-second pass", 1e9, 15 * time.Second, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			step, cost, fits := planBudget(tc.batchNs, tc.budget, 5000)
			if fits != tc.wantFit {
				t.Fatalf("fits=%v, want %v (cost %v at %d runs / %d repeats)", fits, tc.wantFit, cost, step.validationRuns, step.repeats)
			}
			if fits && cost > tc.budget {
				t.Fatalf("planner claimed a fit at %v, over the %v budget", cost, tc.budget)
			}
			if !fits && (step != budgetLadder[len(budgetLadder)-1]) {
				t.Fatalf("a cell that does not fit must fall back to the cheapest rung, got %+v", step)
			}
			if step.repeats < 11 {
				t.Fatalf("%d repeats is below rtcompare's minimum", step.repeats)
			}
		})
	}
}

// TestCompareSuite is the suite itself: it measures Set3 against
// map[T]struct{} across every workload, key type and size in the
// configuration, and writes the CSV files the README charts are drawn from.
//
// This is the entry point a maintainer runs on a quiet machine before updating
// the performance section:
//
//	go test -tags set3lab -run TestCompareSuite -timeout 180m -v ./lab/setcompare
//
// It takes 20 to 40 minutes at the defaults and rather longer with
// SET3_CMP_HUGE=1. Under -short it runs a two-size, one-key-type schedule into
// a temporary directory, which proves the whole pipeline still works without
// pretending the numbers mean anything; that is what the weekly lab job runs.
//
// The test asserts almost nothing about the measurements themselves, which is
// deliberate — a performance number is not a pass/fail property of the code,
// and a threshold here would either be so loose it never fires or so tight it
// fires on a busy machine. What it does assert is that the run produced rows,
// that they were written, and that the suite did not silently skip everything.
func TestCompareSuite(t *testing.T) {
	if testing.CoverMode() != "" {
		t.Skip("coverage instrumentation makes timing comparisons meaningless; run without -cover")
	}

	cfg, err := LoadConfig(testing.Short())
	if err != nil {
		t.Fatalf("configuration: %v", err)
	}
	if testing.Short() && os.Getenv("SET3_CMP_OUT") == "" {
		cfg.OutDir = t.TempDir()
	}
	t.Logf("configuration: %s", cfg)
	t.Logf("output: %s", cfg.OutDir)

	started := time.Now()
	var runtimeRows []RuntimeResult
	var memoryRows []MemoryResult

	if !cfg.SkipRuntime {
		t.Log("=== runtime pass ===")
		runtimeRows = MeasureRuntime(cfg, func(format string, args ...any) { t.Logf(format, args...) })
	}
	if !cfg.SkipMemory {
		t.Log("=== memory pass ===")
		memoryRows = MeasureMemory(cfg, func(format string, args ...any) { t.Logf(format, args...) })
	}

	elapsed := time.Since(started)
	written, err := WriteResults(cfg, runtimeRows, memoryRows, elapsed)
	if err != nil {
		t.Fatalf("writing results: %v", err)
	}

	t.Logf("=== summary after %s ===", elapsed.Round(time.Second))
	if len(runtimeRows) > 0 {
		t.Log("\n" + Summarize(runtimeRows))
	}
	if len(memoryRows) > 0 {
		t.Log("\n" + SummarizeMemory(memoryRows))
	}
	for _, path := range written {
		t.Logf("wrote %s", path)
	}
	t.Logf("batch sink (ignore, it exists so the compiler cannot fold a batch away): %d", SinkValue())

	if !cfg.SkipRuntime {
		assertSomethingWasMeasured(t, runtimeRows)
	}
	if !cfg.SkipMemory && len(memoryRows) == 0 {
		t.Error("the memory pass produced no rows")
	}
}

// assertSomethingWasMeasured fails the run when every cell was skipped, which
// is the one outcome that means the suite is broken rather than the machine
// merely being noisy.
func assertSomethingWasMeasured(t *testing.T, rows []RuntimeResult) {
	t.Helper()
	if len(rows) == 0 {
		t.Fatal("the runtime pass produced no rows at all")
	}
	measured := 0
	for _, r := range rows {
		if !r.Skipped {
			measured++
		}
	}
	if measured == 0 {
		t.Fatalf("all %d cells were skipped; check SET3_CMP_BUDGET and SET3_CMP_HARDCAP", len(rows))
	}
	t.Logf("%d of %d cells were measured", measured, len(rows))
}

// TestMemoryMeasurementIsCalibrated verifies that the memory pass reports the
// footprint of a container it does not know, by first reporting the footprint
// of one whose size is arithmetic.
//
// The memory numbers this suite produces are the surprising ones — a native
// map costs several times what a naive slot count suggests — and a surprising
// number is exactly the kind that has to be defended before it is published.
// A slice of n uint64 occupies 8n bytes and nothing else, so measuring one
// through the same code path the containers go through turns "the ratio looks
// wrong" into a question with an answer.
//
// The tolerance is one percent, which covers the allocator's size classes; a
// failure here means measureOnce is counting something it should not, not that
// the machine was busy.
func TestMemoryMeasurementIsCalibrated(t *testing.T) {
	const n = 1 << 20
	retained, _, _ := measureOnce(func() any { return make([]uint64, n) })

	perElement := retained / n
	if perElement < 7.92 || perElement > 8.08 {
		t.Fatalf("a slice of %d uint64 measured %.4f bytes per element; it is 8 by construction, "+
			"so measureOnce is counting something else", n, perElement)
	}
	t.Logf("calibration: []uint64 measures %.4f bytes per element", perElement)
}

// TestNativeMapFootprintIsWhatWeMeasure records, as an executable note, the
// per-element cost of the container Set3 is compared against.
//
// It is here because the memory result changed drastically with Go's rewrite
// of the map to a Swiss table: on the bucket map that the README's original
// numbers were taken on, map[uint64]struct{} cost about 12 bytes per element,
// and Set3's advantage was the 25% the README still quotes. On the current
// runtime it costs three times that. That is a fact about Go, not about Set3,
// and it deserves to fail loudly when it changes again rather than to be
// discovered in a chart.
//
// The bounds are wide on purpose: this asserts the order of magnitude, so that
// a future Go release which halves or doubles the figure gets noticed.
func TestNativeMapFootprintIsWhatWeMeasure(t *testing.T) {
	const n = 1 << 20
	keys := buildKeys(makeUint64Key, memberDomain, n)

	mapBytes, _, _ := measureOnce(func() any {
		m := make(map[uint64]struct{}, n)
		for _, k := range keys {
			m[k] = struct{}{}
		}
		return m
	}, keys)
	set3Bytes, _, _ := measureOnce(func() any {
		s, _ := buildSet3Shape(shapePresized, keys, n)
		return s
	}, keys)

	mapPer := mapBytes / n
	set3Per := set3Bytes / n
	t.Logf("at %d uint64 elements: Set3 %.2f B/element, map[uint64]struct{} %.2f B/element, ratio %.3f",
		n, set3Per, mapPer, set3Per/mapPer)

	if set3Per < 9 || set3Per > 14 {
		t.Errorf("Set3 measured %.2f bytes per element; expected 9 to 14 for an 8-byte key plus one control byte", set3Per)
	}
	if mapPer < 15 || mapPer > 60 {
		t.Errorf("map[uint64]struct{} measured %.2f bytes per element, outside the 15 to 60 this was last seen in; "+
			"the runtime's map representation has changed and the README's memory claim needs revisiting", mapPer)
	}
}
