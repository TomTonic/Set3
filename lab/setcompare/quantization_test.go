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
	"testing"
	"time"

	"github.com/TomTonic/rtcompare"
)

// quantizationBatches are the batch lengths this test compares, as durations
// rather than as rtcompare's relative bound, for the reason given on
// targetBatchDuration.
//
// The first is what rtcompare's default MaxQuantizationError produces on a
// 30 ns clock; the second is ten times that; the third is what the suite uses;
// the fourth is twice the suite's, and is the one the assertion is about. The
// two short ones are measured so the test can *show* what they would have said,
// which is the evidence for the setting rather than an assertion about it.
var quantizationBatches = []time.Duration{
	30 * time.Microsecond,
	300 * time.Microsecond,
	targetBatchDuration,
	2 * targetBatchDuration,
}

// quantizationProbe is one cell to re-measure at every target.
type quantizationProbe struct {
	scenario string
	keyType  string
	size     int
	why      string
}

// quantizationProbes are chosen for sensitivity, not for coverage.
//
// Quantization can only distort an answer when the granularity of a single
// measurement is an appreciable share of the difference being asked about.
// That makes two things dangerous: a cheap operation, and a small difference.
// These four cells are the extremes of both that the suite actually contains.
var quantizationProbes = []quantizationProbe{
	{scLookupHit95, keyTypeStruct, SizeL2,
		"the smallest difference the suite resolved at all, -1.5%, on a 13 ns operation"},
	{scLookupHit30, keyTypeStruct, SizeRAM,
		"50 MB of Set3 against 160 MB of native map, the largest working set the suite measures, where a short batch samples cache state instead of averaging over it"},
	{scIterate, keyTypeUint64, SizeL1,
		"the cheapest operation in the suite at 1.55 ns per element, and an allocating workload, so it runs at the default target"},
	{scLookupHit30, keyTypeUint64, SizeL1,
		"a cheap steady-state operation, 4 ns, and the cell that first exposed the garbage collection bias described in collectOptionsFor"},
	{scBuildPresized, keyTypeUint64, SizeL1,
		"the cheapest allocating workload, 7 us per operation, which keeps the collection between batches and so is the one that could still drift"},
}

// quantizationOutcome is one (cell, batch length) measurement.
type quantizationOutcome struct {
	batch      time.Duration
	target     float64
	delta      float64
	low, high  float64
	noiseFloor float64
	tieRate    float64
	innerLoops uint64
	batchMicro float64
	resolved   bool
}

// TestQuantizationTargetDoesNotChangeTheAnswer verifies that the suite's
// answers do not depend on how long its batches happen to be.
//
// This is the measurement's own soundness check, and it exists because the
// sister suite in lab/hashperf found the opposite. Comparing two hash functions
// that each cost about a nanosecond, rtcompare's default batch length left so
// many bootstrap replicates tied that the confidence near zero was inflated,
// and the target had to be tightened a hundredfold before the answer held
// still. If that is a property of the default rather than of that particular
// pair of candidates, every number this package produces is suspect.
//
// So the question is settled by measurement rather than by argument: take the
// cells where quantization would bite first — the cheapest operations and the
// smallest differences — and measure each of them at the default target, at
// the one this suite uses, and at the hundredfold-tighter one hashperf needed.
// If the batch length mattered, the three answers would disagree.
//
// The comparison allows for the fact that a confidence interval covers
// sampling uncertainty only: two intervals are required to overlap once each
// has been widened by its own noise floor, which is the part of the
// disagreement that the machine, not the setting, is responsible for.
func TestQuantizationTargetDoesNotChangeTheAnswer(t *testing.T) {
	if testing.CoverMode() != "" {
		t.Skip("coverage instrumentation makes timing comparisons meaningless; run without -cover")
	}

	batches := quantizationBatches
	validationRuns, repeats := 8, 101
	if testing.Short() {
		// Doubling the suite's batch length is the expensive part, and under
		// -short the point is only that the machinery runs.
		batches = quantizationBatches[1:3]
		validationRuns, repeats = 4, 25
	}

	for _, probe := range quantizationProbes {
		t.Run(probe.scenario+"/"+probe.keyType+"/"+shortSize(probe.size), func(t *testing.T) {
			t.Logf("why this cell: %s", probe.why)
			w, err := newWorkload(probe.scenario, probe.keyType, probe.size)
			if err != nil {
				t.Fatalf("building the workload failed: %v", err)
			}
			defer w.Release()

			outcomes := make([]quantizationOutcome, 0, len(batches))
			for _, batch := range batches {
				outcomes = append(outcomes, measureAtBatchLength(t, w, batch, validationRuns, repeats))
			}
			reportQuantization(t, outcomes)
			assertConverged(t, outcomes)
		})
	}
}

// assertConverged checks that the suite's batch length has reached the answer:
// that doubling it does not move the result.
//
// Only the last two outcomes are asserted on. The shorter ones are expected to
// disagree — that disagreement is why targetBatchDuration is what it is — and
// they are reported rather than tested, with a log line whenever one of them
// would have told a different story. Asserting on them would be asserting that
// a setting the suite does not use is wrong, which is not a property worth
// defending against machine noise.
func assertConverged(t *testing.T, outcomes []quantizationOutcome) {
	t.Helper()
	if len(outcomes) < 2 {
		return
	}
	suite := outcomes[len(outcomes)-2]
	longer := outcomes[len(outcomes)-1]

	if !intervalsOverlap(suite, longer) {
		t.Errorf("the answer is still moving at the suite's batch length: %v gives %.2f%% [%+.2f,%+.2f] "+
			"(floor %.2f%%) and %v gives %.2f%% [%+.2f,%+.2f] (floor %.2f%%). "+
			"Raise targetBatchDuration until doubling it stops changing the answer",
			suite.batch, suite.delta, suite.low, suite.high, suite.noiseFloor,
			longer.batch, longer.delta, longer.low, longer.high, longer.noiseFloor)
	}
	if suite.resolved && longer.resolved && signOf(suite.delta) != signOf(longer.delta) {
		t.Errorf("the suite's batch length resolves in the opposite direction to a longer one: %+.2f%% against %+.2f%%",
			suite.delta, longer.delta)
	}

	for _, short := range outcomes[:len(outcomes)-2] {
		if !intervalsOverlap(short, longer) {
			t.Logf("note: a batch of %v would have reported %.2f%% [%+.2f,%+.2f] against the converged %.2f%% "+
				"[%+.2f,%+.2f] — this is the effect targetBatchDuration exists to avoid, not a failure",
				short.batch, short.delta, short.low, short.high, longer.delta, longer.low, longer.high)
		}
	}
}

// measureAtBatchLength runs one full rtcompare comparison at one batch length,
// keeping everything else exactly as the suite would have it.
func measureAtBatchLength(t *testing.T, w *Workload, batch time.Duration, validationRuns, repeats int) quantizationOutcome {
	t.Helper()
	target := quantizationTargetFor(batch)
	opt := collectOptionsFor(w)
	opt.MaxQuantizationError = target
	opt.Repeats = repeats

	a, b := w.candidates()
	report, err := rtcompare.Compare(a, b, rtcompare.CompareOptions{
		Collect:        opt,
		ValidationRuns: validationRuns,
		Resamples:      5_000,
	})
	if err != nil {
		t.Fatalf("comparison at target %g failed: %v", target, err)
	}

	inner := report.ValidationA.InnerLoops
	return quantizationOutcome{
		batch:      batch,
		target:     target,
		delta:      report.Estimate.Delta * 100,
		low:        report.Estimate.Low * 100,
		high:       report.Estimate.High * 100,
		noiseFloor: report.NoiseFloor * 100,
		tieRate:    math.Max(report.ValidationA.TieRate, report.ValidationB.TieRate),
		innerLoops: inner,
		batchMicro: float64(inner) * report.NsPerOpA / 1000,
		resolved:   report.Resolved,
	}
}

// reportQuantization prints the comparison as a table, because the table is
// the actual product of this test: an assertion that passes tells you nothing
// about how much headroom it passed by.
func reportQuantization(t *testing.T, outcomes []quantizationOutcome) {
	t.Helper()
	t.Logf("%-12s %-10s %9s %10s %22s %9s %8s %9s", "asked for", "qerr", "inner", "actual", "difference", "floor", "ties", "resolved")
	for _, o := range outcomes {
		t.Logf("%-12v %-10.2g %9d %8.1fus %8.2f%% [%+.2f,%+.2f] %8.2f%% %7.2f%% %9v",
			o.batch, o.target, o.innerLoops, o.batchMicro, o.delta, o.low, o.high, o.noiseFloor, o.tieRate*100, o.resolved)
	}
}

// intervalsOverlap reports whether two estimates are compatible once each
// interval is widened by the noise floor its run measured. The interval itself
// covers only the spread of the samples; the floor is the part the machine
// contributed, and comparing two separate runs has to allow for both.
func intervalsOverlap(a, b quantizationOutcome) bool {
	aLow, aHigh := a.low-a.noiseFloor, a.high+a.noiseFloor
	bLow, bHigh := b.low-b.noiseFloor, b.high+b.noiseFloor
	return aLow <= bHigh && bLow <= aHigh
}

func signOf(v float64) int {
	if v < 0 {
		return -1
	}
	return 1
}

// shortSize names a size for a subtest, so the test output reads as the cells
// the suite reports rather than as raw element counts.
func shortSize(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%dM", n>>20)
	case n >= 1024:
		return fmt.Sprintf("%dk", n>>10)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// TestSuiteTieRatesStayLow verifies that the batch lengths the suite chooses
// leave the bootstrap with few tied replicates.
//
// A tie is two resampled medians coming out exactly equal, which happens when
// the measurement is coarse relative to the question. Ties count towards the
// confidence at a threshold of zero, so a high rate inflates exactly the figure
// that decides whether a small difference is real. rtcompare's own notes record
// about 15% at the default target for an ordinary candidate; this asserts that
// the suite's workloads, at the targets collectOptionsFor picks for them, stay
// an order of magnitude below that.
//
// It runs the two cheapest operations in the suite, which are where ties would
// appear first.
func TestSuiteTieRatesStayLow(t *testing.T) {
	if testing.CoverMode() != "" {
		t.Skip("coverage instrumentation makes timing comparisons meaningless; run without -cover")
	}
	const maxAcceptableTieRate = 0.05

	for _, probe := range []quantizationProbe{
		{scIterate, keyTypeUint64, SizeL1, ""},
		{scLookupHit30, keyTypeUint64, SizeL1, ""},
	} {
		t.Run(probe.scenario+"/"+probe.keyType, func(t *testing.T) {
			w, err := newWorkload(probe.scenario, probe.keyType, probe.size)
			if err != nil {
				t.Fatalf("building the workload failed: %v", err)
			}
			defer w.Release()

			out := measureAtBatchLength(t, w, targetBatchDuration, 10, 51)
			t.Logf("at the suite's batch length of %v: actual %.1fus, tie rate %.2f%%, difference %.2f%% [%+.2f,%+.2f]",
				targetBatchDuration, out.batchMicro, out.tieRate*100, out.delta, out.low, out.high)

			if out.tieRate > maxAcceptableTieRate {
				t.Errorf("tie rate %.2f%% exceeds %.0f%%; the measurement is too coarse for the question. "+
					"Raise targetBatchDuration", out.tieRate*100, maxAcceptableTieRate*100)
			}
		})
	}
}
