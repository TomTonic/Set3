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
	"runtime"
	"strings"
	"time"

	"github.com/TomTonic/rtcompare"
)

// RuntimeResult is one measured cell: one workload, one key type, one size,
// with the evidence rtcompare produced for it.
//
// Positive DeltaPct means Set3 is faster. A row with Resolved false has not
// established anything — the difference is inside what this machine invents
// between two runs of identical code, or the interval covers zero — and the
// chart tool draws it greyed out rather than dropping it, because "too close
// to call" is a result.
type RuntimeResult struct {
	Scenario     string
	KeyType      string
	Size         int
	Unit         string
	ItemsPerOp   float64
	RawBlockHash bool

	// Set3LoadFactor and MapLoadFactor are the occupancies the two containers
	// were measured at, or zero where the workload has no persistent container
	// to weigh. Read a speed row against them: two containers at different
	// occupancies differ in their space/time trade as well as in their layout.
	Set3LoadFactor float64
	MapLoadFactor  float64

	Set3NsPerItem float64
	MapNsPerItem  float64

	DeltaPct      float64
	CILowPct      float64
	CIHighPct     float64
	Level         float64
	NoiseFloorPct float64
	Resolved      bool

	Repeats         int
	ValidationRuns  int
	InnerLoops      uint64
	BlockLength     int
	Autocorrelation float64
	TieRate         float64
	DriftPSet3      float64
	DriftPMap       float64

	Set3AllocBytesPerItem float64
	MapAllocBytesPerItem  float64
	Set3MallocsPerItem    float64
	MapMallocsPerItem     float64

	Skipped bool
	Note    string
	Seconds float64
}

// budgetStep is one admissible (validation runs, repeats) pair. The ladder is
// ordered from the rtcompare defaults down to the cheapest setting that still
// produces a defensible number, and the planner takes the first rung that fits
// the cell's time budget.
type budgetStep struct {
	validationRuns int
	repeats        int
}

// budgetLadder trades away validation before it trades away repeats, and never
// goes below eleven repeats because that is rtcompare's minimum for a
// confidence at all.
//
// The order encodes which precision is worth more. Repeats set the width of the
// confidence interval — the answer to "how big is the difference". Validation
// runs set how well the noise floor is known — the answer to "is the difference
// bigger than what this machine makes up". Below about ten validation runs the
// *rates* in the validation report stop meaning anything, but the noise floor
// itself, being a quantile of a per-run difference, is usable from far fewer.
// So validation is cut first, and the row records how far it was cut.
var budgetLadder = []budgetStep{
	{40, 101}, {24, 101}, {16, 101}, {10, 101},
	{10, 51}, {6, 51}, {4, 51},
	{4, 25}, {2, 25}, {2, 11},
}

// runtimeRunner measures one cell at a time and accumulates the rows.
type runtimeRunner struct {
	cfg      Config
	logf     func(format string, args ...any)
	results  []RuntimeResult
	started  time.Time
	cellsRun int
}

// MeasureRuntime runs the whole runtime pass and returns one row per cell.
//
// It builds each workload, measures it, and releases it before moving on, so
// that peak memory is one cell rather than the whole schedule. Cells that would
// exceed Config.HardCap even at the cheapest setting are recorded as skipped
// with their estimated cost; a cell whose measurement fails is recorded with
// the error in its Note. Neither aborts the run, because a 30-minute suite that
// throws away 29 minutes of good rows over one bad cell is not useful.
//
// logf receives one line per cell as it completes.
func MeasureRuntime(cfg Config, logf func(format string, args ...any)) []RuntimeResult {
	r := &runtimeRunner{cfg: cfg, logf: logf, started: time.Now()}
	for _, info := range Workloads {
		if !cfg.wantsScenario(info.name) {
			continue
		}
		for _, keyType := range cfg.KeyTypes {
			for _, size := range cfg.Sizes {
				if !info.appliesTo(keyType, size) {
					continue
				}
				r.measureCell(info, keyType, size)
			}
		}
	}
	return r.results
}

// measureCell builds, measures and releases one workload.
func (r *runtimeRunner) measureCell(info scenarioInfo, keyType string, size int) {
	started := time.Now()
	w, err := newWorkload(info.name, keyType, size)
	if err != nil {
		r.record(RuntimeResult{Scenario: info.name, KeyType: keyType, Size: size, Skipped: true, Note: err.Error()})
		return
	}
	defer func() {
		w.Release()
		// The next cell allocates its own key material before it measures
		// anything. Handing it a heap still full of this one's containers
		// would charge this cell's garbage to that cell's first batches.
		runtime.GC()
	}()

	res := r.compareWorkload(w)
	res.Seconds = time.Since(started).Seconds()
	r.record(res)
}

// record appends a row and logs it.
func (r *runtimeRunner) record(res RuntimeResult) {
	r.results = append(r.results, res)
	r.cellsRun++
	switch {
	case res.Skipped:
		r.logf("[%6.1fs] %-15s %-11s n=%-9d SKIPPED: %s",
			time.Since(r.started).Seconds(), res.Scenario, res.KeyType, res.Size, res.Note)
	default:
		verdict := "unresolved"
		if res.Resolved {
			verdict = "resolved"
		}
		r.logf("[%6.1fs] %-15s %-11s n=%-9d Set3 %8.2f vs map %8.2f %-18s  %+6.1f%% [%+.1f%%,%+.1f%%] floor %.2f%% %s%s",
			time.Since(r.started).Seconds(), res.Scenario, res.KeyType, res.Size,
			res.Set3NsPerItem, res.MapNsPerItem, res.Unit,
			res.DeltaPct, res.CILowPct, res.CIHighPct, res.NoiseFloorPct, verdict, noteSuffix(res.Note))
	}
}

func noteSuffix(note string) string {
	if note == "" {
		return ""
	}
	return " | " + note
}

// compareWorkload runs the rtcompare protocol on one workload.
func (r *runtimeRunner) compareWorkload(w *Workload) RuntimeResult {
	res := RuntimeResult{
		Scenario:       w.Scenario,
		KeyType:        w.KeyType,
		Size:           w.Size,
		Unit:           w.Unit,
		ItemsPerOp:     w.ItemsPerOp,
		RawBlockHash:   w.RawBlockHash,
		Set3LoadFactor: w.Set3LoadFactor,
		MapLoadFactor:  w.MapLoadFactor,
		Level:          r.cfg.Level,
	}

	collect := collectOptionsFor(w)
	pilotNs := pilotOpNs(w)
	batchNs := estimateBatchNs(collect, pilotNs)
	step, estimate, fits := planBudget(batchNs, r.cfg.Budget, r.cfg.Resamples)
	if !fits && estimate > r.cfg.HardCap {
		res.Skipped = true
		res.Note = fmt.Sprintf("estimated %.0fs at the cheapest setting, over the %v hard cap", estimate.Seconds(), r.cfg.HardCap)
		return res
	}
	collect.Repeats = step.repeats
	res.Repeats = step.repeats
	res.ValidationRuns = step.validationRuns

	measureAllocations(w, &res, allocProbeOps(pilotNs))
	auditAllocationFlag(w, &res)

	set3Candidate, mapCandidate := w.candidates()
	report, err := rtcompare.Compare(set3Candidate, mapCandidate, rtcompare.CompareOptions{
		Collect:        collect,
		ValidationRuns: step.validationRuns,
		Level:          r.cfg.Level,
		Resamples:      r.cfg.Resamples,
	})
	if err != nil {
		res.Skipped = true
		res.Note = "rtcompare: " + err.Error()
		return res
	}
	fillFromReport(&res, report, w.ItemsPerOp)
	return res
}

// fillFromReport copies the parts of a Report that belong in a CSV row and
// converts per-operation costs into per-item ones.
func fillFromReport(res *RuntimeResult, report rtcompare.Report, itemsPerOp float64) {
	res.Set3NsPerItem = report.NsPerOpA / itemsPerOp
	res.MapNsPerItem = report.NsPerOpB / itemsPerOp
	res.DeltaPct = report.Estimate.Delta * 100
	res.CILowPct = report.Estimate.Low * 100
	res.CIHighPct = report.Estimate.High * 100
	res.Level = report.Estimate.Level
	res.NoiseFloorPct = report.NoiseFloor * 100
	res.Resolved = report.Resolved
	res.InnerLoops = report.ValidationA.InnerLoops
	res.BlockLength = report.BlockLength
	res.Autocorrelation = report.Autocorrelation
	res.TieRate = math.Max(report.ValidationA.TieRate, report.ValidationB.TieRate)
	res.DriftPSet3 = report.DriftA.PValue
	res.DriftPMap = report.DriftB.PValue
	// Appended, not assigned: the row may already carry a note from the
	// allocation audit, which runs before the comparison and is at least as
	// worth reading as a drift warning.
	res.Note = joinNotes(res.Note, strings.Join(report.Warnings, "; "))
}

// joinNotes concatenates two note fragments, either of which may be empty.
func joinNotes(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	default:
		return a + "; " + b
	}
}

// targetBatchDuration is how long one batch of this suite's workloads should
// run, and it is the single most consequential setting in the package.
//
// rtcompare's knob is MaxQuantizationError, a *relative* bound on what the
// clock's granularity contributes. Its default of 0.001 sizes a batch at a
// thousand clock ticks, which on a 30 ns clock is 30 microseconds. That is
// ample for the purpose it is documented for — quantization at that length
// contributes about a tenth of a percent — and it is nowhere near enough for
// this suite, for two reasons that have nothing to do with the clock:
//
//  1. A short batch amortises its own cold start over very few operations.
//     At the default target, the build-presized workload at a thousand
//     elements calibrated to *five* operations per batch. Measured across
//     batch lengths, that cell reported +30.5%, +42.2% and +42.8% at 40 us,
//     190 us and 3.8 ms, converging only at the longest.
//
//  2. A workload whose working set is tens of megabytes is measuring cache
//     and TLB state as much as it is measuring code, and a short batch
//     samples whatever state it happened to start in rather than averaging
//     over it. The lookup-hit30 cell at two million struct keys — 50 MB of
//     Set3 against 160 MB of native map — reported -5.6%, -20.9% and +10.9%
//     at those same three lengths, with noise floors of 6.7%, 3.6% and 0.45%.
//     Only the longest batch produced an answer worth having.
//
// Three milliseconds is where every cell the suite contains has stopped
// moving. Expressing it as a duration rather than as a relative error is the
// point: the clock's granularity is a property of the machine, and deriving
// the knob from the duration keeps the batch the same length on a 30 ns Linux
// clock and a 100 ns Windows one, where a fixed MaxQuantizationError would not.
//
// The cost is real and is paid deliberately. Batches are 16 to 100 times
// longer than the default would make them, so the budget planner gives cheap
// cells fewer validation runs than rtcompare's default of 40. That trades
// precision in the validation *rates*, which need many runs, for correctness
// in the measurement itself; the noise floor that Resolved actually gates on
// needs far fewer runs than those rates do.
//
// TestQuantizationTargetDoesNotChangeTheAnswer is the standing guard on this.
const targetBatchDuration = 3 * time.Millisecond

// quantizationTargetFor converts targetBatchDuration into the relative bound
// rtcompare actually takes, using the clock precision measured on this machine.
func quantizationTargetFor(batch time.Duration) float64 {
	return float64(rtcompare.GetSampleTimePrecision()) / float64(batch.Nanoseconds())
}

// collectOptionsFor picks the measurement options for a workload.
//
// # Garbage collection, and why a non-allocating workload gets none of it
//
// rtcompare offers a collection before every batch (GCBetween) so that a
// collection triggered by one candidate's garbage cannot land inside the
// other candidate's timed region. For a workload that allocates a container
// per operation that is exactly right, and those workloads get it.
//
// For a workload that allocates nothing it is actively harmful, and the suite
// learned this the hard way. A collection walks the heap and evicts the caches,
// so every batch starts cold; the cold start is amortised over however many
// operations the batch performs, which means the measured difference depends on
// the batch length. Worse, it does not bias both candidates equally: Set3's
// table is a third the size of the native map's, so it re-warms faster, and the
// bias runs in Set3's favour. Measured on lookup-hit30/uint64 at a thousand
// elements, across a hundredfold range of batch lengths:
//
//	configuration                   0.001      0.0001      0.00001
//	GCBetween + DisableGC          +28.60%     +25.03%     +23.77%
//	DisableGC only                 +24.70%     +24.78%     +24.31%
//	neither                        +24.45%     +24.43%     +24.44%
//
// The last row is the answer: flat to two hundredths of a percentage point
// across the whole range. The first row, which was the suite's original
// setting, moves by nearly five points across it — and a setting whose answer
// depends on how long you look is not measuring the code.
// A workload that allocates nothing also cannot trigger a collection inside a
// batch, so there is nothing for GCBetween to prevent — and DisableGC is not
// used on its own, per rtcompare's own advice.
func collectOptionsFor(w *Workload) rtcompare.CollectOptions {
	opt := rtcompare.CollectOptions{
		Order:                rtcompare.OrderABBA,
		MaxQuantizationError: quantizationTargetFor(targetBatchDuration),
	}
	opt.GCBetween = w.Allocating
	return opt
}

// estimateBatchNs predicts how long one batch will run, which is what the
// budget planner needs and what nothing in rtcompare will tell you before it
// has already spent the time.
//
// A batch is at least as long as calibration will make it — the clock precision
// divided by the quantization target — and at least as long as a single
// operation, since calibration cannot subdivide one. Taking the larger of the
// two covers both regimes with no branch: cheap operations land on the
// calibration target, expensive ones on their own cost.
func estimateBatchNs(opt rtcompare.CollectOptions, opNs float64) float64 {
	qerr := opt.MaxQuantizationError
	if qerr == 0 {
		qerr = rtcompare.DefaultMaxQuantizationError
	}
	return math.Max(opNs, float64(rtcompare.GetSampleTimePrecision())/qerr)
}

// allocProbeOps picks how many operations the allocation probe runs over.
//
// Probing a single operation is right for a workload that builds a container
// per operation and wrong for everything else, because a steady-state workload
// allocates rarely rather than never: a sliding window reuses tombstones until
// it cannot, and then rehashes. One operation either catches that or misses it,
// and both answers are wrong — 278 kilobytes per operation, or nothing at all.
// Probing about a millisecond's worth of operations amortises the rare event
// over the many that do not allocate, which is the number a reader wants.
func allocProbeOps(opNs float64) uint64 {
	const probeWindowNs = 1e6
	if opNs <= 0 {
		return 1
	}
	n := uint64(probeWindowNs / opNs)
	switch {
	case n < 1:
		return 1
	case n > 1<<20:
		return 1 << 20
	default:
		return n
	}
}

// pilotOpNs times a single operation of each candidate and returns the larger.
//
// Two unmeasured warm-up operations run first, so the number does not include
// page faults or a cold branch predictor. For a cheap operation the result is
// noise, which is fine: the caller takes the maximum against the calibration
// target and the noise is far below it.
func pilotOpNs(w *Workload) float64 {
	one := func(b rtcompare.Batch) float64 {
		b(1)
		b(1)
		runtime.GC()
		t0 := rtcompare.SampleTime()
		b(1)
		t1 := rtcompare.SampleTime()
		return float64(rtcompare.DiffTimeStamps(t0, t1))
	}
	return math.Max(one(w.Set3Batch), one(w.MapBatch))
}

// planBudget picks the highest rung of the ladder whose predicted cost fits the
// budget, and reports whether anything fit at all.
//
// The prediction has two terms. Measurement is (2*validationRuns+1) Collect
// calls of repeats samples, each sample being two batches. Resampling is the
// other half and is often the larger one for cheap candidates: validation
// bootstraps twice per run per candidate, and rtcompare's own notes put a
// forty-run validation at about three seconds at the default resample count.
// Both terms are approximations; they only have to be good enough to keep a
// three-hour schedule from turning into a twelve-hour one.
func planBudget(batchNs float64, budget time.Duration, resamples uint64) (budgetStep, time.Duration, bool) {
	for _, step := range budgetLadder {
		cost := predictCost(step, batchNs, resamples)
		if cost <= budget {
			return step, cost, true
		}
	}
	last := budgetLadder[len(budgetLadder)-1]
	return last, predictCost(last, batchNs, resamples), false
}

// predictCost estimates the wall-clock cost of one cell at one ladder rung.
func predictCost(step budgetStep, batchNs float64, resamples uint64) time.Duration {
	batches := float64(2*step.validationRuns+1) * float64(step.repeats) * 2
	measureSeconds := batches * batchNs / 1e9

	// Empirical: a 40-run validation at 101 repeats and 5000 resamples costs
	// about three seconds of pure resampling, and the cost is linear in all
	// three.
	const secondsPerValidationRun = 3.0 / 40.0
	bootstrapSeconds := secondsPerValidationRun *
		float64(step.validationRuns) *
		(float64(step.repeats) / 101.0) *
		(float64(resamples) / 5000.0)

	return time.Duration((measureSeconds + bootstrapSeconds) * float64(time.Second))
}

// measureAllocations records what one operation allocates, outside the timed
// comparison.
//
// It is deliberately not part of the rtcompare run. Allocation is not a noisy
// quantity that needs statistics — the same batch allocates the same bytes
// every time — and reading MemStats inside a measured region would cost more
// than the thing being measured. One operation with a collection before it and
// a MemStats pair around it is both cheaper and more accurate than any
// sampling scheme would be.
func measureAllocations(w *Workload, res *RuntimeResult, ops uint64) {
	res.Set3AllocBytesPerItem, res.Set3MallocsPerItem = allocPerItem(w.Set3Batch, w.ItemsPerOp, ops)
	res.MapAllocBytesPerItem, res.MapMallocsPerItem = allocPerItem(w.MapBatch, w.ItemsPerOp, ops)
}

// allocationFlagThreshold is the per-operation allocation above which a
// workload counts as allocating. It is one Go page: below that, a batch of any
// plausible length cannot push the heap far enough to trigger a collection
// inside itself, so there is nothing for GCBetween to keep out of the measured
// region — and everything for it to cost. See collectOptionsFor.
const allocationFlagThreshold = 8192

// auditAllocationFlag checks the workload's declared allocation behaviour
// against what it actually allocated, and records a disagreement in the row.
//
// The declaration decides the garbage collection strategy, and the wrong
// strategy silently biases the result by several percentage points in Set3's
// favour (again, see collectOptionsFor). A declaration is a claim about the
// code, and this is the measurement that checks it — which matters because the
// claim is easy to get wrong: the iterate workload was declared allocating on
// the reasoning that its iterator hands out a closure per pass, and it turned
// out the compiler stack-allocates that closure and the workload allocates
// nothing at all.
func auditAllocationFlag(w *Workload, res *RuntimeResult) {
	perOp := res.Set3AllocBytesPerItem * w.ItemsPerOp
	mapPerOp := res.MapAllocBytesPerItem * w.ItemsPerOp
	if mapPerOp > perOp {
		perOp = mapPerOp
	}
	allocates := perOp >= allocationFlagThreshold
	if allocates == w.Allocating {
		return
	}
	res.Note = joinNotes(res.Note, fmt.Sprintf(
		"workload is declared Allocating=%v but allocates %.0f bytes per operation; "+
			"the garbage collection strategy for this row is the wrong one", w.Allocating, perOp))
}

// allocPerItem runs a batch of operations and reports what it allocated, per
// logical item.
func allocPerItem(b rtcompare.Batch, itemsPerOp float64, ops uint64) (bytes, mallocs float64) {
	if itemsPerOp <= 0 || ops == 0 {
		return 0, 0
	}
	var before, after runtime.MemStats
	b(1) // warm: the first call faults in whatever the closure needs
	runtime.GC()
	runtime.ReadMemStats(&before)
	b(ops)
	runtime.ReadMemStats(&after)

	perItem := itemsPerOp * float64(ops)
	return float64(after.TotalAlloc-before.TotalAlloc) / perItem,
		float64(after.Mallocs-before.Mallocs) / perItem
}
