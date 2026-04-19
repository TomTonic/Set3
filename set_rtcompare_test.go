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

// nolint:gosec // Test file uses deterministic synthetic data and runtime metrics collection.
package set3

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"strconv"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/rtcompare"
)

var rtcompareSetSink uint64

type rtEligibleStructKey struct {
	A uint64
	B uint64
	C uint64
}

type rtNonEligibleFloatStructKey struct {
	A uint64
	F float64
}

type rtNonEligibleStringStructKey struct {
	A uint64
	S string
}

type setRtcompareConfig struct {
	repeats        uint64
	size           int
	precisionLevel uint64
	refreshEvery   uint64
	forever        bool
	maxCycles      uint64
	mode           string
}

func defaultSetRtcompareConfig() setRtcompareConfig {
	return setRtcompareConfig{
		repeats:        envUint64("SET3_RTCOMPARE_REPEATS", 257),
		size:           int(envUint64("SET3_RTCOMPARE_SIZE", 16_384)), //nolint:gosec
		precisionLevel: envUint64("SET3_RTCOMPARE_PRECISION", 10_000),
		refreshEvery:   envUint64("SET3_RTCOMPARE_REFRESH", 32),
		forever:        envBool("SET3_RTCOMPARE_FOREVER", false),
		maxCycles:      envUint64("SET3_RTCOMPARE_MAX_CYCLES", 0),
		mode:           envString("SET3_RTCOMPARE_MODE", "full"),
	}
}

func envUint64(key string, def uint64) uint64 {
	raw := os.Getenv(key)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		return def
	}
	return v
}

func envBool(key string, def bool) bool {
	raw := os.Getenv(key)
	if raw == "" {
		return def
	}
	v, err := strconv.ParseBool(raw)
	if err != nil {
		return def
	}
	return v
}

func envString(key, def string) string {
	raw := os.Getenv(key)
	if raw == "" {
		return def
	}
	return raw
}

// skipSetRtcompareIfCoverageEnabled avoids unreliable microbenchmark-like
// comparisons under coverage instrumentation.
func skipSetRtcompareIfCoverageEnabled(t *testing.T) {
	t.Helper()
	if testing.CoverMode() != "" {
		t.Skip("skipping set rtcompare perf test under coverage instrumentation; run without -cover")
	}
}

func measureKernelSample(run func() uint64, totalOps uint64) (nsPerOp, allocBytesPerOp, mallocsPerOp float64, checksum uint64) {
	var before runtime.MemStats
	var after runtime.MemStats
	runtime.ReadMemStats(&before)
	t0 := rtcompare.SampleTime()
	checksum = run()
	t1 := rtcompare.SampleTime()
	runtime.ReadMemStats(&after)
	diff := rtcompare.DiffTimeStamps(t0, t1)
	if diff <= 0 || totalOps == 0 {
		return 0, 0, 0, checksum
	}
	nsPerOp = float64(diff) / float64(totalOps)
	allocBytesPerOp = float64(after.TotalAlloc-before.TotalAlloc) / float64(totalOps)
	mallocsPerOp = float64(after.Mallocs-before.Mallocs) / float64(totalOps)
	return nsPerOp, allocBytesPerOp, mallocsPerOp, checksum
}

func strictlyPositiveMetric(v float64) float64 {
	if v > 0 {
		return v
	}
	return math.SmallestNonzeroFloat64
}

func safeRatio(numerator, denominator float64) float64 {
	if denominator == 0 {
		if numerator == 0 {
			return 1.0
		}
		return math.Inf(1)
	}
	return numerator / denominator
}

func runSet3Workload[T comparable](keys, hits, misses []T) uint64 {
	s := EmptyWithCapacity[T](uint32(len(keys))) //nolint:gosec
	for _, k := range keys {
		s.Add(k)
	}

	var hitCount uint64
	for _, k := range hits {
		if s.Contains(k) {
			hitCount++
		}
	}

	var missCount uint64
	for _, k := range misses {
		if s.Contains(k) {
			missCount++
		}
	}

	for i := 0; i < len(keys); i += 2 {
		s.Remove(keys[i])
	}
	for i := 0; i < len(keys); i += 2 {
		s.Add(keys[i])
	}

	if hitCount != uint64(len(hits)) || missCount != 0 {
		panic("set3 workload invariant violated")
	}
	return hitCount ^ (missCount << 32) ^ uint64(s.Size())
}

func runNativeMapWorkload[T comparable](keys, hits, misses []T) uint64 {
	m := make(map[T]struct{}, len(keys))
	for _, k := range keys {
		m[k] = struct{}{}
	}

	var hitCount uint64
	for _, k := range hits {
		if _, ok := m[k]; ok {
			hitCount++
		}
	}

	var missCount uint64
	for _, k := range misses {
		if _, ok := m[k]; ok {
			missCount++
		}
	}

	for i := 0; i < len(keys); i += 2 {
		delete(m, keys[i])
	}
	for i := 0; i < len(keys); i += 2 {
		m[keys[i]] = struct{}{}
	}

	if hitCount != uint64(len(hits)) || missCount != 0 {
		panic("native map workload invariant violated")
	}
	return hitCount ^ (missCount << 32) ^ uint64(len(m))
}

func runSet3BuildOnlyWorkload[T comparable](keys []T) uint64 {
	s := EmptyWithCapacity[T](uint32(len(keys))) //nolint:gosec
	for _, k := range keys {
		s.Add(k)
	}
	if s.Size() != uint32(len(keys)) { //nolint:gosec
		panic("set3 build-only workload invariant violated")
	}
	return uint64(s.Size())
}

func runNativeMapBuildOnlyWorkload[T comparable](keys []T) uint64 {
	m := make(map[T]struct{}, len(keys))
	for _, k := range keys {
		m[k] = struct{}{}
	}
	if len(m) != len(keys) {
		panic("native map build-only workload invariant violated")
	}
	return uint64(len(m))
}

func runSet3SteadyWorkload[T comparable](s *Set3[T], keys, hits, misses []T) uint64 {
	var hitCount uint64
	for _, k := range hits {
		if s.Contains(k) {
			hitCount++
		}
	}

	var missCount uint64
	for _, k := range misses {
		if s.Contains(k) {
			missCount++
		}
	}

	for i := 0; i < len(keys); i += 2 {
		s.Remove(keys[i])
	}
	for i := 0; i < len(keys); i += 2 {
		s.Add(keys[i])
	}

	if hitCount != uint64(len(hits)) || missCount != 0 || s.Size() != uint32(len(keys)) { //nolint:gosec
		panic("set3 steady workload invariant violated")
	}
	return hitCount ^ (missCount << 32) ^ uint64(s.Size())
}

func runNativeMapSteadyWorkload[T comparable](m map[T]struct{}, keys, hits, misses []T) uint64 {
	var hitCount uint64
	for _, k := range hits {
		if _, ok := m[k]; ok {
			hitCount++
		}
	}

	var missCount uint64
	for _, k := range misses {
		if _, ok := m[k]; ok {
			missCount++
		}
	}

	for i := 0; i < len(keys); i += 2 {
		delete(m, keys[i])
	}
	for i := 0; i < len(keys); i += 2 {
		m[keys[i]] = struct{}{}
	}

	if hitCount != uint64(len(hits)) || missCount != 0 || len(m) != len(keys) {
		panic("native map steady workload invariant violated")
	}
	return hitCount ^ (missCount << 32) ^ uint64(len(m))
}

func reportPairedMetricABBA(
	t *testing.T,
	metricLabel, unit, labelA, labelB string,
	repeats, precisionLevel uint64,
	speedups []float64,
	samplesA, samplesB []float64,
) {
	t.Helper()
	if len(samplesA) == 0 || len(samplesA) != len(samplesB) {
		t.Fatalf("invalid sample lengths for %s: A=%d B=%d", metricLabel, len(samplesA), len(samplesB))
	}
	if len(samplesA) < 11 {
		meanA, _, stdA := rtcompare.Statistics(samplesA)
		meanB, _, stdB := rtcompare.Statistics(samplesB)
		medianA := rtcompare.Median(samplesA)
		medianB := rtcompare.Median(samplesB)
		fmt.Printf("\nRTCOMPARE %s summary (%s vs %s)\n", metricLabel, labelA, labelB)
		fmt.Printf("  repeats=%d (below rtcompare minimum=11 for confidence computation)\n", repeats)
		fmt.Printf("  mean   %s: %s=%.6f (sd=%.6f), %s=%.6f (sd=%.6f)\n", unit, labelA, meanA, stdA, labelB, meanB, stdB)
		fmt.Printf("  median %s: %s=%.6f, %s=%.6f\n", unit, labelA, medianA, labelB, medianB)
		return
	}

	pairedLogRatioBoverA := make([]float64, 0, len(samplesA))
	ratioAoverB := make([]float64, 0, len(samplesA))
	ratioBoverA := make([]float64, 0, len(samplesA))
	baselineOnes := make([]float64, 0, len(samplesA))

	for i := range samplesA {
		a, b := samplesA[i], samplesB[i]
		if a <= 0 || b <= 0 {
			t.Fatalf("invalid %s sample at pair %d: %s=%g %s, %s=%g %s", metricLabel, i, labelA, a, unit, labelB, b, unit)
		}
		lr := math.Log(b) - math.Log(a)
		pairedLogRatioBoverA = append(pairedLogRatioBoverA, lr)
		ratioAoverB = append(ratioAoverB, math.Exp(-lr))
		ratioBoverA = append(ratioBoverA, math.Exp(lr))
		baselineOnes = append(baselineOnes, 1.0)
	}

	resultsAFaster, err := rtcompare.CompareSamples(ratioAoverB, baselineOnes, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples (%s better in %s) failed: %v", labelA, metricLabel, err)
	}
	resultsBFaster, err := rtcompare.CompareSamples(ratioBoverA, baselineOnes, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples (%s better in %s) failed: %v", labelB, metricLabel, err)
	}

	meanA, _, stdA := rtcompare.Statistics(samplesA)
	meanB, _, stdB := rtcompare.Statistics(samplesB)
	medianA := rtcompare.Median(samplesA)
	medianB := rtcompare.Median(samplesB)
	meanLogRatio, _, _ := rtcompare.Statistics(pairedLogRatioBoverA)
	geomBoverA := math.Exp(meanLogRatio)
	geomAoverB := 1.0 / geomBoverA

	fmt.Printf("\nRTCOMPARE %s summary (%s vs %s)\n", metricLabel, labelA, labelB)
	fmt.Printf("  repeats=%d precisionLevel=%d\n", repeats, precisionLevel)
	fmt.Printf("  mean   %s: %s=%.6f (sd=%.6f), %s=%.6f (sd=%.6f)\n", unit, labelA, meanA, stdA, labelB, meanB, stdB)
	fmt.Printf("  median %s: %s=%.6f, %s=%.6f\n", unit, labelA, medianA, labelB, medianB)
	fmt.Printf("  geometric factor: %s/%s=%.9f, %s/%s=%.9f\n", labelB, labelA, geomBoverA, labelA, labelB, geomAoverB)
	fmt.Printf("  bootstrap confidence (%s):\n", metricLabel)
	fmt.Printf("    %s better than %s by at least X%%:\n", labelA, labelB)
	for _, r := range resultsAFaster {
		fmt.Printf("      X=%.2f%% -> %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
	fmt.Printf("    %s better than %s by at least X%%:\n", labelB, labelA)
	for _, r := range resultsBFaster {
		fmt.Printf("      X=%.2f%% -> %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}

func reportInterim(
	t *testing.T,
	caseName string,
	pairsDone uint64,
	timesSet3, timesMap, allocSet3, allocMap, mallocSet3, mallocMap []float64,
) {
	t.Helper()
	if len(timesSet3) == 0 || len(timesSet3) != len(timesMap) {
		return
	}
	medSet := rtcompare.Median(timesSet3)
	medMap := rtcompare.Median(timesMap)
	medAllocSet := rtcompare.Median(allocSet3)
	medAllocMap := rtcompare.Median(allocMap)
	medMallocSet := rtcompare.Median(mallocSet3)
	medMallocMap := rtcompare.Median(mallocMap)

	fmt.Printf("\n[interim %s] pairs=%d\n", caseName, pairsDone)
	fmt.Printf("  median time ns/op: set3=%.6f map=%.6f map/set3=%.6f\n", medSet, medMap, safeRatio(medMap, medSet))
	fmt.Printf("  median alloc B/op: set3=%.6f map=%.6f map/set3=%.6f\n", medAllocSet, medAllocMap, safeRatio(medAllocMap, medAllocSet))
	fmt.Printf("  median mallocs/op: set3=%.6f map=%.6f map/set3=%.6f\n", medMallocSet, medMallocMap, safeRatio(medMallocMap, medMallocSet))
}

func runSetVsMapRtcompare[T comparable](
	t *testing.T,
	cfg setRtcompareConfig,
	caseName string,
	keys, hits, misses []T,
) {
	t.Helper()
	if len(keys) == 0 || len(hits) == 0 || len(misses) == 0 {
		t.Fatalf("%s: empty key/query slices are invalid", caseName)
	}

	v1 := EmptyWithCapacity[T](uint32(len(keys))) //nolint:gosec
	v2 := make(map[T]struct{}, len(keys))
	if reflect.DeepEqual(v1, v2) {
		t.Fatalf("%s: sanity check failed: empty Set3 and empty map are deeply equal (unexpected)", caseName)
	}

	var modeName string
	var totalOps uint64
	var runSet func() uint64
	var runMap func() uint64

	switch cfg.mode {
	case "full":
		modeName = "full"
		totalOps = uint64(len(keys) + len(hits) + len(misses) + len(keys)/2 + len(keys)/2) //nolint:gosec
		runSet = func() uint64 { return runSet3Workload(keys, hits, misses) }
		runMap = func() uint64 { return runNativeMapWorkload(keys, hits, misses) }
	case "build", "build-only":
		modeName = "build-only"
		totalOps = uint64(len(keys)) //nolint:gosec
		runSet = func() uint64 { return runSet3BuildOnlyWorkload(keys) }
		runMap = func() uint64 { return runNativeMapBuildOnlyWorkload(keys) }
	case "steady", "steady-state", "steady-state-only":
		modeName = "steady-state-only"
		totalOps = uint64(len(hits) + len(misses) + len(keys)/2 + len(keys)/2) //nolint:gosec

		steadySet := EmptyWithCapacity[T](uint32(len(keys))) //nolint:gosec
		for _, k := range keys {
			steadySet.Add(k)
		}
		steadyMap := make(map[T]struct{}, len(keys))
		for _, k := range keys {
			steadyMap[k] = struct{}{}
		}

		runSet = func() uint64 { return runSet3SteadyWorkload(steadySet, keys, hits, misses) }
		runMap = func() uint64 { return runNativeMapSteadyWorkload(steadyMap, keys, hits, misses) }
	default:
		t.Fatalf("%s: invalid SET3_RTCOMPARE_MODE=%q (allowed: full, build-only, steady-state-only)", caseName, cfg.mode)
	}

	timesSet3 := make([]float64, 0, cfg.repeats)
	timesMap := make([]float64, 0, cfg.repeats)
	allocSet3 := make([]float64, 0, cfg.repeats)
	allocMap := make([]float64, 0, cfg.repeats)
	mallocSet3 := make([]float64, 0, cfg.repeats)
	mallocMap := make([]float64, 0, cfg.repeats)

	speedups := []float64{0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	cycle := uint64(0)
	for {
		cycle++
		for i := uint64(0); i < cfg.repeats; i++ {
			var setTime, mapTime float64
			var setAlloc, mapAlloc float64
			var setMallocs, mapMallocs float64
			var checksumA, checksumB uint64

			if i%2 == 0 {
				runtime.GC()
				debug.SetGCPercent(-1)
				setTime, setAlloc, setMallocs, checksumA = measureKernelSample(runSet, totalOps)
				debug.SetGCPercent(gcval)

				runtime.GC()
				debug.SetGCPercent(-1)
				mapTime, mapAlloc, mapMallocs, checksumB = measureKernelSample(runMap, totalOps)
				debug.SetGCPercent(gcval)
			} else {
				runtime.GC()
				debug.SetGCPercent(-1)
				mapTime, mapAlloc, mapMallocs, checksumB = measureKernelSample(runMap, totalOps)
				debug.SetGCPercent(gcval)

				runtime.GC()
				debug.SetGCPercent(-1)
				setTime, setAlloc, setMallocs, checksumA = measureKernelSample(runSet, totalOps)
				debug.SetGCPercent(gcval)
			}

			if setTime <= 0 || mapTime <= 0 || setAlloc < 0 || mapAlloc < 0 || setMallocs < 0 || mapMallocs < 0 {
				t.Fatalf("%s: invalid sample at pair %d: set(time=%g alloc=%g malloc=%g) map(time=%g alloc=%g malloc=%g)",
					caseName, i, setTime, setAlloc, setMallocs, mapTime, mapAlloc, mapMallocs)
			}

			setAlloc = strictlyPositiveMetric(setAlloc)
			mapAlloc = strictlyPositiveMetric(mapAlloc)
			setMallocs = strictlyPositiveMetric(setMallocs)
			mapMallocs = strictlyPositiveMetric(mapMallocs)

			rtcompareSetSink ^= checksumA
			rtcompareSetSink ^= checksumB
			timesSet3 = append(timesSet3, setTime)
			timesMap = append(timesMap, mapTime)
			allocSet3 = append(allocSet3, setAlloc)
			allocMap = append(allocMap, mapAlloc)
			mallocSet3 = append(mallocSet3, setMallocs)
			mallocMap = append(mallocMap, mapMallocs)

			pairsDone := uint64(len(timesSet3))
			if cfg.refreshEvery > 0 && pairsDone%cfg.refreshEvery == 0 {
				reportInterim(t, caseName, pairsDone, timesSet3, timesMap, allocSet3, allocMap, mallocSet3, mallocMap)
			}
		}

		reportPairedMetricABBA(t, "runtime", "ns/op", "Set3", "nativeMap", uint64(len(timesSet3)), cfg.precisionLevel, speedups, timesSet3, timesMap)
		reportPairedMetricABBA(t, "allocation", "B/op", "Set3", "nativeMap", uint64(len(allocSet3)), cfg.precisionLevel, speedups, allocSet3, allocMap)
		reportPairedMetricABBA(t, "mallocs", "mallocs/op", "Set3", "nativeMap", uint64(len(mallocSet3)), cfg.precisionLevel, speedups, mallocSet3, mallocMap)

		if !cfg.forever {
			break
		}
		if cfg.maxCycles > 0 && cycle >= cfg.maxCycles {
			break
		}

		fmt.Printf("\n[mode %s] continuing with another cycle for case=%s\n", modeName, caseName)
	}
}

func generateUint64CaseData(n int) (keys, hits, misses []uint64) {
	rng := rtcompare.NewDPRNG()
	keys = make([]uint64, n)
	hits = make([]uint64, n)
	misses = make([]uint64, n)
	seen := make(map[uint64]struct{}, n)
	for i := range n {
		var v uint64
		for {
			v = rng.Uint64() ^ (uint64(i) * 0x9e3779b97f4a7c15)
			if _, ok := seen[v]; !ok {
				seen[v] = struct{}{}
				break
			}
		}
		keys[i] = v
	}
	for i := range n {
		hits[i] = keys[(i*13+7)%n]
	}
	for i := range n {
		for {
			v := rng.Uint64() ^ (uint64(i) * 0xd6e8feb86659fd93)
			if _, ok := seen[v]; !ok {
				seen[v] = struct{}{}
				misses[i] = v
				break
			}
		}
	}
	return keys, hits, misses
}

func generateStringCaseData(n int) (keys, hits, misses []string) {
	rng := rtcompare.NewDPRNG()
	keys = make([]string, n)
	hits = make([]string, n)
	misses = make([]string, n)
	seen := make(map[string]struct{}, n)
	for i := range n {
		for {
			x := rng.Uint64() ^ (uint64(i) * 0x9e3779b97f4a7c15)
			k := "k:" + strconv.FormatUint(x, 16)
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				keys[i] = k
				break
			}
		}
	}
	for i := range n {
		hits[i] = keys[(i*17+3)%n]
	}
	for i := range n {
		for {
			x := rng.Uint64() ^ (uint64(i) * 0xa0761d6478bd642f)
			m := "m:" + strconv.FormatUint(x, 16)
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				misses[i] = m
				break
			}
		}
	}
	return keys, hits, misses
}

func generateEligibleStructCaseData(n int) (keys, hits, misses []rtEligibleStructKey) {
	rng := rtcompare.NewDPRNG()
	keys = make([]rtEligibleStructKey, n)
	hits = make([]rtEligibleStructKey, n)
	misses = make([]rtEligibleStructKey, n)
	seen := make(map[rtEligibleStructKey]struct{}, n)
	for i := range n {
		for {
			a := rng.Uint64() ^ uint64(i)
			b := rng.Uint64() ^ (uint64(i) * 0x517cc1b727220a95)
			c := rng.Uint64() ^ (uint64(i) * 0x6c8e9cf570932bd5)
			k := rtEligibleStructKey{A: a, B: b, C: c}
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				keys[i] = k
				break
			}
		}
	}
	for i := range n {
		hits[i] = keys[(i*19+5)%n]
	}
	for i := range n {
		for {
			m := rtEligibleStructKey{
				A: rng.Uint64() ^ (uint64(i) * 0x94d049bb133111eb),
				B: rng.Uint64() ^ (uint64(i) * 0xff51afd7ed558ccd),
				C: rng.Uint64() ^ (uint64(i) * 0xc4ceb9fe1a85ec53),
			}
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				misses[i] = m
				break
			}
		}
	}
	return keys, hits, misses
}

func generateNonEligibleFloatStructCaseData(n int) (keys, hits, misses []rtNonEligibleFloatStructKey) {
	rng := rtcompare.NewDPRNG()
	keys = make([]rtNonEligibleFloatStructKey, n)
	hits = make([]rtNonEligibleFloatStructKey, n)
	misses = make([]rtNonEligibleFloatStructKey, n)
	seen := make(map[rtNonEligibleFloatStructKey]struct{}, n)
	for i := range n {
		for {
			a := rng.Uint64() ^ uint64(i)
			fBits := (rng.Uint64() & ((1 << 53) - 1)) | (uint64(1023) << 52)
			f := math.Float64frombits(fBits)
			k := rtNonEligibleFloatStructKey{A: a, F: f}
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				keys[i] = k
				break
			}
		}
	}
	for i := range n {
		hits[i] = keys[(i*23+11)%n]
	}
	for i := range n {
		for {
			fBits := (rng.Uint64() & ((1 << 53) - 1)) | (uint64(1024) << 52)
			m := rtNonEligibleFloatStructKey{
				A: rng.Uint64() ^ (uint64(i) * 0x2545f4914f6cdd1d),
				F: math.Float64frombits(fBits),
			}
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				misses[i] = m
				break
			}
		}
	}
	return keys, hits, misses
}

func generateNonEligibleStringStructCaseData(n int) (keys, hits, misses []rtNonEligibleStringStructKey) {
	rng := rtcompare.NewDPRNG()
	keys = make([]rtNonEligibleStringStructKey, n)
	hits = make([]rtNonEligibleStringStructKey, n)
	misses = make([]rtNonEligibleStringStructKey, n)
	seen := make(map[rtNonEligibleStringStructKey]struct{}, n)
	for i := range n {
		for {
			a := rng.Uint64() ^ uint64(i)
			x := rng.Uint64() ^ (uint64(i) * 0x243f6a8885a308d3)
			s := "s:" + strconv.FormatUint(x, 16)
			k := rtNonEligibleStringStructKey{A: a, S: s}
			if _, ok := seen[k]; !ok {
				seen[k] = struct{}{}
				keys[i] = k
				break
			}
		}
	}
	for i := range n {
		hits[i] = keys[(i*29+13)%n]
	}
	for i := range n {
		for {
			a := rng.Uint64() ^ (uint64(i) * 0x369dea0f31a53f85)
			x := rng.Uint64() ^ (uint64(i) * 0xdb4f0b9175ae2165)
			m := rtNonEligibleStringStructKey{A: a, S: "m:" + strconv.FormatUint(x, 16)}
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				misses[i] = m
				break
			}
		}
	}
	return keys, hits, misses
}

func assertRawEligibilityExpectation[T comparable](t *testing.T, expected bool) {
	t.Helper()
	var zero T
	typeInfo := hashing.CanUseUnsafeRawByteBlockHasherType(reflect.TypeOf(zero))
	if typeInfo.Eligible != expected {
		t.Fatalf("raw-byte eligibility mismatch for %T: got %v want %v (reason: %s)", zero, typeInfo.Eligible, expected, typeInfo.Reason)
	}
}

// TestRtcompare_Set3_vs_NativeMap compares Set3[T] against map[T]struct{}
// for realistic set workloads (insert/contains/remove/reinsert), reporting
// paired AB/BA runtime and memory metrics with rtcompare confidence values.
//
// Use environment variables to control run mode:
// - SET3_RTCOMPARE_REPEATS: number of AB/BA pairs per cycle
// - SET3_RTCOMPARE_SIZE: key count per sample
// - SET3_RTCOMPARE_REFRESH: print interim summary every N pairs
// - SET3_RTCOMPARE_FOREVER: true for continuous refresh loops
// - SET3_RTCOMPARE_MAX_CYCLES: optional cap when FOREVER=true (0=unbounded)
func TestRtcompare_Set3_vs_NativeMap(t *testing.T) {
	skipSetRtcompareIfCoverageEnabled(t)
	if testing.Short() {
		t.Skip("skipping rtcompare set perf test in -short mode")
	}

	cfg := defaultSetRtcompareConfig()
	if cfg.size < 64 {
		t.Fatalf("SET3_RTCOMPARE_SIZE too small: got %d, want >= 64", cfg.size)
	}
	if cfg.repeats < 4 {
		t.Fatalf("SET3_RTCOMPARE_REPEATS too small: got %d, want >= 4", cfg.repeats)
	}

	t.Logf("rtcompare set config: size=%d repeats=%d precision=%d refresh=%d forever=%v maxCycles=%d",
		cfg.size, cfg.repeats, cfg.precisionLevel, cfg.refreshEvery, cfg.forever, cfg.maxCycles)
	t.Logf("rtcompare set mode: %s (allowed: full, build-only, steady-state-only)", cfg.mode)

	t.Run("uint64 primitive", func(t *testing.T) {
		keys, hits, misses := generateUint64CaseData(cfg.size)
		runSetVsMapRtcompare(t, cfg, "uint64", keys, hits, misses)
	})

	t.Run("string", func(t *testing.T) {
		keys, hits, misses := generateStringCaseData(cfg.size)
		runSetVsMapRtcompare(t, cfg, "string", keys, hits, misses)
	})

	t.Run("eligible struct raw-byte", func(t *testing.T) {
		assertRawEligibilityExpectation[rtEligibleStructKey](t, true)
		keys, hits, misses := generateEligibleStructCaseData(cfg.size)
		runSetVsMapRtcompare(t, cfg, "eligible-struct", keys, hits, misses)
	})

	t.Run("non-eligible struct float", func(t *testing.T) {
		assertRawEligibilityExpectation[rtNonEligibleFloatStructKey](t, false)
		keys, hits, misses := generateNonEligibleFloatStructCaseData(cfg.size)
		runSetVsMapRtcompare(t, cfg, "non-eligible-float-struct", keys, hits, misses)
	})

	t.Run("non-eligible struct string", func(t *testing.T) {
		assertRawEligibilityExpectation[rtNonEligibleStringStructKey](t, false)
		keys, hits, misses := generateNonEligibleStringStructCaseData(cfg.size)
		runSetVsMapRtcompare(t, cfg, "non-eligible-string-struct", keys, hits, misses)
	})

	fmt.Printf("\nrtcompare sink (ignore): %d\n", rtcompareSetSink)
}
