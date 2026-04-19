package performance

import (
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/hashing/alternatives"
	"github.com/TomTonic/rtcompare"
)

var rtcompareHashSink uint64

func testHashUint32WH(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u32 := uint32(rng.Uint64())
		sum ^= alternatives.WH32(u32, seed)
	}
	return sum
}

func testHashUint32WHdet(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u32 := uint32(rng.Uint64())
		sum ^= hashing.WH32DetGR(u32, seed)
	}
	return sum
}

func testHashUint32SM(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u32 := uint32(rng.Uint64())
		u64 := 0x0000000100000001 * uint64(u32)
		sum ^= hashing.Splitmix64(u64 ^ seed)
	}
	return sum
}

func testHashUint64WH(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= alternatives.WH64(u64, seed)
	}
	return sum
}

func testHashUint64WHdet(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= hashing.WH64Det(u64, seed)
	}
	return sum
}

func testHashUint64SM(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= hashing.Splitmix64(u64 ^ seed)
	}
	return sum
}

// Performance comparisons are intentionally skipped under coverage instrumentation.
// Coverage adds counter updates in hot paths and can invert tiny runtime deltas.
func skipRtcomparePerfIfCoverageEnabled(t *testing.T) {
	t.Helper()
	if testing.CoverMode() != "" {
		t.Skip("skipping rtcompare perf test under coverage instrumentation; run without -cover for reliable runtime comparisons")
	}
}

// reportPairedABBA analyzes paired AB/BA samples and prints an always-visible summary.
// A and B are two kernels measured on paired inputs; smaller ns/op is better.
// The paired log-ratio is defined as L = log(B/A).
func reportPairedABBA(
	t *testing.T,
	labelA, labelB string,
	rounds, repeats, precisionLevel uint64,
	speedups []float64,
	timesA, timesB []float64,
) {
	t.Helper()

	if len(timesA) != len(timesB) {
		t.Fatalf("mismatched sample lengths: %s=%d, %s=%d", labelA, len(timesA), labelB, len(timesB))
	}

	pairedLogRatioBoverA := make([]float64, 0, len(timesA))
	ratioAoverB := make([]float64, 0, len(timesA))
	ratioBoverA := make([]float64, 0, len(timesA))
	baselineOnes := make([]float64, 0, len(timesA))

	for i := 0; i < len(timesA); i++ {
		a := timesA[i]
		b := timesB[i]
		if a <= 0 || b <= 0 {
			t.Fatalf("invalid sample at pair %d: %s=%g ns/op, %s=%g ns/op", i, labelA, a, labelB, b)
		}

		lr := math.Log(b) - math.Log(a) // L = log(B/A)
		pairedLogRatioBoverA = append(pairedLogRatioBoverA, lr)
		ratioAoverB = append(ratioAoverB, math.Exp(-lr)) // == A/B
		ratioBoverA = append(ratioBoverA, math.Exp(lr))  // == B/A
		baselineOnes = append(baselineOnes, 1.0)
	}

	resultsAFaster, err := rtcompare.CompareSamples(ratioAoverB, baselineOnes, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples (%s faster) failed: %v", labelA, err)
	}
	resultsBFaster, err := rtcompare.CompareSamples(ratioBoverA, baselineOnes, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples (%s faster) failed: %v", labelB, err)
	}

	meanA, _, stdA := rtcompare.Statistics(timesA)
	meanB, _, stdB := rtcompare.Statistics(timesB)
	medianA := rtcompare.Median(timesA)
	medianB := rtcompare.Median(timesB)
	medianLogRatio := rtcompare.Median(pairedLogRatioBoverA)
	meanLogRatio, _, stdLogRatio := rtcompare.Statistics(pairedLogRatioBoverA)

	geomBoverA := math.Exp(meanLogRatio)
	geomAoverB := 1.0 / geomBoverA
	medianBoverA := math.Exp(medianLogRatio)
	medianAoverB := 1.0 / medianBoverA

	fmt.Printf("\nRTCOMPARE paired AB/BA summary (%s vs %s)\n", labelA, labelB)
	fmt.Printf("  rounds=%d repeats=%d precisionLevel=%d\n", rounds, repeats, precisionLevel)
	fmt.Printf("  mean   ns/op: %s=%.6f (sd=%.6f), %s=%.6f (sd=%.6f)\n", labelA, meanA, stdA, labelB, meanB, stdB)
	fmt.Printf("  median ns/op: %s=%.6f, %s=%.6f\n", labelA, medianA, labelB, medianB)
	fmt.Printf("  paired log-ratio L = log(%s/%s): median=%.9f, mean=%.9f, sd=%.9f\n", labelB, labelA, medianLogRatio, meanLogRatio, stdLogRatio)
	fmt.Printf("  geometric speed factors from paired L:\n")
	fmt.Printf("    %s / %s      = %.9f\n", labelB, labelA, geomBoverA)
	fmt.Printf("    %s / %s      = %.9f\n", labelA, labelB, geomAoverB)
	fmt.Printf("    median %s/%s = %.9f\n", labelB, labelA, medianBoverA)
	fmt.Printf("    median %s/%s = %.9f\n", labelA, labelB, medianAoverB)

	fmt.Printf("  bootstrap confidence (paired ratios via rtcompare):\n")
	fmt.Printf("    %s faster than %s by at least X%%:\n", labelA, labelB)
	for _, r := range resultsAFaster {
		fmt.Printf("      X=%.2f%% -> %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
	fmt.Printf("    %s faster than %s by at least X%%:\n", labelB, labelA)
	for _, r := range resultsBFaster {
		fmt.Printf("      X=%.2f%% -> %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}

	t.Logf("paired AB/BA summary emitted to stdout; median%s=%.6f ns/op median%s=%.6f ns/op", labelA, medianA, labelB, medianB)
	t.Logf("paired log-ratio median=%.9f mean=%.9f (L=log(%s/%s))", medianLogRatio, meanLogRatio, labelB, labelA)
	for _, r := range resultsAFaster {
		t.Logf("%s faster by >= %.2f%% -> confidence %.3f%%", labelA, r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
	for _, r := range resultsBFaster {
		t.Logf("%s faster by >= %.2f%% -> confidence %.3f%%", labelB, r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}

func TestRtcompare_HashUint32WH_vs_HashUint32SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	rngWHVal := rtcompare.NewDPRNG()
	rngWH := &rngWHVal
	rngSMVal := rtcompare.DPRNG{State: rngWH.State, Round: rngWH.Round}
	rngSM := &rngSMVal
	rtcompareHashSink ^= testHashUint32WH(rngWH, seed, 1024)
	rtcompareHashSink ^= testHashUint32SM(rngSM, seed, 1024)

	timesWH := make([]float64, 0, repeats)
	timesSM := make([]float64, 0, repeats)

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for i := uint64(0); i < repeats; i++ {
		var diffWH int64
		var diffSM int64
		if i%2 == 0 {
			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)
		} else {
			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)
		}

		if diffWH <= 0 || diffSM <= 0 {
			t.Fatalf("invalid timing sample at pair %d: diffWH=%d ns, diffSM=%d ns", i, diffWH, diffSM)
		}

		durWH := float64(diffWH) / float64(rounds)
		durSM := float64(diffSM) / float64(rounds)
		if durWH <= 0 || durSM <= 0 {
			t.Fatalf("invalid per-op timing at pair %d: durWH=%g ns/op, durSM=%g ns/op", i, durWH, durSM)
		}

		timesWH = append(timesWH, durWH)
		timesSM = append(timesSM, durSM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	reportPairedABBA(t, "WH32", "SM32", rounds, repeats, precisionLevel, speedups, timesWH, timesSM)
}

func TestRtcompare_HashUint32WHdet_vs_HashUint32SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	rngWHdetVal := rtcompare.NewDPRNG()
	rngWHdet := &rngWHdetVal
	rngSMVal := rtcompare.DPRNG{State: rngWHdet.State, Round: rngWHdet.Round}
	rngSM := &rngSMVal
	rtcompareHashSink ^= testHashUint32WHdet(rngWHdet, seed, 1024)
	rtcompareHashSink ^= testHashUint32SM(rngSM, seed, 1024)

	timesWHdet := make([]float64, 0, repeats)
	timesSM := make([]float64, 0, repeats)

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for i := uint64(0); i < repeats; i++ {
		var diffWHdet int64
		var diffSM int64

		// Deterministic AB/BA alternation to reduce first-vs-second bias.
		if i%2 == 0 {
			// A then B: WHdet then SM
			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32WHdet(rngWHdet, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tA0, tA1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)
		} else {
			// B then A: SM then WHdet
			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint32WHdet(rngWHdet, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tA0, tA1)
		}

		if diffWHdet <= 0 || diffSM <= 0 {
			t.Fatalf("invalid timing sample at pair %d: diffWHdet=%d ns, diffSM=%d ns", i, diffWHdet, diffSM)
		}

		durWHdet := float64(diffWHdet) / float64(rounds)
		durSM := float64(diffSM) / float64(rounds)
		if durWHdet <= 0 || durSM <= 0 {
			t.Fatalf("invalid per-op timing at pair %d: durWHdet=%g ns/op, durSM=%g ns/op", i, durWHdet, durSM)
		}

		timesWHdet = append(timesWHdet, durWHdet)
		timesSM = append(timesSM, durSM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	reportPairedABBA(t, "WHdet32", "SM32", rounds, repeats, precisionLevel, speedups, timesWHdet, timesSM)
}

func TestRtcompare_HashUint64WH_vs_HashUint64SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	rngWHVal := rtcompare.NewDPRNG()
	rngWH := &rngWHVal
	rngSMVal := rtcompare.DPRNG{State: rngWH.State, Round: rngWH.Round}
	rngSM := &rngSMVal
	rtcompareHashSink ^= testHashUint64WH(rngWH, seed, 1024)
	rtcompareHashSink ^= testHashUint64SM(rngSM, seed, 1024)

	timesWH := make([]float64, 0, repeats)
	timesSM := make([]float64, 0, repeats)

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for i := uint64(0); i < repeats; i++ {
		var diffWH int64
		var diffSM int64
		if i%2 == 0 {
			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)
		} else {
			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)
		}

		if diffWH <= 0 || diffSM <= 0 {
			t.Fatalf("invalid timing sample at pair %d: diffWH=%d ns, diffSM=%d ns", i, diffWH, diffSM)
		}

		durWH := float64(diffWH) / float64(rounds)
		durSM := float64(diffSM) / float64(rounds)
		if durWH <= 0 || durSM <= 0 {
			t.Fatalf("invalid per-op timing at pair %d: durWH=%g ns/op, durSM=%g ns/op", i, durWH, durSM)
		}

		timesWH = append(timesWH, durWH)
		timesSM = append(timesSM, durSM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	reportPairedABBA(t, "WH64", "SM64", rounds, repeats, precisionLevel, speedups, timesWH, timesSM)
}

func TestRtcompare_HashUint64WH_vs_HashUint64WHdet(t *testing.T) {
	const (
		repeats        = 3145
		rounds         = 400_000
		precisionLevel = 10_000
	)
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	rngWHVal := rtcompare.NewDPRNG()
	rngWH := &rngWHVal
	rngWHdetVal := rtcompare.DPRNG{State: rngWH.State, Round: rngWH.Round}
	rngWHdet := &rngWHdetVal
	rtcompareHashSink ^= testHashUint64WH(rngWH, seed, 1024)
	rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, 1024)

	timesWH := make([]float64, 0, repeats)
	timesWHdet := make([]float64, 0, repeats)

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for i := uint64(0); i < repeats; i++ {
		var diffWH int64
		var diffWHdet int64
		if i%2 == 0 {
			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tB0, tB1)
		} else {
			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tB0, tB1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WH(rngWH, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWH = rtcompare.DiffTimeStamps(tA0, tA1)
		}

		if diffWH <= 0 || diffWHdet <= 0 {
			t.Fatalf("invalid timing sample at pair %d: diffWH=%d ns, diffWHdet=%d ns", i, diffWH, diffWHdet)
		}

		durWH := float64(diffWH) / float64(rounds)
		durWHdet := float64(diffWHdet) / float64(rounds)
		if durWH <= 0 || durWHdet <= 0 {
			t.Fatalf("invalid per-op timing at pair %d: durWH=%g ns/op, durWHdet=%g ns/op", i, durWH, durWHdet)
		}

		timesWH = append(timesWH, durWH)
		timesWHdet = append(timesWHdet, durWHdet)
	}

	speedups := []float64{0.00125, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	reportPairedABBA(t, "WH64", "WHdet64", rounds, repeats, precisionLevel, speedups, timesWH, timesWHdet)
}

func TestRtcompare_HashUint64WHdet_vs_HashUint64SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	rngWHdetVal := rtcompare.NewDPRNG()
	rngWHdet := &rngWHdetVal
	rngSMVal := rtcompare.DPRNG{State: rngWHdet.State, Round: rngWHdet.Round}
	rngSM := &rngSMVal
	rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, 1024)
	rtcompareHashSink ^= testHashUint64SM(rngSM, seed, 1024)

	timesWHdet := make([]float64, 0, repeats)
	timesSM := make([]float64, 0, repeats)

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for i := uint64(0); i < repeats; i++ {
		var diffWHdet int64
		var diffSM int64
		if i%2 == 0 {
			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tA0, tA1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)
		} else {
			runtime.GC()
			debug.SetGCPercent(-1)
			tB0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64SM(rngSM, seed, rounds)
			tB1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffSM = rtcompare.DiffTimeStamps(tB0, tB1)

			runtime.GC()
			debug.SetGCPercent(-1)
			tA0 := rtcompare.SampleTime()
			rtcompareHashSink ^= testHashUint64WHdet(rngWHdet, seed, rounds)
			tA1 := rtcompare.SampleTime()
			debug.SetGCPercent(gcval)
			diffWHdet = rtcompare.DiffTimeStamps(tA0, tA1)
		}

		if diffWHdet <= 0 || diffSM <= 0 {
			t.Fatalf("invalid timing sample at pair %d: diffWHdet=%d ns, diffSM=%d ns", i, diffWHdet, diffSM)
		}

		durWHdet := float64(diffWHdet) / float64(rounds)
		durSM := float64(diffSM) / float64(rounds)
		if durWHdet <= 0 || durSM <= 0 {
			t.Fatalf("invalid per-op timing at pair %d: durWHdet=%g ns/op, durSM=%g ns/op", i, durWHdet, durSM)
		}

		timesWHdet = append(timesWHdet, durWHdet)
		timesSM = append(timesSM, durSM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	reportPairedABBA(t, "WHdet64", "SM64", rounds, repeats, precisionLevel, speedups, timesWHdet, timesSM)
}

// --- from hashing_perf32_test.go ---

var rtcompareHashSink2 uint64

// TestRtcompare_HashUint32WH_vs_HashUint32WHdet tests the performance
// difference between wh32 (random keys) and wh32det (deterministic).
func TestRtcompare_HashUint32WH_vs_HashUint32WHdet(t *testing.T) {
	const (
		repeats        = 3145
		rounds         = 400_000
		precisionLevel = 10_000
	)

	seed := uint64(0x1234_5678_9abc_def0)

	rngA_val := rtcompare.NewDPRNG()
	rngA := &rngA_val
	rngB_val := rtcompare.DPRNG{State: rngA.State, Round: rngA.Round}
	rngB := &rngB_val
	rtcompareHashSink2 ^= testHashUint32WH(rngA, seed, 1024)
	rtcompareHashSink2 ^= testHashUint32WHdet(rngB, seed, 1024)

	var timesWH []float64
	var timesWHdet []float64

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for range repeats {
		runtime.GC()
		debug.SetGCPercent(-1)

		t1 := rtcompare.SampleTime()
		rtcompareHashSink2 ^= testHashUint32WH(rngA, seed, rounds)
		t2 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durWH := float64(rtcompare.DiffTimeStamps(t1, t2)) / float64(rounds)
		timesWH = append(timesWH, durWH)

		runtime.GC()
		debug.SetGCPercent(-1)

		t3 := rtcompare.SampleTime()
		rtcompareHashSink2 ^= testHashUint32WHdet(rngB, seed, rounds)
		t4 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durWHdet := float64(rtcompare.DiffTimeStamps(t3, t4)) / float64(rounds)
		timesWHdet = append(timesWHdet, durWHdet)
	}

	speedups := []float64{0.00125, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	results, err := rtcompare.CompareSamples(timesWHdet, timesWH, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples failed: %v", err)
	}
	t.Logf("rtcompare: hashUint32WHdet (A) vs hashUint32WH (B); rounds=%d repeats=%d", rounds, repeats)
	for _, r := range results {
		t.Logf("speedup(A vs B) ≥ %.2f%% → confidence: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}
