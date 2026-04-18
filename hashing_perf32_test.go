package set3

import (
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/TomTonic/rtcompare"
)

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
