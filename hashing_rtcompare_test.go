package set3

import (
	"runtime"
	"runtime/debug"
	"testing"

	"github.com/TomTonic/rtcompare"
)

var rtcompareHashSink uint64

func testHashUint32WH(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u32 := uint32(rng.Uint64())
		sum ^= wh32(u32, seed)
	}
	return sum
}

func testHashUint32SM(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u32 := uint32(rng.Uint64())
		u64 := 0x0000000100000001 * uint64(u32)
		sum ^= splitmix64(u64 ^ seed)
	}
	return sum
}

func testHashUint64WH(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= wh64(u64, seed)
	}
	return sum
}

func testHashUint64WHdet(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= wh64det(u64, seed)
	}
	return sum
}

func testHashUint64SM(rng *rtcompare.DPRNG, seed, count uint64) uint64 {
	var sum uint64
	for range count {
		u64 := rng.Uint64()
		sum ^= splitmix64(u64 ^ seed)
	}
	return sum
}

func TestRtcompare_HashUint32WH_vs_HashUint32SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)

	seed := uint64(0x1234_5678_9abc_def0)

	// Warm-up both methods once to reduce one-time effects.
	rngA_val := rtcompare.NewDPRNG()
	rngA := &rngA_val
	rngB_val := rtcompare.DPRNG{State: rngA.State, Round: rngA.Round}
	rngB := &rngB_val
	rtcompareHashSink ^= testHashUint32WH(rngA, seed, 1024)
	rtcompareHashSink ^= testHashUint32SM(rngB, seed, 1024)

	var timesU32WH []float64
	var timesU32SM []float64

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for range repeats {
		// Reduce GC noise.
		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint32WH.
		t1 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint32WH(rngA, seed, rounds)
		t2 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)

		durU32WH := float64(rtcompare.DiffTimeStamps(t1, t2)) / float64(rounds)
		timesU32WH = append(timesU32WH, durU32WH)

		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint32SM.
		t3 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint32SM(rngB, seed, rounds)
		t4 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durU32SM := float64(rtcompare.DiffTimeStamps(t3, t4)) / float64(rounds)
		timesU32SM = append(timesU32SM, durU32SM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	results, err := rtcompare.CompareSamples(timesU32SM, timesU32WH, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples failed: %v", err)
	}

	t.Logf("rtcompare: hashUint32SM (A) vs hashUint32WH (B); rounds=%d repeats=%d precisionLevel=%d", rounds, repeats, precisionLevel)
	for _, r := range results {
		t.Logf("speedup(A vs B) ≥ %.2f%% -> confidence: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}

func TestRtcompare_HashUint64WH_vs_HashUint64SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)

	seed := uint64(0x1234_5678_9abc_def0)

	// Warm-up both methods once to reduce one-time effects.
	rngA_val := rtcompare.NewDPRNG()
	rngA := &rngA_val
	rngB_val := rtcompare.DPRNG{State: rngA.State, Round: rngA.Round}
	rngB := &rngB_val
	rtcompareHashSink ^= testHashUint64WH(rngA, seed, 1024)
	rtcompareHashSink ^= testHashUint64SM(rngB, seed, 1024)

	var timesU64WH []float64
	var timesU64SM []float64

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for range repeats {
		// Reduce GC noise.
		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64WH.
		t1 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64WH(rngA, seed, rounds)
		t2 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)

		durU64WH := float64(rtcompare.DiffTimeStamps(t1, t2)) / float64(rounds)
		timesU64WH = append(timesU64WH, durU64WH)

		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64SM.
		t3 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64SM(rngB, seed, rounds)
		t4 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durU64SM := float64(rtcompare.DiffTimeStamps(t3, t4)) / float64(rounds)
		timesU64SM = append(timesU64SM, durU64SM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	results, err := rtcompare.CompareSamples(timesU64SM, timesU64WH, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples failed: %v", err)
	}

	t.Logf("rtcompare: hashUint64SM (A) vs hashUint64WH (B); rounds=%d repeats=%d precisionLevel=%d", rounds, repeats, precisionLevel)
	for _, r := range results {
		t.Logf("speedup(A vs B) ≥ %.2f%% -> confidence: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}

func TestRtcompare_HashUint64WH_vs_HashUint64WHdet(t *testing.T) {
	const (
		repeats        = 3145
		rounds         = 400_000
		precisionLevel = 10_000
	)

	seed := uint64(0x1234_5678_9abc_def0)

	// Warm-up both methods once to reduce one-time effects.
	rngA_val := rtcompare.NewDPRNG()
	rngA := &rngA_val
	rngB_val := rtcompare.DPRNG{State: rngA.State, Round: rngA.Round}
	rngB := &rngB_val
	rtcompareHashSink ^= testHashUint64WH(rngA, seed, 1024)
	rtcompareHashSink ^= testHashUint64WHdet(rngB, seed, 1024)

	var timesU64WH []float64
	var timesU64WHshort []float64

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for range repeats {
		// Reduce GC noise.
		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64WH.
		t1 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64WH(rngA, seed, rounds)
		t2 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)

		durU64WH := float64(rtcompare.DiffTimeStamps(t1, t2)) / float64(rounds)
		timesU64WH = append(timesU64WH, durU64WH)

		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64WHshort.
		t3 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64WHdet(rngB, seed, rounds)
		t4 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durU64WHshort := float64(rtcompare.DiffTimeStamps(t3, t4)) / float64(rounds)
		timesU64WHshort = append(timesU64WHshort, durU64WHshort)
	}

	speedups := []float64{0.00125, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	results, err := rtcompare.CompareSamples(timesU64WHshort, timesU64WH, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples failed: %v", err)
	}

	t.Logf("rtcompare: hashUint64WHdet (A) vs hashUint64WH (B); rounds=%d repeats=%d precisionLevel=%d", rounds, repeats, precisionLevel)
	for _, r := range results {
		t.Logf("speedup(A vs B) ≥ %.2f%% -> confidence: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}

func TestRtcompare_HashUint64WHdet_vs_HashUint64SM(t *testing.T) {
	const (
		repeats        = 5137
		rounds         = 100_000
		precisionLevel = 10_000
	)

	seed := uint64(0x1234_5678_9abc_def0)

	// Warm-up both methods once to reduce one-time effects.
	rngA_val := rtcompare.NewDPRNG()
	rngA := &rngA_val
	rngB_val := rtcompare.DPRNG{State: rngA.State, Round: rngA.Round}
	rngB := &rngB_val
	rtcompareHashSink ^= testHashUint64WHdet(rngA, seed, 1024)
	rtcompareHashSink ^= testHashUint64SM(rngB, seed, 1024)

	var timesU64WH []float64
	var timesU64SM []float64

	gcval := debug.SetGCPercent(-1)
	debug.SetGCPercent(gcval)
	defer debug.SetGCPercent(gcval)

	for range repeats {
		// Reduce GC noise.
		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64WH.
		t1 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64WHdet(rngA, seed, rounds)
		t2 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)

		durU64WH := float64(rtcompare.DiffTimeStamps(t1, t2)) / float64(rounds)
		timesU64WH = append(timesU64WH, durU64WH)

		runtime.GC()
		debug.SetGCPercent(-1)

		// Measure hashUint64SM.
		t3 := rtcompare.SampleTime()
		rtcompareHashSink ^= testHashUint64SM(rngB, seed, rounds)
		t4 := rtcompare.SampleTime()
		debug.SetGCPercent(gcval)
		durU64SM := float64(rtcompare.DiffTimeStamps(t3, t4)) / float64(rounds)
		timesU64SM = append(timesU64SM, durU64SM)
	}

	speedups := []float64{0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4}
	results, err := rtcompare.CompareSamples(timesU64WH, timesU64SM, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("rtcompare.CompareSamples failed: %v", err)
	}

	t.Logf("rtcompare: hashUint64WHdet (A) vs hashUint64SM (B); rounds=%d repeats=%d precisionLevel=%d", rounds, repeats, precisionLevel)
	for _, r := range results {
		t.Logf("speedup(A vs B) ≥ %.2f%% → confidence: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
}
