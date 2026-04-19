package performance

// This file preserves the commented-out performance comparison test from the
// original hashing_test.go. It is kept for future reference and potential
// reactivation.

// This test compares the performance of splitmix64 vs wh32 on hashing 32-bit inputs.
// It is skipped in -short mode.
/* func Test32BitHasherPerformanceComparison(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long performance test in short mode")
	}

	const N = 100_000_000

	const repeats = 5555
	const innerLoops = uint32(10_000)
	const expectedSpeedup = 0.1 // expect wh32 to be at least 10% faster than splitmix64
	const minConfidence = 0.95  // require at least 95% confidence

	timesWh32 := make([]float64, 0, repeats)
	timesSplitmix := make([]float64, 0, repeats)
	x := uint64(0)

	for range repeats {
		runtime.GC()
		prev := debug.SetGCPercent(-1)
		t1 := rtcompare.SampleTime()
		for i := range innerLoops {
			x ^= alternatives.WH32(i, 0xC0FFEE1234567890)
		}
		t2 := rtcompare.SampleTime()
		_ = debug.SetGCPercent(prev)
		timesWh32 = append(timesWh32, float64(rtcompare.DiffTimeStamps(t1, t2))/float64(innerLoops))

		runtime.GC()
		_ = debug.SetGCPercent(-1)
		t3 := rtcompare.SampleTime()
		for i := range innerLoops {
			x ^= hashing.Splitmix64(uint64(i) ^ 0xC0FFEE1234567890)
		}
		t4 := rtcompare.SampleTime()
		_ = debug.SetGCPercent(prev)
		timesSplitmix = append(timesSplitmix, float64(rtcompare.DiffTimeStamps(t3, t4))/float64(innerLoops))
	}

	mWh32 := rtcompare.QuickMedian(timesWh32)
	mSplitmix := rtcompare.QuickMedian(timesSplitmix)
	t.Logf("x=%d", x)
	t.Logf("median call (wh32)=%.1f ns, (splitmix64)=%.1f ns", mWh32, mSplitmix)

	if mSplitmix < mWh32 {
		t.Fatalf("expected wh32 to be faster: splitmix64=%.1f >= wh32=%.1f", mSplitmix, mWh32)
	}

	speedups := []float64{expectedSpeedup}
	results, err := rtcompare.CompareSamples(timesWh32, timesSplitmix, speedups, 10_000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	if len(results) < 1 {
		t.Fatalf("expected at least 1 result from CompareSamples, got %d", len(results))
	}
	for _, r := range results {
		t.Logf("Speedup ≥ %.2f%% → Confidence: %.3f%%\n", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
	}
	res := results[0]
	if res.Confidence < minConfidence {
		t.Fatalf("expected confidence >= %.2f for speedup %.1f, got %.3f", minConfidence, res.RelativeSpeedupSampleAvsSampleB, res.Confidence)
	}
} */
