package quality

// This file preserves commented-out tests from the original hashing_test.go.
// They are kept for future reference and potential reactivation.

// Test64BitHasherUniformDistribution performs a long-running uniformity
// check for splitmix64. It hashes a large number of sequential inputs
// and accumulates counts into 2^20 buckets. The test verifies that the
// observed coefficient of variation (relative standard deviation) across
// buckets is on the same order as the statistical expectation for a
// multinomial distribution. This test is intentionally expensive and is
// skipped when `go test -short` is used.
/* func Test32BitHasherUniformDistribution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = 1 << 20 // 2^20 buckets = 1048576
	// samplesPerBucket controls how many samples fall on average into
	// each bucket. Choose 256 for a reasonable tradeoff between runtime
	// and statistical stability: mean = 256, expected rel stddev ~= 1/sqrt(256).
	const samplesPerBucket = 256
	N := uint64(buckets * samplesPerBucket)

	cprng := rtcompare.NewCPRNG(8192) // 8KB buffer

	countsHashes := make([]uint32, buckets)
	countsRands := make([]uint32, buckets)
	mask := uint64(buckets - 1)

	for range N {
		//r := getRealRandUint64() // perturb input to avoid any correlations
		r := cprng.Uint32()
		//r := i
		countsRands[uint64(r)&mask]++
		v := hashing.Splitmix64(uint64(r)) // splitmix64 does not match test criteria --- IGNORE ---
		//v := wh32(r, 0xC0FFEE1234567890) // better than splitmix64
		countsHashes[v&mask]++
	}

	// compute observed variance

	mean := float64(samplesPerBucket)
	var sqsumHashes float64
	var maxDevHashes float64
	var maxDevIdx int = -1
	var sqsumRands float64
	var maxDevRands float64

	for i := range buckets {
		c := float64(countsHashes[i])
		d := float64(c) - mean
		sqsumHashes += d * d
		if math.Abs(d)/mean > maxDevHashes {
			maxDevHashes = math.Abs(d) / mean
			maxDevIdx = i
		}
		cR := float64(countsRands[i])
		dR := float64(cR) - mean
		sqsumRands += dR * dR
		if math.Abs(dR)/mean > maxDevRands {
			maxDevRands = math.Abs(dR) / mean
		}
	}
	obsVar := sqsumHashes / float64(buckets)
	obsStdDev := math.Sqrt(obsVar)
	rndVar := sqsumRands / float64(buckets)
	rndStdDev := math.Sqrt(rndVar)

	t.Logf("uniformity: samples=%d buckets=%d mean=%.3f obsStdDevHash=%.6f obsStdDevRands=%.6f maxDevHash=%.6f maxDevRands=%.6f",
		N, buckets, mean, obsStdDev, rndStdDev, maxDevHashes, maxDevRands)

	// Allow some slack: require observed rel std to be within 1.04x expected.
	if obsStdDev > 1.04*rndStdDev {
		t.Fatalf("observed relative stddev too large: got %.6f, want <= %.6f (1.04x expected)", obsStdDev, 1.04*rndStdDev)
	}

	// Also ensure no single bucket deviates excessively (e.g. > 35%). This
	// guards against pathological clustering even if the overall stddev is ok.
	if maxDevHashes > 0.35 {
		t.Fatalf("a bucket deviated too much from mean: maxDev=%.6f at index %d, while maxDevRands=%.6f", maxDevHashes, maxDevIdx, maxDevRands)
	}
} */

// This test ensures that all 32 bit inputs (0..math.MaxUint32) produce a uniform
// distribution on the lower 7 bits after hashing with wh32 or splitmix64. It is
// a long-running test and is skipped in -short mode.
/* func Test32BitHasherUniformDistribution_Lower7Bits(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long uniformity test in short mode")
	}

	const buckets = uint64(1) << 7 // 128 buckets
	const N = uint64(1) << 32
	const samplesPerBucket = N / buckets

	countsWh32 := make([]uint32, buckets)
	countsSplitmix := make([]uint32, buckets)
	const mask = uint64(buckets - 1)

	for i := range N {
		v1 := alternatives.WH32(uint32(i), 0xC0FFEE1234567890)
		countsWh32[v1&mask]++
		v2 := hashing.Splitmix64(uint64(i) ^ 0xC0FFEE1234567890)
		countsSplitmix[v2&mask]++
	}

	mean := float64(samplesPerBucket)
	var sqsumWh32 float64
	var maxDevWh32 float64
	var maxDevIdxWh32 int64 = -1
	var sqsumSplitmix float64
	var maxDevSplitmix float64
	var maxDevIdxSplitmix int64 = -1

	for i := range buckets {
		c1 := float64(countsWh32[i])
		d1 := float64(c1) - mean
		sqsumWh32 += d1 * d1
		if math.Abs(d1)/mean > maxDevWh32 {
			maxDevWh32 = math.Abs(d1) / mean
			maxDevIdxWh32 = int64(i)
		}
		c2 := float64(countsSplitmix[i])
		d2 := float64(c2) - mean
		sqsumSplitmix += d2 * d2
		if math.Abs(d2)/mean > maxDevSplitmix {
			maxDevSplitmix = math.Abs(d2) / mean
			maxDevIdxSplitmix = int64(i)
		}
	}
	expectedRelStd := math.Sqrt((1.0 - 1.0/float64(buckets)) / mean)

	obsStdWh32 := math.Sqrt(sqsumWh32 / float64(buckets))
	obsRelStdWh32 := obsStdWh32 / mean

	obsStdSplitmix := math.Sqrt(sqsumSplitmix / float64(buckets))
	obsRelStdSplitmix := obsStdSplitmix / mean

	t.Logf("lower 7 bits uniformity test: samples=%s buckets=%s mean=%s", benchmarks.Pow2String(uint64(N)), benchmarks.Pow2String(buckets), benchmarks.Pow2String(uint64(mean)))
	t.Logf("wh32       : obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f at idx %d",
		obsRelStdWh32, expectedRelStd, maxDevWh32, maxDevIdxWh32)
	t.Logf("splitmix64 : obsRelStd=%.6f expectedRelStd=%.6f maxRelDev=%.6f at idx %d",
		obsRelStdSplitmix, expectedRelStd, maxDevSplitmix, maxDevIdxSplitmix)

	if obsRelStdWh32 > 1.03*expectedRelStd {
		t.Fatalf("wh32: observed relative stddev too large: got %.6f, want <= %.6f (3%% expected)", obsRelStdWh32, 3.0*expectedRelStd)
	}
	if maxDevWh32 > .003 {
		t.Fatalf("wh32: a bucket deviated too much from mean: maxRelDev=%.6f at index %d", maxDevWh32, maxDevIdxWh32)
	}

	if obsRelStdSplitmix > 1.03*expectedRelStd {
		t.Fatalf("splitmix64: observed relative stddev too large: got %.6f, want <= %.6f (3%% expected)", obsRelStdSplitmix, 3.0*expectedRelStd)
	}
	if maxDevSplitmix > .003 {
		t.Fatalf("splitmix64: a bucket deviated too much from mean: maxRelDev=%.6f at index %d", maxDevSplitmix, maxDevIdxSplitmix)
	}
} */

/* func TestCompareWh64Splitmix64(t *testing.T) {
	type stats struct {
		name       string
		cv         float64
		maxDevPct  float64
		collisions uint32
		avgHamming float64
		duration   time.Duration
	}

	const (
		N       = 1 << 21 // number of input values to hash
		Buckets = 3317    // number of buckets
		AvM     = 100_000 // samples for avalanche test
		seed    = 0xC0FFEE1234567890
	)

	run := func(name string, hf func(uint64, uint64) uint64) stats {
		start := time.Now()

		counts := make([]uint32, Buckets)
		seen := make(map[uint64]uint32)
		for i := range uint64(N) {
			h := hf(i, seed)
			counts[int(h&(Buckets-1))]++
			seen[h]++
		}

		// collisions
		coll := uint32(0)
		for _, v := range seen {
			if v > 1 {
				coll += v - 1
			}
		}

		// coefficient of variation
		mean := float64(N) / float64(Buckets)
		var sumsq float64
		maxDev := 0.0
		for _, c := range counts {
			diff := float64(c) - mean
			sumsq += diff * diff
			if dev := math.Abs(diff) / mean; dev > maxDev {
				maxDev = dev
			}
		}
		variance := sumsq / float64(Buckets)
		cv := math.Sqrt(variance) / mean

		// avalanche: flip one input bit and measure output hamming distance
		var totalHd int
		for i := range uint64(AvM) {
			x := i
			h1 := hf(x, seed)
			// flip one bit to test sensitivity
			h2 := hf(x^(1<<uint64(i&0x3F)), seed)
			hammingDistance := bits.OnesCount64(h1 ^ h2)
			totalHd += hammingDistance
		}
		avgHd := float64(totalHd) / float64(AvM)

		return stats{
			name:       name,
			cv:         cv,
			maxDevPct:  maxDev * 100,
			collisions: coll,
			avgHamming: avgHd,
			duration:   time.Since(start),
		}
	}

	s1 := run("splitmix64", func(v, seed uint64) uint64 { return hashing.Splitmix64(v ^ seed) })
	s2 := run("wh64      ", func(v, seed uint64) uint64 { return alternatives.WH64(v, seed) })

	fmt.Println("Hash comparison (N=", N, " buckets=", Buckets, "):")
	for _, s := range []stats{s1, s2} {
		fmt.Printf("%s: CV=%.6g maxDev=%.3f%% collisions=%d avgHamming=%.2f time=%v\n",
			s.name, s.cv, s.maxDevPct, s.collisions, s.avgHamming, s.duration)
	}
} */

/* func TestCompareWh64Splitmix64_MultiBuckets(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping multi-bucket hash comparison in short mode")
	}

	const (
		samplesPerBucket = uint64(1024)
		avm              = uint64(10_000) // avalanche samples per bucket case
		seed             = 0xC0FFEE1234567890
		bucketCases      = uint64(245)
	)

	// prepare result slices
	var (
		cvSMresults       []float64
		cvWHresults       []float64
		maxDevSMresults   []float64
		maxDevWHresults   []float64
		collSMresults     []float64
		collWHresults     []float64
		avgHdSMresults    []float64
		avgHdWHresults    []float64
		durationSMresults []float64
		durationWHresults []float64
	)

	bucketsCh := benchmarks.FilteredNumbers(bucketCases)
	for b := range bucketsCh {
		if b <= 0 {
			continue
		}
		buckets := b
		N := uint64(buckets * samplesPerBucket)

		run := func(hf func(uint64, uint64) uint64) (cv, maxDevPct, collF, avgHdDFO, durSec float64) {
			start := time.Now()

			counts := make([]uint64, buckets)
			seen := make(map[uint64]uint32)

			for i := range N {
				h := hf(i, seed)
				idx := int(h % uint64(buckets))
				counts[idx]++
				seen[h]++
			}

			// collisions
			var coll uint64
			for _, v := range seen {
				if v > 1 {
					coll += uint64(v - 1)
				}
			}

			mean := float64(N) / float64(buckets)
			var sumsq float64
			var maxDev float64
			for _, c := range counts {
				diff := float64(c) - mean
				sumsq += diff * diff
				if dev := math.Abs(diff) / mean; dev > maxDev {
					maxDev = dev
				}
			}
			variance := sumsq / float64(buckets)
			cvVal := math.Sqrt(variance) / mean

			// avalanche: flip a few different low bits to test sensitivity
			var totalHd uint64
			for i := range avm {
				x := i * buckets
				h1 := hf(x, seed)
				h2 := hf(x^(uint64(1)<<(uint(i)&0x3F)), seed)
				totalHd += uint64(bits.OnesCount64(h1 ^ h2))
			}
			avgHd := float64(totalHd) / float64(avm)
			avgHdDFO = math.Abs(avgHd - 32.0)

			return cvVal, maxDev * 100.0, float64(coll), avgHdDFO, time.Since(start).Seconds()
		}

		cvS, maxDevS, collS, avgHdS, durS := run(func(v, seed uint64) uint64 { return hashing.Splitmix64(v ^ seed) })
		cvW, maxDevW, collW, avgHdW, durW := run(func(v, seed uint64) uint64 { return alternatives.WH64(v, seed) })

		// append results
		cvSMresults = append(cvSMresults, cvS)
		cvWHresults = append(cvWHresults, cvW)
		maxDevSMresults = append(maxDevSMresults, maxDevS)
		maxDevWHresults = append(maxDevWHresults, maxDevW)
		collSMresults = append(collSMresults, collS)
		collWHresults = append(collWHresults, collW)
		avgHdSMresults = append(avgHdSMresults, avgHdS)
		avgHdWHresults = append(avgHdWHresults, avgHdW)
		durationSMresults = append(durationSMresults, durS)
		durationWHresults = append(durationWHresults, durW)

		t.Logf("buckets=%d N=%d\nsplitmix(cv=%.6g,maxDev=%.3f%%,coll=%.0f,avgHd=%.2f,dur=%.3fs)\n    wh64(cv=%.6g,maxDev=%.3f%%,coll=%.0f,avgHd=%.2f,dur=%.3fs)",
			buckets, N, cvS, maxDevS, collS, avgHdS, durS, cvW, maxDevW, collW, avgHdW, durW)
	}

	// Log summary lengths for inspection
	// t.Logf("summary lens: buckets=%d cvSM=%d cvWH=%d", bucketCases, len(cvSMresults), len(cvWHresults))
	// t.Logf("cvSM=%v", cvSMresults)
	// t.Logf("cvWH=%v", cvWHresults)
	// t.Logf("maxDevSM=%v", maxDevSMresults)
	// t.Logf("maxDevWH=%v", maxDevWHresults)
	// t.Logf("collSM=%v", collSMresults)
	// t.Logf("collWH=%v", collWHresults)
	// t.Logf("avgHdSM=%v", avgHdSMresults)
	// t.Logf("avgHdWH=%v", avgHdWHresults)
	// t.Logf("durSM=%v", durationSMresults)
	// t.Logf("durWH=%v", durationWHresults)

	// non-failing test; results are for manual inspection / further assertions

	// Pairwise statistical comparisons using rtcompare
	relativeGains := []float64{-0.20, -0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10, 0.20}
	iterations := 10_000

	compare := func(name string, a, b []float64) {
		results, err := rtcompare.CompareSamples(a, b, relativeGains, uint64(iterations))
		if err != nil {
			t.Fatalf("CompareSamples failed for %s: %v", name, err)
		}
		if len(results) < 1 {
			t.Fatalf("expected at least 1 result from CompareSamples for %s", name)
		}
		for _, r := range results {
			t.Logf("Compare %s: gain=%.3f -> confidence=%.4f", name, r.RelativeSpeedupSampleAvsSampleB, r.Confidence)
		}
	}

	// compare CV (lower is better)
	compare("CV (WH/SM)", cvWHresults, cvSMresults)
	compare("MaxDevPct (WH/SM)", maxDevWHresults, maxDevSMresults)
	compare("Collisions (WH/SM)", collWHresults, collSMresults)
	compare("AvgHamming (WH/SM)", avgHdWHresults, avgHdSMresults)
	compare("DurationSec (WH/SM)", durationWHresults, durationSMresults)
} */
