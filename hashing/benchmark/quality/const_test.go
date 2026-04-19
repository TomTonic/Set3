package quality

import (
	"math"
	"math/bits"
	"runtime"
	"slices"
	"sync"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/hashing/benchmarks"
	"github.com/TomTonic/rtcompare"
)

// getGroupIndex computes the group index for a given hash value and group count.
// This mirrors the approach used by the Set3 hash set for bucket selection.
func getGroupIndex(hash, groupCount uint64) uint64 {
	hi, _ := bits.Mul64(hash, groupCount)
	return hi
}

// computeRelStdDevAndMaxDev calculates the relative standard deviation and
// maximum relative deviation for a slice of bucket counts.
func computeRelStdDevAndMaxDev(buckets []uint32) (relStdDev, maxDev float64) {
	n := float64(len(buckets))
	var sum uint64
	for _, v := range buckets {
		sum += uint64(v)
	}
	mean := float64(sum) / n
	var varianceSum float64
	for _, v := range buckets {
		diff := float64(v) - mean
		varianceSum += diff * diff
		dev := math.Abs(diff) / mean
		if dev > maxDev {
			maxDev = dev
		}
	}
	stdDev := math.Sqrt(varianceSum / n)
	relStdDev = stdDev / mean
	return relStdDev, maxDev
}

func testConstantForSplitMix(bitWidth, bucketCount uint32, constant uint64) (relStdDev, maxDev float64) {
	buckets := make([]uint32, bucketCount)
	for i := range uint64(1<<bitWidth - 1) {
		h := hashing.Splitmix64(i * constant) // this is the same as if seed=0
		bucket := getGroupIndex(h, uint64(bucketCount))
		buckets[bucket]++
	}
	relStdDev, maxDev = computeRelStdDevAndMaxDev(buckets)
	return relStdDev, maxDev
}

// This test compares different constants to distribute 16-bit values
// into a 64-bit value for better bit dispersion. The results are hashed
// with splitmix64 and mapped into 128 buckets (low 7 bits).
func TestHashingCompare16BitConstantsForSplitMix7BitBuckets(t *testing.T) {
	var testDistribConstants = []uint64{
		0x0001000100010001, // simple replication
		hashing.GoldenRatio48,
		hashing.Sqrt2_1_48,
		hashing.Pie7_48,
		0x000100010000FFD1, // the largest prime p such that p*65535 < 2^64
	}
	t.Logf("Will iterate through %d constants\n", len(testDistribConstants))
	for ci, c := range testDistribConstants {
		buckets := [128]uint32{}
		for i := range uint64(1<<16 - 1) {
			h := hashing.Splitmix64(i * c) // this is the same as if seed=0
			bucket := h & 0x7F             // 128 buckets
			buckets[bucket]++
		}
		relStdDev, maxDev := computeRelStdDevAndMaxDev(buckets[:])
		t.Logf("%#10x (%d of %d): relStdDev=%.6f%%, maxDev=%.6f%%\n", c, ci+1, len(testDistribConstants), relStdDev*100, maxDev*100)
	}
}

// This test compares different constants to distribute 16-bit values
// into a 64-bit value for better bit dispersion. The results are hashed
// with splitmix64 and mapped into various numbers of groups.
func TestHashingCompare16BitConstantsForSplitMixGroupCountBuckets(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode (runtime over 15 minutes)")
	}
	var testDistribConstants = []uint64{
		0x0001000100010001, // simple replication
		hashing.GoldenRatio48,
		hashing.Sqrt2_1_48,
		hashing.Pie7_48,
		0x000100010000FFD1, // the largest prime p such that p*65535 < 2^64
	}
	const numberOfGroupCountsToTest = 1721 // select higher values to get more accurate results. however 307 makes the test run for over 90 minutes, for example.
	groupCountsToTest := make([]uint32, 0, numberOfGroupCountsToTest)
	for v := range benchmarks.FilteredNumbers(numberOfGroupCountsToTest) {
		groupCountsToTest = append(groupCountsToTest, uint32(v))
		if v >= 10_083 { // Set3 will not need more then 10_083 groups to store 16-bit values in a set
			break
		}
	}

	relStdDevResults := make([][numberOfGroupCountsToTest]float64, len(testDistribConstants))
	maxDevResults := make([][numberOfGroupCountsToTest]float64, len(testDistribConstants))

	t.Logf("Will iterate through %d constants testing %d group counts: %v\n", len(testDistribConstants), len(groupCountsToTest), groupCountsToTest)

	// Run the different (constant, groupCount) tests concurrently.
	var wg sync.WaitGroup
	var mu sync.Mutex
	// limit concurrency to number of CPUs
	sem := make(chan struct{}, runtime.NumCPU())

	for ci, c := range testDistribConstants {
		for gi, gc := range groupCountsToTest {
			wg.Add(1)
			sem <- struct{}{}
			go func(ci int, gi int, c uint64, gc uint32) {
				defer wg.Done()
				relStdDev, maxDev := testConstantForSplitMix(16, gc, c)
				mu.Lock()
				relStdDevResults[ci][gi] = relStdDev
				maxDevResults[ci][gi] = maxDev
				mu.Unlock()
				<-sem
			}(ci, gi, c, gc)
		}
	}
	wg.Wait()

	t.Logf("---------------\n")
	thresholds := []float64{0.005, 0.01, 0.02, 0.04}
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			c1 := testDistribConstants[i]
			c2 := testDistribConstants[j]
			relStdDev1 := relStdDevResults[i]
			relStdDev2 := relStdDevResults[j]
			results, err := rtcompare.CompareSamples(relStdDev1[:], relStdDev2[:], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("Comparing StdDev for constant %#10x (A) vs %#10x (B):\n", c1, c2)
			for _, r := range results {
				t.Logf("  Confidence for \"A ≥ %2.2f%% better then B\": %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}
	t.Logf("---------------\n")
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			c1 := testDistribConstants[i]
			c2 := testDistribConstants[j]
			maxDev1 := maxDevResults[i]
			maxDev2 := maxDevResults[j]
			results, err := rtcompare.CompareSamples(maxDev1[:], maxDev2[:], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("Comparing MaxDev for constant %#10x (A) vs %#10x (B):\n", c1, c2)
			for _, r := range results {
				t.Logf("  Confidence for \"A ≥ %2.2f%% better then B\": %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}
}

// This test compares different constants to distribute 32-bit values
// into a 64-bit value for better bit dispersion. The results are hashed
// with splitmix64 and mapped into 128 buckets (low 7 bits).
func TestHashingCompare32BitConstantsForSplitMix7BitBuckets(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode (runtime about 40 seconds)")
	}
	var testDistribConstants = []uint64{
		0x0000_0001_0000_0001,
		hashing.GoldenRatio32,
		hashing.Sqrt2_1_32,
		hashing.Pie7_32,
		0x00000000FFFFFFFB,
	}
	t.Logf("Will iterate through %d constants\n", len(testDistribConstants))
	for ci, c := range testDistribConstants {
		buckets := [128]uint32{}
		for i := range uint64(1<<32 - 1) {
			h := hashing.Splitmix64(i * c) // this is the same as if seed=0
			bucket := h & 0x7F             // 128 buckets
			buckets[bucket]++
		}
		relStdDev, maxDev := computeRelStdDevAndMaxDev(buckets[:])
		t.Logf("%#10x (%d of %d): relStdDev=%.6f%%, maxDev=%.6f%%\n", c, ci+1, len(testDistribConstants), relStdDev*100, maxDev*100)
	}
}

// This test compares different constants to distribute 32-bit values
// into a 64-bit value for better bit dispersion. The results are hashed
// with splitmix64 and mapped into various numbers of groups.
func TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets(t *testing.T) {
	t.Skip("runtime over 90 minutes!")
	const maxAvgGroupLoad = 6.5
	const groupGrowthFactor = 1.8
	var testDistribConstants = []uint64{
		0x0000_0001_0000_0001,
		hashing.GoldenRatio32,
		hashing.Sqrt2_1_32,
		hashing.Pie7_32,
		0x00000000FFFFFFFB,
	}
	const numberOfGroupCountsToTest = 307 // select higher values to get more accurate results. however 307 makes the test run for over 90 minutes, for example.
	groupCountsToTest := make([]uint32, 0, numberOfGroupCountsToTest)
	usualStartNumber := benchmarks.NextPrime(uint64(math.Ceil(21.0 / maxAvgGroupLoad)))
	current := uint32(usualStartNumber)
	for range min(numberOfGroupCountsToTest, 29) {
		groupCountsToTest = append(groupCountsToTest, current)
		current = uint32(benchmarks.NextPrime(uint64(float64(current) * groupGrowthFactor)))
	}
	for v := range benchmarks.FilteredNumbers(numberOfGroupCountsToTest) {
		groupCountsToTest = append(groupCountsToTest, uint32(v))
		// make sure to produce enough unique numbers to match the required count
		slices.Sort(groupCountsToTest)
		groupCountsToTest = benchmarks.UniqueSortedUint32s(groupCountsToTest)
		if len(groupCountsToTest) >= numberOfGroupCountsToTest {
			break
		}
	}

	relStdDevResults := make([][numberOfGroupCountsToTest]float64, len(testDistribConstants))
	maxDevResults := make([][numberOfGroupCountsToTest]float64, len(testDistribConstants))

	t.Logf("Will iterate through %d constants testing %d group counts: %v\n", len(testDistribConstants), len(groupCountsToTest), groupCountsToTest)
	t.Logf("Expect a runtime of about %d minutes!\n", numberOfGroupCountsToTest*30/90)

	// Run the different (constant, groupCount) tests concurrently.
	var wg sync.WaitGroup
	var mu sync.Mutex
	// limit concurrency to number of CPUs
	sem := make(chan struct{}, runtime.NumCPU())

	for ci, c := range testDistribConstants {
		for gi, gc := range groupCountsToTest {
			wg.Add(1)
			sem <- struct{}{}
			go func(ci int, gi int, c uint64, gc uint32) {
				defer wg.Done()
				relStdDev, maxDev := testConstantForSplitMix(32, gc, c)
				mu.Lock()
				relStdDevResults[ci][gi] = relStdDev
				maxDevResults[ci][gi] = maxDev
				t.Logf("%#10x → %5d grps: relStdDev=%.6f%%, maxDev=%.6f%%\n", c, gc, relStdDev*100, maxDev*100)
				mu.Unlock()
				<-sem
			}(ci, gi, c, gc)
		}
	}
	wg.Wait()

	t.Logf("---------------\n")
	thresholds := []float64{0.005, 0.01, 0.02, 0.04}
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			c1 := testDistribConstants[i]
			c2 := testDistribConstants[j]
			relStdDev1 := relStdDevResults[i]
			relStdDev2 := relStdDevResults[j]
			results, err := rtcompare.CompareSamples(relStdDev1[:], relStdDev2[:], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("Comparing StdDev for constant %#10x (A) vs %#10x (B):\n", c1, c2)
			for _, r := range results {
				t.Logf("  Confidence for \"A ≥ %2.2f%% better then B\": %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}
	t.Logf("---------------\n")
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			c1 := testDistribConstants[i]
			c2 := testDistribConstants[j]
			maxDev1 := maxDevResults[i]
			maxDev2 := maxDevResults[j]
			results, err := rtcompare.CompareSamples(maxDev1[:], maxDev2[:], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("Comparing MaxDev for constant %#10x (A) vs %#10x (B):\n", c1, c2)
			for _, r := range results {
				t.Logf("  Confidence for \"A ≥ %2.2f%% better then B\": %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}
}
