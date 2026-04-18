package set3

import (
	"runtime"
	"sync"
	"testing"

	"github.com/TomTonic/rtcompare"
)

// TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets_Fast is a
// faster variant of TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets
// that tests a reduced set of group counts (those actually reachable by Set3
// for uint32 cardinalities). Each (constant, groupCount) pair iterates all
// 2^32-1 values — the test is exhaustive over the value range.
func TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets_Fast(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	var testDistribConstants = []uint64{
		0x0000_0001_0000_0001, // simple replication
		goldenRatio32,
		sqrt2_1_32,
		pie7_32,
		0x00000000FFFFFFFB, // largest prime p such that p*2^32 < 2^64
	}
	constNames := []string{
		"replication",
		"goldenRatio32",
		"sqrt2_1_32",
		"pie7_32",
		"largestPrime",
	}

	// Select group counts that Set3 would actually use for various uint32 set sizes.
	// These cover a range from small sets to sets holding all 2^32 values.
	groupCountsToTest := []uint32{
		3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 59, 67, 79, 97,
	}

	numberOfGroupCountsToTest := len(groupCountsToTest)
	relStdDevResults := make([][]float64, len(testDistribConstants))
	maxDevResults := make([][]float64, len(testDistribConstants))
	for i := range testDistribConstants {
		relStdDevResults[i] = make([]float64, numberOfGroupCountsToTest)
		maxDevResults[i] = make([]float64, numberOfGroupCountsToTest)
	}

	t.Logf("Exhaustive 32-bit test: %d constants × %d group counts", len(testDistribConstants), numberOfGroupCountsToTest)
	t.Logf("Group counts: %v", groupCountsToTest)

	// Run concurrently.
	var wg sync.WaitGroup
	var mu sync.Mutex
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
				mu.Unlock()
				<-sem
			}(ci, gi, c, gc)
		}
	}
	wg.Wait()

	// Log raw results.
	for ci, name := range constNames {
		for gi, gc := range groupCountsToTest {
			t.Logf("%15s → %5d grps: relStdDev=%.6f%%, maxDev=%.6f%%", name, gc, relStdDevResults[ci][gi]*100, maxDevResults[ci][gi]*100)
		}
	}

	// Statistical comparison using rtcompare.
	t.Logf("--- StdDev comparisons ---")
	thresholds := []float64{0.005, 0.01, 0.02, 0.04}
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			results, err := rtcompare.CompareSamples(relStdDevResults[i], relStdDevResults[j], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("StdDev %s (A) vs %s (B):", constNames[i], constNames[j])
			for _, r := range results {
				t.Logf("  A ≥ %2.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}

	t.Logf("--- MaxDev comparisons ---")
	for i := range len(testDistribConstants) {
		for j := range len(testDistribConstants) {
			if i == j {
				continue
			}
			results, err := rtcompare.CompareSamples(maxDevResults[i], maxDevResults[j], thresholds, 10000)
			if err != nil {
				t.Fatalf("rtcompare.CompareSamples failed: %v", err)
			}
			t.Logf("MaxDev %s (A) vs %s (B):", constNames[i], constNames[j])
			for _, r := range results {
				t.Logf("  A ≥ %2.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100.0, r.Confidence*100.0)
			}
		}
	}
}
