package quality

import (
	"math/bits"
	"runtime"
	"sync"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/rtcompare"
)

// TestDistributionQuality_SM_vs_WHdet_32bit compares the actual hash functions
// hashI32SM (splitmix64 with goldenRatio32 spread) and hashI32WHdet (deterministic
// wyhash) for their bucket distribution quality across a range of group counts.
// Both are tested exhaustively over the full uint32 range.
func TestDistributionQuality_SM_vs_WHdet_32bit(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	// Group counts that Set3 actually uses (primes from calcNextGroupCount progression)
	groupCountsToTest := []uint64{
		3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 59, 67, 79, 97,
	}

	seeds := []uint64{0, 0x1234_5678_9abc_def0, 0xDEAD_BEEF_CAFE_BABE}

	type result struct {
		groupCount uint64
		seed       uint64
		smRelStd   float64
		smMaxDev   float64
		whRelStd   float64
		whMaxDev   float64
	}

	var mu sync.Mutex
	var results []result
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	for _, gc := range groupCountsToTest {
		for _, seed := range seeds {
			wg.Add(1)
			sem <- struct{}{}
			go func(gc, seed uint64) {
				defer wg.Done()
				smBuckets := make([]uint32, gc)
				whBuckets := make([]uint32, gc)

				for i := uint64(0); i < 1<<32; i++ {
					u32 := uint32(i)

					// SM path: same as HashI32SM
					smV := hashing.GoldenRatio32 * uint64(u32)
					smH := hashing.Splitmix64(smV ^ seed)
					smHi, _ := bits.Mul64(smH, gc)
					smBuckets[smHi]++

					// WHdet path: same as HashI32WHdet
					whH := hashing.WH32DetGR(u32, seed)
					whHi, _ := bits.Mul64(whH, gc)
					whBuckets[whHi]++
				}

				smRelStd, smMaxDev := computeRelStdDevAndMaxDev(smBuckets)
				whRelStd, whMaxDev := computeRelStdDevAndMaxDev(whBuckets)

				mu.Lock()
				results = append(results, result{gc, seed, smRelStd, smMaxDev, whRelStd, whMaxDev})
				mu.Unlock()
				<-sem
			}(gc, seed)
		}
	}
	wg.Wait()

	// Collect per-groupCount pairs for rtcompare
	var smRelStds, whRelStds []float64
	var smMaxDevs, whMaxDevs []float64

	for _, r := range results {
		t.Logf("gc=%3d seed=%016x  SM: relStd=%.8f%% maxDev=%.8f%%  WHdet: relStd=%.8f%% maxDev=%.8f%%",
			r.groupCount, r.seed,
			r.smRelStd*100, r.smMaxDev*100,
			r.whRelStd*100, r.whMaxDev*100)
		smRelStds = append(smRelStds, r.smRelStd)
		whRelStds = append(whRelStds, r.whRelStd)
		smMaxDevs = append(smMaxDevs, r.smMaxDev)
		whMaxDevs = append(whMaxDevs, r.whMaxDev)
	}

	thresholds := []float64{0.001, 0.005, 0.01, 0.02, 0.04}

	t.Logf("--- RelStdDev: SM (A) vs WHdet (B) ---")
	res1, err := rtcompare.CompareSamples(smRelStds, whRelStds, thresholds, 10000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	for _, r := range res1 {
		t.Logf("  SM ≥ %.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100, r.Confidence*100)
	}

	t.Logf("--- RelStdDev: WHdet (A) vs SM (B) ---")
	res2, err := rtcompare.CompareSamples(whRelStds, smRelStds, thresholds, 10000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	for _, r := range res2 {
		t.Logf("  WHdet ≥ %.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100, r.Confidence*100)
	}

	t.Logf("--- MaxDev: SM (A) vs WHdet (B) ---")
	res3, err := rtcompare.CompareSamples(smMaxDevs, whMaxDevs, thresholds, 10000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	for _, r := range res3 {
		t.Logf("  SM ≥ %.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100, r.Confidence*100)
	}

	t.Logf("--- MaxDev: WHdet (A) vs SM (B) ---")
	res4, err := rtcompare.CompareSamples(whMaxDevs, smMaxDevs, thresholds, 10000)
	if err != nil {
		t.Fatalf("CompareSamples failed: %v", err)
	}
	for _, r := range res4 {
		t.Logf("  WHdet ≥ %.2f%% better: %.3f%%", r.RelativeSpeedupSampleAvsSampleB*100, r.Confidence*100)
	}
}
