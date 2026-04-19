package quality

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/hashing/alternatives"
	"github.com/TomTonic/Set3/hashing/benchmark"
	"github.com/TomTonic/rtcompare"
)

// rankCandidate32 describes one mixer+constant combination.
// The widening step always multiplies uint32 -> uint64 using the candidate constant.
// The mixer is then either hashing.Splitmix64(seed^widened) or alternatives.WH32DetExtMul(widened, seed).
type rankCandidate32 struct {
	mixerName string
	constName string
	constant  uint64
	useWH32   bool
	isProd    bool // true if this combination matches the current production hash for uint32/int32
}

func (c rankCandidate32) label() string {
	s := fmt.Sprintf("%s + %s", c.mixerName, c.constName)
	if c.isProd {
		s += " [PROD]"
	}
	return s
}

// rankResult32 stores quality metrics for one candidate.
// All percentage values are "lower is better".
type rankResult32 struct {
	candidate rankCandidate32
	samples   uint64
	seed      uint64

	lowRelStdPct  float64
	lowMaxDevPct  float64
	highRelStdPct float64
	highMaxDevPct float64
	lowChi2       float64
	highChi2      float64
	groupCounts   int
	groupSamples  uint64

	groupRelStdMeanPct  float64
	groupMaxDevMeanPct  float64
	groupRelStdWorstPct float64
	groupMaxDevWorstPct float64

	// Composite score tuned for Set3 needs.
	// Interpretation: lower is better.
	score float64
}

// Weights for a Set3-oriented ranking.
//
// Why these weights:
//   - lowMaxDev (29%): strongest single factor, because one overloaded H2 bucket can
//     create long probe chains even if global variance looks good.
//   - lowRelStd (20%): broad low-bit uniformity matters for H2 filtering quality.
//   - groupMaxMean (22%) + groupRelMean (16%): direct pressure on real Set3 group
//     mapping behavior via getGroupIndex across many group counts.
//   - groupMaxWorst (8%): guards against pathological outliers at specific table sizes.
//   - highMaxDev (3%) + highRelStd (2%): secondary signal for bits 63–57 (the top 7 bits
//     of the hash), which are the most influential inputs to the Mul64 group mapping.
//     They serve as an independent cross-check; largely redundant with the group-mapping
//     metrics but catch extreme bit-quality failures early.
//
// The weights sum to exactly 1.00 by design.
const (
	weightLowMaxDev     = 0.29
	weightLowRelStd     = 0.20
	weightGroupMaxMean  = 0.22
	weightGroupRelMean  = 0.16
	weightGroupMaxWorst = 0.08
	weightHighMaxDev    = 0.03
	weightHighRelStd    = 0.02

	// group-count selection knobs for Set3 mapping analysis.
	groupCountRehashPrefix = 33
	groupCountFilteredHint = 300
	groupCountFilteredMax  = 10_083 // 65536 values do not need more than this number of buckets
	groupCountRankCap      = 72

	// Keep expected bucket occupancy >= sampleCount/groupCountSampleOccFloor
	// in sampling-based group mapping metrics.
	groupCountSampleOccFloor = 128
)

const quickSampleCountDefault uint64 = 2_000_000

// hashRankSampleSeedDefault is used for deterministic sample-based ranking tests.
// You can override it via SET3_HASH_RANK_SAMPLE_SEED to verify stability across seeds.
const hashRankSampleSeedDefault uint64 = 0x96C0_FFEE_C0FF_EE69

var hashRankDefaultSeeds = []uint64{
	0x1234_5678_9abc_def0,
	0x0fed_cba9_8765_4321,
	0xdead_beef_dead_beef,
	0xbeef_dead_beef_dead,
	0x0123_4567_89ab_cdef,
}

// groupCountsSet3Rank is intentionally derived from two sources:
// 1) the actual Set3 rehash growth path (first steps),
// 2) additional filtered values to avoid overfitting to one growth trajectory.
// This borrows the core idea from TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets,
// but keeps the list capped for practical runtime.
//
// It is initialized lazily to avoid package init-order issues with nextPrime caches.
var (
	groupCountsSet3Rank     []uint32
	groupCountsSet3RankOnce sync.Once
)

func getGroupCountsSet3Rank() []uint32 {
	groupCountsSet3RankOnce.Do(func() {
		groupCountsSet3Rank = buildGroupCountsSet3Rank()
	})
	return groupCountsSet3Rank
}

// calcReqNrOfGroupsLocal mirrors Set3's calcReqNrOfGroups using maxAvgGroupLoad.
func calcReqNrOfGroupsLocal(reqCapa uint32) uint32 {
	reqNrOfGroups := uint32((float64(reqCapa) + maxAvgGroupLoad - 1) / maxAvgGroupLoad)
	if reqNrOfGroups <= 1 {
		return 1
	}
	return reqNrOfGroups
}

// calcNextGroupCountLocal mirrors Set3's calcNextGroupCount growth formula.
func calcNextGroupCountLocal(currentGroupCount uint32) uint32 {
	n := uint32(max(math.Ceil(float64(currentGroupCount)*3.0/2.0), 2))
	p := benchmark.NextPrime(uint64(n))
	return uint32(p)
}

func hashForCandidate32(c rankCandidate32, u uint32, seed uint64) uint64 {
	widened := c.constant * uint64(u)
	if c.useWH32 {
		return alternatives.WH32DetExtMul(widened, seed)
	}
	return hashing.Splitmix64(seed ^ widened)
}

// candidateSet32 returns all mixer+constant combinations to rank.
//
// Production context:
//   - The current production function for uint32/int32 in Set3 (MakeRuntimeHasher) is
//     hashI32WHdet, which is exactly wh32det(u, seed). wh32det hardcodes the widening
//     constant 0x0000_0001_0000_0001 (replication: val placed in both 32-bit halves),
//     so the production path equals wh32detExtMul + replication_0x0000000100000001 [PROD].
//   - hashI32SM (= splitmix64 + goldenRatio32) is defined in the codebase but is NOT
//     currently assigned to uint32/int32 in MakeRuntimeHasher. It is included here as
//     the primary challenger.
//
// Widening constant note for wh32detExtMul:
//
//	The WH mixer computes mix(widened^p1, widened^seed), where both sides share the same
//	widened value. The replication constant puts val in both 32-bit halves, giving the
//	full-width multiply in mix() two independent but equal lanes to work with — the
//	structurally intended design. Other widening constants are exploratory: they break
//	the structural assumption but may still produce useful entropy.
//
// Widening constant note for splitmix64:
//
//	splitmix64 takes any 64-bit scalar; the widening constant determines which 64-bit
//	value enters the mixer. goldenRatio32 is the established canonical choice (proven in
//	prior tests). Other constants are exploratory alternatives.
func candidateSet32() []rankCandidate32 {
	constants := []struct {
		name  string
		value uint64
	}{
		{name: "goldenRatio32", value: hashing.GoldenRatio32},
		{name: "sqrt2_1_32", value: hashing.Sqrt2_1_32},
		{name: "pie7_32", value: hashing.Pie7_32},
		{name: "replication_0x0000000100000001", value: 0x0000000100000001},
		{name: "largestPrime_0x00000000FFFFFFFB", value: 0x00000000FFFFFFFB},
	}

	candidates := make([]rankCandidate32, 0, len(constants)*2)
	for _, k := range constants {
		// Mark the WH+replication combination as production (= hashI32WHdet).
		isWHProd := k.value == 0x0000000100000001
		candidates = append(candidates,
			rankCandidate32{mixerName: "splitmix64", constName: k.name, constant: k.value, useWH32: false},
			rankCandidate32{mixerName: "wh32detExtMul", constName: k.name, constant: k.value, useWH32: true, isProd: isWHProd},
		)
	}
	return candidates
}

func buildGroupCountsSet3Rank() []uint32 {
	counts := make([]uint32, 0, groupCountRehashPrefix+groupCountRankCap)

	// Mirror Set3's real growth sequence for early-to-mid table sizes.
	usualStartNumber := benchmark.NextPrime(uint64(calcReqNrOfGroupsLocal(21)))
	current := uint32(usualStartNumber)
	for range groupCountRehashPrefix {
		counts = append(counts, current)
		current = uint32(benchmark.NextPrime(uint64(calcNextGroupCountLocal(current))))
	}

	// Add extra sizes from the filtered generator used in existing comparison tests.
	for v := range benchmark.FilteredNumbers(groupCountFilteredHint) {
		// Normalize to prime counts to stay aligned with Set3's real table sizing.
		counts = append(counts, uint32(benchmark.NextPrime(uint64(v))))
		if v >= groupCountFilteredMax {
			break
		}
	}

	slices.Sort(counts)
	counts = dedupeSortedUint32(counts)

	// Keep runtime bounded by selecting evenly spaced representatives.
	if len(counts) > groupCountRankCap {
		counts = pickEvenlySpacedSortedUint32(counts, groupCountRankCap)
	}

	return counts
}

func dedupeSortedUint32(in []uint32) []uint32 {
	if len(in) == 0 {
		return in
	}
	j := 0
	for i := 1; i < len(in); i++ {
		if in[i] != in[j] {
			j++
			in[j] = in[i]
		}
	}
	return in[:j+1]
}

func pickEvenlySpacedSortedUint32(sorted []uint32, target int) []uint32 {
	if target <= 0 || len(sorted) == 0 {
		return nil
	}
	if len(sorted) <= target {
		out := make([]uint32, len(sorted))
		copy(out, sorted)
		return out
	}

	out := make([]uint32, 0, target)
	last := len(sorted) - 1
	for i := range target {
		idx := int((uint64(i)*uint64(last) + uint64(target-1)/2) / uint64(target-1))
		v := sorted[idx]
		if len(out) == 0 || out[len(out)-1] != v {
			out = append(out, v)
		}
	}

	// Backfill if rounding produced duplicates.
	if len(out) < target {
		for _, v := range sorted {
			if len(out) == target {
				break
			}
			seen := false
			for _, present := range out {
				if present == v {
					seen = true
					break
				}
			}
			if !seen {
				out = append(out, v)
			}
		}
		slices.Sort(out)
	}

	return out
}

func selectGroupCountsForSample(sampleCount uint64) []uint32 {
	all := getGroupCountsSet3Rank()
	if len(all) == 0 {
		return nil
	}

	// If sampleCount is small compared to groupCount, maxDev becomes dominated by
	// sparse-sampling noise. Cap group counts to keep expected occupancy meaningful.
	maxGroupCount := uint32(sampleCount / groupCountSampleOccFloor)
	if maxGroupCount < 3 {
		maxGroupCount = 3
	}

	selected := make([]uint32, 0, len(all))
	for _, gc := range all {
		if gc <= maxGroupCount {
			selected = append(selected, gc)
		}
	}

	// Ensure at least a small baseline set for very small samples.
	if len(selected) >= 8 {
		return selected
	}
	n := min(8, len(all))
	out := make([]uint32, n)
	copy(out, all[:n])
	return out
}

func computeBucketMetrics(hist []uint64, total uint64) (relStdPct, maxDevPct, chi2 float64) {
	bucketCount := len(hist)
	if bucketCount == 0 || total == 0 {
		return 0, 0, 0
	}
	expected := float64(total) / float64(bucketCount)

	var sumSq float64
	for i := range bucketCount {
		obs := float64(hist[i])
		diff := obs - expected
		sumSq += diff * diff
		chi2 += (diff * diff) / expected

		devPct := math.Abs(diff) / expected * 100.0
		if devPct > maxDevPct {
			maxDevPct = devPct
		}
	}

	std := math.Sqrt(sumSq / float64(bucketCount))
	relStdPct = std / expected * 100.0
	return relStdPct, maxDevPct, chi2
}

func aggregateGroupMappingMetrics(groupHists [][]uint64, samples uint64) (relMean, maxMean, relWorst, maxWorst float64) {
	if len(groupHists) == 0 {
		return 0, 0, 0, 0
	}
	for _, hist := range groupHists {
		rel, max, _ := computeBucketMetrics(hist, samples)
		relMean += rel
		maxMean += max
		if rel > relWorst {
			relWorst = rel
		}
		if max > maxWorst {
			maxWorst = max
		}
	}
	n := float64(len(groupHists))
	relMean /= n
	maxMean /= n
	return relMean, maxMean, relWorst, maxWorst
}

func scoreResult(result *rankResult32) {
	result.score =
		weightLowMaxDev*result.lowMaxDevPct +
			weightLowRelStd*result.lowRelStdPct +
			weightGroupMaxMean*result.groupMaxDevMeanPct +
			weightGroupRelMean*result.groupRelStdMeanPct +
			weightGroupMaxWorst*result.groupMaxDevWorstPct +
			weightHighMaxDev*result.highMaxDevPct +
			weightHighRelStd*result.highRelStdPct
}

func evaluateCandidateFromValues(c rankCandidate32, values []uint32, seed uint64) rankResult32 {
	var lowHist [128]uint64
	var highHist [128]uint64
	groupCounts := selectGroupCountsForSample(uint64(len(values)))
	groupHists := make([][]uint64, len(groupCounts))
	for i, gc := range groupCounts {
		groupHists[i] = make([]uint64, gc)
	}

	for _, u := range values {
		h := hashForCandidate32(c, u, seed)
		lowHist[h&0x7f]++
		highHist[(h>>57)&0x7f]++
		for i, gc := range groupCounts {
			bucket := getGroupIndex(h, uint64(gc))
			groupHists[i][int(bucket)]++
		}
	}

	result := rankResult32{candidate: c, samples: uint64(len(values)), seed: seed, groupCounts: len(groupCounts), groupSamples: uint64(len(values))}
	result.lowRelStdPct, result.lowMaxDevPct, result.lowChi2 = computeBucketMetrics(lowHist[:], result.samples)
	result.highRelStdPct, result.highMaxDevPct, result.highChi2 = computeBucketMetrics(highHist[:], result.samples)
	result.groupRelStdMeanPct, result.groupMaxDevMeanPct, result.groupRelStdWorstPct, result.groupMaxDevWorstPct =
		aggregateGroupMappingMetrics(groupHists, result.samples)
	scoreResult(&result)
	return result
}

func evaluateCandidateExhaustive(c rankCandidate32, seed uint64, t *testing.T) rankResult32 {
	const total uint64 = 1 << 32
	const progressStep uint64 = 1 << 29 // log progress every ~536M values

	// Use all group counts directly: with 2^32 samples the expected occupancy
	// per bucket is always >> 128 for every group count in the ranking set, so
	// no filtering via selectGroupCountsForSample is needed.
	groupCounts := getGroupCountsSet3Rank()
	groupHists := make([][]uint64, len(groupCounts))
	for i, gc := range groupCounts {
		groupHists[i] = make([]uint64, gc)
	}

	var lowHist [128]uint64
	var highHist [128]uint64

	for i := uint64(0); i < total; i++ {
		u := uint32(i)
		h := hashForCandidate32(c, u, seed)
		lowHist[h&0x7f]++
		highHist[(h>>57)&0x7f]++ // bits 63–57: top 7 bits
		for gi, gc := range groupCounts {
			groupHists[gi][getGroupIndex(h, uint64(gc))]++
		}
		if i != 0 && i%progressStep == 0 {
			t.Logf("%s: processed %d / %d values (%.2f%%)", c.label(), i, total, 100.0*float64(i)/float64(total))
		}
	}

	result := rankResult32{candidate: c, samples: total, seed: seed, groupCounts: len(groupCounts), groupSamples: total}
	result.lowRelStdPct, result.lowMaxDevPct, result.lowChi2 = computeBucketMetrics(lowHist[:], total)
	result.highRelStdPct, result.highMaxDevPct, result.highChi2 = computeBucketMetrics(highHist[:], total)
	result.groupRelStdMeanPct, result.groupMaxDevMeanPct, result.groupRelStdWorstPct, result.groupMaxDevWorstPct =
		aggregateGroupMappingMetrics(groupHists, total)
	scoreResult(&result)
	return result
}

func evaluateCandidatesConcurrent(candidates []rankCandidate32, workerCount int, eval func(rankCandidate32) rankResult32) []rankResult32 {
	if len(candidates) == 0 {
		return nil
	}
	if workerCount < 1 {
		workerCount = 1
	}
	if workerCount > len(candidates) {
		workerCount = len(candidates)
	}

	results := make([]rankResult32, len(candidates))
	sem := make(chan struct{}, workerCount)
	var wg sync.WaitGroup

	for i, c := range candidates {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, c rankCandidate32) {
			defer wg.Done()
			results[i] = eval(c)
			<-sem
		}(i, c)
	}

	wg.Wait()
	return results
}

func rankResults(results []rankResult32) {
	slices.SortFunc(results, func(a, b rankResult32) int {
		if a.score < b.score {
			return -1
		}
		if a.score > b.score {
			return 1
		}
		if a.groupMaxDevMeanPct < b.groupMaxDevMeanPct {
			return -1
		}
		if a.groupMaxDevMeanPct > b.groupMaxDevMeanPct {
			return 1
		}
		if a.groupRelStdMeanPct < b.groupRelStdMeanPct {
			return -1
		}
		if a.groupRelStdMeanPct > b.groupRelStdMeanPct {
			return 1
		}
		if a.lowMaxDevPct < b.lowMaxDevPct {
			return -1
		}
		if a.lowMaxDevPct > b.lowMaxDevPct {
			return 1
		}
		if a.lowRelStdPct < b.lowRelStdPct {
			return -1
		}
		if a.lowRelStdPct > b.lowRelStdPct {
			return 1
		}
		if a.highMaxDevPct < b.highMaxDevPct {
			return -1
		}
		if a.highMaxDevPct > b.highMaxDevPct {
			return 1
		}
		if a.highRelStdPct < b.highRelStdPct {
			return -1
		}
		if a.highRelStdPct > b.highRelStdPct {
			return 1
		}
		return 0
	})
}

func printRanking(title string, results []rankResult32) {
	if len(results) == 0 {
		fmt.Printf("\n%s\n(no results)\n", title)
		return
	}
	groupCounts := selectGroupCountsForSample(results[0].groupSamples)
	fmt.Printf("\n%s\n", title)
	fmt.Printf("Set3-weighted score = %.2f*lowMaxDev + %.2f*lowRelStd + %.2f*grpMaxMean + %.2f*grpRelMean + %.2f*grpMaxWorst + %.2f*highMaxDev + %.2f*highRelStd\n",
		weightLowMaxDev, weightLowRelStd, weightGroupMaxMean, weightGroupRelMean, weightGroupMaxWorst, weightHighMaxDev, weightHighRelStd)
	fmt.Printf("Group mapping uses getGroupIndex(hash, groupCount) across %d Set3-relevant group counts: %s\n", len(groupCounts), summarizeGroupCounts(groupCounts))
	fmt.Printf("Lower score is better. low7 and group mapping dominate; high7 remains a secondary signal.\n")

	for i, r := range results {
		fmt.Printf("#%02d  %-40s  score=%10.6f  low7(relStd=%8.6f%% maxDev=%8.6f%%)  groupMap[%02d](relMean=%8.6f%% maxMean=%8.6f%% relWorst=%8.6f%% maxWorst=%8.6f%%)  high7(relStd=%8.6f%% maxDev=%8.6f%%)  chi2(low/high)=(%10.6f/%10.6f)\n",
			i+1,
			r.candidate.label(),
			r.score,
			r.lowRelStdPct,
			r.lowMaxDevPct,
			r.groupCounts,
			r.groupRelStdMeanPct,
			r.groupMaxDevMeanPct,
			r.groupRelStdWorstPct,
			r.groupMaxDevWorstPct,
			r.highRelStdPct,
			r.highMaxDevPct,
			r.lowChi2,
			r.highChi2,
		)
	}
}

func summarizeGroupCounts(counts []uint32) string {
	if len(counts) == 0 {
		return "[]"
	}
	if len(counts) <= 24 {
		return fmt.Sprintf("%v", counts)
	}
	return fmt.Sprintf("%v ... %v", counts[:12], counts[len(counts)-12:])
}

func envUint64(name string, fallback uint64) uint64 {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		return fallback
	}
	return v
}

func envInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	if v < 1 {
		return fallback
	}
	return v
}

// hashRankSeedsFromEnv returns ranking seeds from SET3_HASH_RANK_SEEDS.
// Format: comma-separated uint64 values, decimal or base-prefixed (0x...).
// Example: SET3_HASH_RANK_SEEDS="0x1,0x2,42"
// If parsing fails or the variable is empty, defaults are returned.
func hashRankSeedsFromEnv() []uint64 {
	raw := os.Getenv("SET3_HASH_RANK_SEEDS")
	if strings.TrimSpace(raw) == "" {
		out := make([]uint64, len(hashRankDefaultSeeds))
		copy(out, hashRankDefaultSeeds)
		return out
	}

	parts := strings.Split(raw, ",")
	seeds := make([]uint64, 0, len(parts))
	for _, p := range parts {
		token := strings.TrimSpace(p)
		if token == "" {
			continue
		}
		v, err := strconv.ParseUint(token, 0, 64)
		if err != nil {
			out := make([]uint64, len(hashRankDefaultSeeds))
			copy(out, hashRankDefaultSeeds)
			return out
		}
		seeds = append(seeds, v)
	}

	if len(seeds) == 0 {
		out := make([]uint64, len(hashRankDefaultSeeds))
		copy(out, hashRankDefaultSeeds)
		return out
	}

	return seeds
}

// randomUint32SampleDeterministic returns a deterministic uint32 sample stream.
// Determinism matters for ranking, otherwise tiny score differences between close
// candidates can flip from run to run and weaken decision confidence.
func randomUint32SampleDeterministic(n uint64, sampleSeed uint64) []uint32 {
	vals := make([]uint32, n)
	rng := rtcompare.NewDPRNG(sampleSeed)
	for i := uint64(0); i < n; i++ {
		vals[i] = uint32(rng.Uint64())
	}
	return vals
}

// TestHashRank32MixersAndWideningQuickCPRNG is a fast, experimentation-friendly version.
// It uses deterministic uint32 inputs from rtcompare's DPRNG and ranks all
// mixer+constant candidates.
// Override sample count via: SET3_HASH32_QUICK_SAMPLES=<N>.
// Override sample seed via:  SET3_HASH_RANK_SAMPLE_SEED=<u64>.
// Override hash seeds via:   SET3_HASH_RANK_SEEDS="0x...,0x...,...".
//
// Parallelism can be controlled via:
//
//	SET3_HASH32_QUICK_WORKERS=<N>
func TestHashRank32MixersAndWideningQuickCPRNG(t *testing.T) {
	sampleCount := envUint64("SET3_HASH32_QUICK_SAMPLES", quickSampleCountDefault)
	sampleSeed := envUint64("SET3_HASH_RANK_SAMPLE_SEED", hashRankSampleSeedDefault)
	hashSeeds := hashRankSeedsFromEnv()
	if sampleCount < 100_000 {
		t.Fatalf("SET3_HASH32_QUICK_SAMPLES too small (%d); need at least 100000 for a useful signal", sampleCount)
	}

	values := randomUint32SampleDeterministic(sampleCount, sampleSeed)
	candidates := candidateSet32()
	workerCount := envInt("SET3_HASH32_QUICK_WORKERS", runtime.GOMAXPROCS(0))
	workerCount = min(workerCount, len(candidates))
	t.Logf("Quick ranking uses %d workers over %d candidates (sampleSeed=%#x, hashSeeds=%d)", workerCount, len(candidates), sampleSeed, len(hashSeeds))

	seedSums := make([]rankResult32, len(candidates))
	for i, c := range candidates {
		seedSums[i].candidate = c
	}

	for si, seed := range hashSeeds {
		results := evaluateCandidatesConcurrent(candidates, workerCount, func(c rankCandidate32) rankResult32 {
			return evaluateCandidateFromValues(c, values, seed)
		})

		ranked := append([]rankResult32(nil), results...)
		rankResults(ranked)
		printRanking(fmt.Sprintf("QUICK RANKING SEED %d/%d (DPRNG sample, N=%d, hashSeed=%#x, sampleSeed=%#x)", si+1, len(hashSeeds), sampleCount, seed, sampleSeed), ranked)

		for i := range results {
			r := results[i]
			seedSums[i].samples += r.samples
			seedSums[i].groupCounts = r.groupCounts
			seedSums[i].groupSamples += r.groupSamples
			seedSums[i].lowRelStdPct += r.lowRelStdPct
			seedSums[i].lowMaxDevPct += r.lowMaxDevPct
			seedSums[i].highRelStdPct += r.highRelStdPct
			seedSums[i].highMaxDevPct += r.highMaxDevPct
			seedSums[i].lowChi2 += r.lowChi2
			seedSums[i].highChi2 += r.highChi2
			seedSums[i].groupRelStdMeanPct += r.groupRelStdMeanPct
			seedSums[i].groupMaxDevMeanPct += r.groupMaxDevMeanPct
			seedSums[i].groupRelStdWorstPct += r.groupRelStdWorstPct
			seedSums[i].groupMaxDevWorstPct += r.groupMaxDevWorstPct
		}
	}

	combined := make([]rankResult32, len(seedSums))
	div := float64(len(hashSeeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
		combined[i].seed = 0
		combined[i].samples /= uint64(len(hashSeeds))
		combined[i].groupSamples /= uint64(len(hashSeeds))
		combined[i].lowRelStdPct /= div
		combined[i].lowMaxDevPct /= div
		combined[i].highRelStdPct /= div
		combined[i].highMaxDevPct /= div
		combined[i].lowChi2 /= div
		combined[i].highChi2 /= div
		combined[i].groupRelStdMeanPct /= div
		combined[i].groupMaxDevMeanPct /= div
		combined[i].groupRelStdWorstPct /= div
		combined[i].groupMaxDevWorstPct /= div
		scoreResult(&combined[i])
	}

	rankResults(combined)
	printRanking(fmt.Sprintf("QUICK RANKING COMBINED (seeds=%d, N=%d, sampleSeed=%#x)", len(hashSeeds), sampleCount, sampleSeed), combined)

	best := combined[0]
	t.Logf("Best quick candidate across %d seeds: %s (score=%.6f)", len(hashSeeds), best.candidate.label(), best.score)
}

// TestHashRank32MixersAndWideningExhaustive evaluates all candidates over the full uint32 domain.
// This is intentionally heavy. Activate explicitly with:
//
//	SET3_HASH32_EXHAUSTIVE=1 go test -run TestHashRank32MixersAndWideningExhaustive -timeout 0
//
// All metrics — H2 control-byte quality (low7), top-bit sanity (high7), and group mapping
// via getGroupIndex across all Set3-relevant group counts — are evaluated over the full 2^32
// domain in a single pass per candidate. No separate sampling step is needed.
//
// Candidate-level parallelism can be controlled via:
//
//	SET3_HASH32_EXHAUSTIVE_WORKERS=<N>
func TestHashRank32MixersAndWideningExhaustive(t *testing.T) {
	if os.Getenv("SET3_HASH32_EXHAUSTIVE") != "1" {
		t.Skip("set SET3_HASH32_EXHAUSTIVE=1 to run exhaustive 2^32 ranking test")
	}

	hashSeeds := hashRankSeedsFromEnv()
	candidates := candidateSet32()
	defaultWorkers := max(runtime.GOMAXPROCS(0)/2, 1)
	workerCount := envInt("SET3_HASH32_EXHAUSTIVE_WORKERS", defaultWorkers)
	workerCount = min(workerCount, len(candidates))
	t.Logf("Exhaustive ranking uses %d workers over %d candidates and %d hash seeds", workerCount, len(candidates), len(hashSeeds))

	seedSums := make([]rankResult32, len(candidates))
	for i, c := range candidates {
		seedSums[i].candidate = c
	}

	for si, seed := range hashSeeds {
		var nextIdx int
		var idxMu sync.Mutex
		results := evaluateCandidatesConcurrent(candidates, workerCount, func(c rankCandidate32) rankResult32 {
			idxMu.Lock()
			nextIdx++
			idx := nextIdx
			idxMu.Unlock()
			t.Logf("seed %d/%d [%d/%d] evaluating %s over full uint32 range", si+1, len(hashSeeds), idx, len(candidates), c.label())
			return evaluateCandidateExhaustive(c, seed, t)
		})

		ranked := append([]rankResult32(nil), results...)
		rankResults(ranked)
		printRanking(fmt.Sprintf("EXHAUSTIVE RANKING SEED %d/%d (full uint32 domain, seed=%#x)", si+1, len(hashSeeds), seed), ranked)

		for i := range results {
			r := results[i]
			seedSums[i].samples += r.samples
			seedSums[i].groupCounts = r.groupCounts
			seedSums[i].groupSamples += r.groupSamples
			seedSums[i].lowRelStdPct += r.lowRelStdPct
			seedSums[i].lowMaxDevPct += r.lowMaxDevPct
			seedSums[i].highRelStdPct += r.highRelStdPct
			seedSums[i].highMaxDevPct += r.highMaxDevPct
			seedSums[i].lowChi2 += r.lowChi2
			seedSums[i].highChi2 += r.highChi2
			seedSums[i].groupRelStdMeanPct += r.groupRelStdMeanPct
			seedSums[i].groupMaxDevMeanPct += r.groupMaxDevMeanPct
			seedSums[i].groupRelStdWorstPct += r.groupRelStdWorstPct
			seedSums[i].groupMaxDevWorstPct += r.groupMaxDevWorstPct
		}
	}

	combined := make([]rankResult32, len(seedSums))
	div := float64(len(hashSeeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
		combined[i].seed = 0
		combined[i].samples /= uint64(len(hashSeeds))
		combined[i].groupSamples /= uint64(len(hashSeeds))
		combined[i].lowRelStdPct /= div
		combined[i].lowMaxDevPct /= div
		combined[i].highRelStdPct /= div
		combined[i].highMaxDevPct /= div
		combined[i].lowChi2 /= div
		combined[i].highChi2 /= div
		combined[i].groupRelStdMeanPct /= div
		combined[i].groupMaxDevMeanPct /= div
		combined[i].groupRelStdWorstPct /= div
		combined[i].groupMaxDevWorstPct /= div
		scoreResult(&combined[i])
	}

	rankResults(combined)
	printRanking(fmt.Sprintf("EXHAUSTIVE RANKING COMBINED (full uint32 domain, seeds=%d)", len(hashSeeds)), combined)

	best := combined[0]
	t.Logf("Best exhaustive candidate across %d seeds: %s (score=%.6f)", len(hashSeeds), best.candidate.label(), best.score)
}
