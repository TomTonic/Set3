package quality

// rank_multi_test.go — ranking tests for float32, int16, int64, float64.
//
// Each test follows the same methodology as TestHashRank32MixersAndWideningExhaustive:
//   - Enumerate all unique values of the type and hash each exactly once.
//   - Measure low-7-bit (H2 control byte) quality, top-7-bit quality, and
//     group-mapping quality via getGroupIndex across Set3-relevant group counts.
//   - Produce a ranked score table identical in format to the 32-bit ranking.
//
// Activation:
//
//	SET3_HASH_EXHAUSTIVE_MULTI=1 go test -run "TestHashRankF32|TestHashRankI16|TestHashRankI64|TestHashRankF64" -timeout 0
//
// For int64/float64 the domain is 2^64 – a full exhaustive scan is not feasible.
// Those tests use a large CPRNG sample instead; see their individual doc comments.

import (
	"fmt"
	"math"
	"math/bits"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/hashing/alternatives"
	"github.com/TomTonic/Set3/hashing/benchmarks"
	"github.com/TomTonic/rtcompare"
)

// ─────────────────────────────────────────────────────────────────────────────
// Generic infrastructure shared by all four tests
// ─────────────────────────────────────────────────────────────────────────────

type rankCandidateM struct {
	label  string
	isProd bool
	hash   func(raw uint64, seed uint64) uint64 // receives the canonical bit pattern
}

type rankResultM struct {
	label  string
	isProd bool

	samples      uint64
	groupCounts  int
	groupSamples uint64

	lowRelStdPct  float64
	lowMaxDevPct  float64
	highRelStdPct float64
	highMaxDevPct float64
	lowChi2       float64
	highChi2      float64

	groupRelStdMeanPct  float64
	groupMaxDevMeanPct  float64
	groupRelStdWorstPct float64
	groupMaxDevWorstPct float64

	score float64
}

func scoreResultM(r *rankResultM) {
	r.score =
		weightLowMaxDev*r.lowMaxDevPct +
			weightLowRelStd*r.lowRelStdPct +
			weightGroupMaxMean*r.groupMaxDevMeanPct +
			weightGroupRelMean*r.groupRelStdMeanPct +
			weightGroupMaxWorst*r.groupMaxDevWorstPct +
			weightHighMaxDev*r.highMaxDevPct +
			weightHighRelStd*r.highRelStdPct
}

func evaluateCandidateMFromBits(c rankCandidateM, rawBits []uint64, seed uint64) rankResultM {
	var lowHist [128]uint64
	var highHist [128]uint64
	groupCounts := selectGroupCountsForSample(uint64(len(rawBits)))
	groupHists := make([][]uint64, len(groupCounts))
	for i, gc := range groupCounts {
		groupHists[i] = make([]uint64, gc)
	}

	for _, raw := range rawBits {
		h := c.hash(raw, seed)
		lowHist[h&0x7f]++
		highHist[(h>>57)&0x7f]++
		for i, gc := range groupCounts {
			groupHists[i][getGroupIndex(h, uint64(gc))]++
		}
	}

	r := rankResultM{
		label:        c.label,
		isProd:       c.isProd,
		samples:      uint64(len(rawBits)),
		groupCounts:  len(groupCounts),
		groupSamples: uint64(len(rawBits)),
	}
	r.lowRelStdPct, r.lowMaxDevPct, r.lowChi2 = computeBucketMetrics(lowHist[:], r.samples)
	r.highRelStdPct, r.highMaxDevPct, r.highChi2 = computeBucketMetrics(highHist[:], r.samples)
	r.groupRelStdMeanPct, r.groupMaxDevMeanPct, r.groupRelStdWorstPct, r.groupMaxDevWorstPct =
		aggregateGroupMappingMetrics(groupHists, r.samples)
	scoreResultM(&r)
	return r
}

// evaluateCandidateMExhaustive iterates uint64(0)..total-1, converts each index
// to a raw bit pattern via toRaw, and hashes it. Used for fully enumerable types.
func evaluateCandidateMExhaustive(c rankCandidateM, total uint64, toRaw func(uint64) uint64, seed uint64, t *testing.T) rankResultM {
	const progressStep uint64 = 1 << 29

	groupCounts := getGroupCountsSet3Rank()
	groupHists := make([][]uint64, len(groupCounts))
	for i, gc := range groupCounts {
		groupHists[i] = make([]uint64, gc)
	}

	var lowHist [128]uint64
	var highHist [128]uint64

	for i := uint64(0); i < total; i++ {
		raw := toRaw(i)
		h := c.hash(raw, seed)
		lowHist[h&0x7f]++
		highHist[(h>>57)&0x7f]++
		for gi, gc := range groupCounts {
			groupHists[gi][getGroupIndex(h, uint64(gc))]++
		}
		if i != 0 && i%progressStep == 0 {
			t.Logf("%s: processed %d / %d (%.2f%%)", c.label, i, total, 100.0*float64(i)/float64(total))
		}
	}

	r := rankResultM{
		label:        c.label,
		isProd:       c.isProd,
		samples:      total,
		groupCounts:  len(groupCounts),
		groupSamples: total,
	}
	r.lowRelStdPct, r.lowMaxDevPct, r.lowChi2 = computeBucketMetrics(lowHist[:], total)
	r.highRelStdPct, r.highMaxDevPct, r.highChi2 = computeBucketMetrics(highHist[:], total)
	r.groupRelStdMeanPct, r.groupMaxDevMeanPct, r.groupRelStdWorstPct, r.groupMaxDevWorstPct =
		aggregateGroupMappingMetrics(groupHists, total)
	scoreResultM(&r)
	return r
}

// evaluateCandidateMFromChan reads canonical bit patterns from ch and hashes each.
// total is the expected number of values (used for group-count selection and metrics).
func evaluateCandidateMFromChan(c rankCandidateM, ch <-chan uint64, total uint64, seed uint64, t *testing.T) rankResultM {
	const progressStep uint64 = 1 << 29

	// For sample-based runs, cap group counts to keep expected occupancy per bucket
	// meaningful; otherwise maxDev is dominated by sparse-sampling noise.
	groupCounts := selectGroupCountsForSample(total)
	groupHists := make([][]uint64, len(groupCounts))
	for i, gc := range groupCounts {
		groupHists[i] = make([]uint64, gc)
	}

	var lowHist [128]uint64
	var highHist [128]uint64
	var count uint64

	for raw := range ch {
		h := c.hash(raw, seed)
		lowHist[h&0x7f]++
		highHist[(h>>57)&0x7f]++
		for gi, gc := range groupCounts {
			groupHists[gi][getGroupIndex(h, uint64(gc))]++
		}
		count++
		if count != 0 && count%progressStep == 0 {
			t.Logf("%s: processed %d / %d (%.2f%%)", c.label, count, total, 100.0*float64(count)/float64(total))
		}
	}

	r := rankResultM{
		label:        c.label,
		isProd:       c.isProd,
		samples:      count,
		groupCounts:  len(groupCounts),
		groupSamples: count,
	}
	r.lowRelStdPct, r.lowMaxDevPct, r.lowChi2 = computeBucketMetrics(lowHist[:], count)
	r.highRelStdPct, r.highMaxDevPct, r.highChi2 = computeBucketMetrics(highHist[:], count)
	r.groupRelStdMeanPct, r.groupMaxDevMeanPct, r.groupRelStdWorstPct, r.groupMaxDevWorstPct =
		aggregateGroupMappingMetrics(groupHists, count)
	scoreResultM(&r)
	return r
}

func evaluateCandidatesMConcurrent(candidates []rankCandidateM, workerCount int, eval func(rankCandidateM) rankResultM) []rankResultM {
	if len(candidates) == 0 {
		return nil
	}
	if workerCount < 1 {
		workerCount = 1
	}
	if workerCount > len(candidates) {
		workerCount = len(candidates)
	}
	results := make([]rankResultM, len(candidates))
	sem := make(chan struct{}, workerCount)
	var wg sync.WaitGroup
	for i, c := range candidates {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, c rankCandidateM) {
			defer wg.Done()
			results[i] = eval(c)
			<-sem
		}(i, c)
	}
	wg.Wait()
	return results
}

func rankResultsM(results []rankResultM) {
	// Stable sort: primary = score, then group quality, then low quality.
	for i := 1; i < len(results); i++ {
		for j := i; j > 0; j-- {
			a, b := results[j-1], results[j]
			less := a.score > b.score ||
				(a.score == b.score && a.groupMaxDevMeanPct > b.groupMaxDevMeanPct) ||
				(a.score == b.score && a.groupMaxDevMeanPct == b.groupMaxDevMeanPct && a.lowMaxDevPct > b.lowMaxDevPct)
			if less {
				results[j-1], results[j] = results[j], results[j-1]
			} else {
				break
			}
		}
	}
}

func printRankingM(title string, results []rankResultM) {
	if len(results) == 0 {
		fmt.Printf("\n%s\n(no results)\n", title)
		return
	}
	groupCounts := selectGroupCountsForSample(results[0].groupSamples)
	fmt.Printf("\n%s\n", title)
	fmt.Printf("Set3-weighted score = %.2f*lowMaxDev + %.2f*lowRelStd + %.2f*grpMaxMean + %.2f*grpRelMean + %.2f*grpMaxWorst + %.2f*highMaxDev + %.2f*highRelStd\n",
		weightLowMaxDev, weightLowRelStd, weightGroupMaxMean, weightGroupRelMean, weightGroupMaxWorst, weightHighMaxDev, weightHighRelStd)
	fmt.Printf("Group mapping: getGroupIndex across %d group counts: %s\n", len(groupCounts), summarizeGroupCounts(groupCounts))
	for i, r := range results {
		prod := " "
		if r.isProd {
			prod = "*"
		}
		fmt.Printf("#%02d%s %-52s  score=%10.6f  low7(relStd=%8.6f%% maxDev=%8.6f%%)  groupMap[%02d](relMean=%8.6f%% maxMean=%8.6f%% relWorst=%8.6f%% maxWorst=%8.6f%%)  high7(relStd=%8.6f%% maxDev=%8.6f%%)  chi2(low/high)=(%10.6f/%10.6f)\n",
			i+1, prod, r.label, r.score,
			r.lowRelStdPct, r.lowMaxDevPct,
			r.groupCounts,
			r.groupRelStdMeanPct, r.groupMaxDevMeanPct,
			r.groupRelStdWorstPct, r.groupMaxDevWorstPct,
			r.highRelStdPct, r.highMaxDevPct,
			r.lowChi2, r.highChi2,
		)
	}
	fmt.Printf("  (* = current production function)\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// Channel generators: canonical float values
// ─────────────────────────────────────────────────────────────────────────────

const (
	canonicalNaNF32 uint64 = 0x7fc00000
	canonicalNaNF64 uint64 = 0x7ff8000000000000
)

// isSpecialF32 reports whether bits represents ±0 or any NaN.
func isSpecialF32(bits uint32) bool {
	if bits == 0 || bits == 0x80000000 {
		return true // ±0
	}
	return math.IsNaN(float64(math.Float32frombits(bits)))
}

// isSpecialF64 reports whether bits represents ±0 or any NaN.
func isSpecialF64(bits uint64) bool {
	if bits == 0 || bits == 1<<63 {
		return true // ±0
	}
	return math.IsNaN(math.Float64frombits(bits))
}

// canonicalF32Values returns a channel that emits every canonical float32 value
// exactly once as a uint64 bit pattern:
//  1. 0x00000000 (canonical +0.0)
//  2. 0x7fc00000 (canonical NaN)
//  3. all 32-bit patterns that are neither ±0 nor any NaN, iterated via xorshift32*
//     so the order is pseudo-random but the full set is covered exactly.
//
// Total emitted: (1<<32) - (1<<24) + 2  =  4,278,190,082
func canonicalF32Values() <-chan uint64 {
	ch := make(chan uint64, 1<<16)
	go func() {
		defer close(ch)
		// 1. canonical zero
		ch <- 0
		// 2. canonical NaN
		ch <- canonicalNaNF32
		// 3. all non-special values via xorshift32* (period 2^32-1, never produces 0)
		x := benchmarks.XorShift32Star{State: rtcompare.NewCPRNG(16).Uint32() | 1}
		for i := uint64(0); i < (1<<32)-1; i++ {
			u := x.Uint32()
			if !isSpecialF32(u) {
				ch <- uint64(u)
			}
		}
	}()
	return ch
}

// canonicalF64Values returns a channel that emits canonical float64 values as uint64:
//  1. 0x0000000000000000 (canonical +0.0)
//  2. 0x7ff8000000000000 (canonical NaN)
//  3. up to sampleCount non-special values drawn from a deterministic DPRNG.
//
// Total emitted: sampleCount + 2
func canonicalF64Values(sampleCount uint64, sampleSeed uint64) <-chan uint64 {
	ch := make(chan uint64, 8192)
	go func() {
		defer close(ch)
		// 1. canonical zero
		ch <- 0
		// 2. canonical NaN
		ch <- canonicalNaNF64
		// 3. DPRNG-sampled non-special values
		rng := rtcompare.NewDPRNG(sampleSeed)
		emitted := uint64(0)
		for emitted < sampleCount {
			u := rng.Uint64()
			if !isSpecialF64(u) {
				ch <- u
				emitted++
			}
		}
	}()
	return ch
}

// ─────────────────────────────────────────────────────────────────────────────
// Generator correctness tests
// ─────────────────────────────────────────────────────────────────────────────

// TestCanonicalF32ValuesGenerator checks the channel generator contract.
// Without SET3_HASH_EXHAUSTIVE_MULTI=1 it only verifies the first two values
// (canonical zero and canonical NaN) and drains the rest in the background.
// With SET3_HASH_EXHAUSTIVE_MULTI=1 it exhaustively verifies the full count.
func TestCanonicalF32ValuesGenerator(t *testing.T) {
	ch := canonicalF32Values()

	v0, ok0 := <-ch
	if !ok0 || v0 != 0 {
		t.Fatalf("first value: got %#x ok=%v, want 0x00000000", v0, ok0)
	}
	v1, ok1 := <-ch
	if !ok1 || v1 != canonicalNaNF32 {
		t.Fatalf("second value: got %#x ok=%v, want %#x", v1, ok1, canonicalNaNF32)
	}

	if os.Getenv("SET3_HASH_EXHAUSTIVE_MULTI") != "1" {
		// Drain the channel in the background to avoid leaking the goroutine.
		go func() {
			for range ch {
			}
		}()
		t.Skip("set SET3_HASH_EXHAUSTIVE_MULTI=1 to run exhaustive check")
	}

	// Exhaustive: count the remaining non-special values and verify none are special.
	const expectedNonSpecial uint64 = (1 << 32) - (1 << 24)
	count := uint64(0)
	for u := range ch {
		if isSpecialF32(uint32(u)) {
			t.Fatalf("generator emitted special value %#x", u)
		}
		count++
	}
	if count != expectedNonSpecial {
		t.Fatalf("non-special count: got %d, want %d", count, expectedNonSpecial)
	}
	t.Logf("canonicalF32Values: emitted 2 + %d = %d values (correct)", count, count+2)
}

// TestCanonicalF64ValuesGenerator verifies the float64 generator contract.
func TestCanonicalF64ValuesGenerator(t *testing.T) {
	const sampleCount = 100_000
	ch := canonicalF64Values(sampleCount, hashRankSampleSeedDefault)

	v0, ok0 := <-ch
	if !ok0 || v0 != 0 {
		t.Fatalf("first value: got %#x ok=%v, want 0", v0, ok0)
	}
	v1, ok1 := <-ch
	if !ok1 || v1 != canonicalNaNF64 {
		t.Fatalf("second value: got %#x ok=%v, want %#x", v1, ok1, canonicalNaNF64)
	}

	count := uint64(0)
	for u := range ch {
		if isSpecialF64(u) {
			t.Fatalf("generator emitted special value %#x", u)
		}
		count++
	}
	if count != sampleCount {
		t.Fatalf("non-special count: got %d, want %d", count, sampleCount)
	}
	t.Logf("canonicalF64Values: emitted 2 + %d = %d values (correct)", count, count+2)
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) float32 — exhaustive over all distinct canonical float32 values
// ─────────────────────────────────────────────────────────────────────────────
//
// float32 has 2^32 distinct bit patterns, but only (1<<32)-(1<<24)+2 distinct
// canonical values after normalization (±0 collapse to one zero; all NaN patterns
// collapse to one canonical NaN). The channel generator covers all of them exactly.
//
// Production: hashF32SM (= splitmix64 + goldenRatio32 widening) [PROD-SM]
//             hashF32WHdet (= wh32det) [PROD-WH]

func candidateSetF32() []rankCandidateM {
	return []rankCandidateM{
		// ── splitmix64 variants ─────────────────────────────────────────────
		{
			label:  "splitmix64 + goldenRatio32 [PROD-SM]",
			isProd: true,
			hash: func(raw, seed uint64) uint64 {
				v := hashing.GoldenRatio32 * raw
				return hashing.Splitmix64(seed ^ v)
			},
		},
		{
			label: "splitmix64 + sqrt2_1_32",
			hash: func(raw, seed uint64) uint64 {
				v := hashing.Sqrt2_1_32 * raw
				return hashing.Splitmix64(seed ^ v)
			},
		},
		{
			label: "splitmix64 + pie7_32",
			hash: func(raw, seed uint64) uint64 {
				v := hashing.Pie7_32 * raw
				return hashing.Splitmix64(seed ^ v)
			},
		},
		{
			label: "splitmix64 + replication_0x0000000100000001",
			hash: func(raw, seed uint64) uint64 {
				v := uint64(0x0000_0001_0000_0001) * raw
				return hashing.Splitmix64(seed ^ v)
			},
		},
		{
			label: "splitmix64 + largestPrime32",
			hash: func(raw, seed uint64) uint64 {
				v := uint64(0x0000_0000_FFFF_FFFB) * raw
				return hashing.Splitmix64(seed ^ v)
			},
		},
		// ── wh32detExtMul variants ───────────────────────────────────────────
		{
			label:  "wh32detExtMul + goldenRatio32 [PROD-WH]",
			isProd: true,
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH32DetExtMul(hashing.GoldenRatio32*raw, seed)
			},
		},
		{
			label: "wh32detExtMul + sqrt2_1_32",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH32DetExtMul(hashing.Sqrt2_1_32*raw, seed)
			},
		},
		{
			label: "wh32detExtMul + pie7_32",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH32DetExtMul(hashing.Pie7_32*raw, seed)
			},
		},
		{
			label: "wh32detExtMul + replication_0x0000000100000001",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH32DetExtMul(uint64(0x0000_0001_0000_0001)*raw, seed)
			},
		},
		{
			label: "wh32detExtMul + largestPrime32",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH32DetExtMul(uint64(0x0000_0000_FFFF_FFFB)*raw, seed)
			},
		},
	}
}

// TestHashRankF32Exhaustive evaluates all candidates over all distinct canonical
// float32 values ((1<<32)-(1<<24)+2 total).
// Activate with:
//
//	SET3_HASH_EXHAUSTIVE_MULTI=1 go test -run TestHashRankF32Exhaustive -timeout 0
//
// Parallelism: SET3_HASH_MULTI_WORKERS=<N>
func TestHashRankF32Exhaustive(t *testing.T) {
	if os.Getenv("SET3_HASH_EXHAUSTIVE_MULTI") != "1" {
		t.Skip("set SET3_HASH_EXHAUSTIVE_MULTI=1 to run")
	}

	const total uint64 = (1 << 32) - (1 << 24) + 2
	seeds := hashRankSeedsFromEnv()
	candidates := candidateSetF32()
	defaultWorkers := max(runtime.GOMAXPROCS(0)/2, 1)
	workerCount := envInt("SET3_HASH_MULTI_WORKERS", defaultWorkers)
	workerCount = min(workerCount, len(candidates))
	t.Logf("float32 exhaustive: %d workers, %d candidates, %d canonical values, %d seeds", workerCount, len(candidates), total, len(seeds))

	seedSums := make([]rankResultM, len(candidates))
	for i, c := range candidates {
		seedSums[i].label = c.label
		seedSums[i].isProd = c.isProd
	}

	for si, seed := range seeds {
		var nextIdx int
		var idxMu sync.Mutex
		results := evaluateCandidatesMConcurrent(candidates, workerCount, func(c rankCandidateM) rankResultM {
			idxMu.Lock()
			nextIdx++
			idx := nextIdx
			idxMu.Unlock()
			t.Logf("seed %d/%d [%d/%d] %s", si+1, len(seeds), idx, len(candidates), c.label)
			return evaluateCandidateMFromChan(c, canonicalF32Values(), total, seed, t)
		})

		ranked := append([]rankResultM(nil), results...)
		rankResultsM(ranked)
		printRankingM(fmt.Sprintf("FLOAT32 EXHAUSTIVE SEED %d/%d (all canonical values, seed=%#x)", si+1, len(seeds), seed), ranked)

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

	combined := make([]rankResultM, len(seedSums))
	div := float64(len(seeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
		combined[i].samples /= uint64(len(seeds))
		combined[i].groupSamples /= uint64(len(seeds))
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
		scoreResultM(&combined[i])
	}

	rankResultsM(combined)
	printRankingM(fmt.Sprintf("FLOAT32 EXHAUSTIVE COMBINED (all canonical values, seeds=%d)", len(seeds)), combined)
	t.Logf("Best across %d seeds: %s (score=%.6f)", len(seeds), combined[0].label, combined[0].score)
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) int16 — exhaustive over all 2^16 values
// ─────────────────────────────────────────────────────────────────────────────
//
// int16 has only 65536 distinct values, so a full exhaustive pass is trivially fast.
//
// Production: hashI16SM (= splitmix64 + spread16to64=pie7_48) [PROD]

func candidateSetI16() []rankCandidateM {
	return []rankCandidateM{
		// ── splitmix64 variants ─────────────────────────────────────────────
		{
			label:  "splitmix64 + pie7_48=spread16to64 [PROD]",
			isProd: true,
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.Pie7_48 * raw))
			},
		},
		{
			label: "splitmix64 + goldenRatio48",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.GoldenRatio48 * raw))
			},
		},
		{
			label: "splitmix64 + sqrt2_1_48",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.Sqrt2_1_48 * raw))
			},
		},
		{
			label: "splitmix64 + replication_0x0001000100010001",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (uint64(0x0001_0001_0001_0001) * raw))
			},
		},
		// ── wh16detExtMul variants ───────────────────────────────────────────
		{
			label: "wh16detExtMul + replication_0x0001000100010001 [WH-structural]",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH16DetExtMul(uint64(0x0001_0001_0001_0001)*raw, seed)
			},
		},
		{
			label: "wh16detExtMul + pie7_48",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH16DetExtMul(hashing.Pie7_48*raw, seed)
			},
		},
		{
			label: "wh16detExtMul + goldenRatio48",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH16DetExtMul(hashing.GoldenRatio48*raw, seed)
			},
		},
		{
			label: "wh16detExtMul + sqrt2_1_48",
			hash: func(raw, seed uint64) uint64 {
				return alternatives.WH16DetExtMul(hashing.Sqrt2_1_48*raw, seed)
			},
		},
	}
}

// TestHashRankI16Exhaustive evaluates all candidates over all 2^16 int16 values.
func TestHashRankI16Exhaustive(t *testing.T) {
	const total uint64 = 1 << 16
	seeds := hashRankSeedsFromEnv()
	candidates := candidateSetI16()
	workerCount := min(runtime.GOMAXPROCS(0), len(candidates))
	t.Logf("int16 exhaustive: %d workers, %d candidates, %d values, %d seeds", workerCount, len(candidates), total, len(seeds))

	seedSums := make([]rankResultM, len(candidates))
	for i, c := range candidates {
		seedSums[i].label = c.label
		seedSums[i].isProd = c.isProd
	}

	for si, seed := range seeds {
		var nextIdx int
		var idxMu sync.Mutex
		results := evaluateCandidatesMConcurrent(candidates, workerCount, func(c rankCandidateM) rankResultM {
			idxMu.Lock()
			nextIdx++
			idx := nextIdx
			idxMu.Unlock()
			t.Logf("seed %d/%d [%d/%d] %s", si+1, len(seeds), idx, len(candidates), c.label)
			return evaluateCandidateMExhaustive(c, total, func(i uint64) uint64 { return i }, seed, t)
		})

		ranked := append([]rankResultM(nil), results...)
		rankResultsM(ranked)
		printRankingM(fmt.Sprintf("INT16 EXHAUSTIVE SEED %d/%d (all 2^16 values, seed=%#x)", si+1, len(seeds), seed), ranked)

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

	combined := make([]rankResultM, len(seedSums))
	div := float64(len(seeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
		combined[i].samples /= uint64(len(seeds))
		combined[i].groupSamples /= uint64(len(seeds))
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
		scoreResultM(&combined[i])
	}

	rankResultsM(combined)
	printRankingM(fmt.Sprintf("INT16 EXHAUSTIVE COMBINED (all 2^16 values, seeds=%d)", len(seeds)), combined)
	t.Logf("Best across %d seeds: %s (score=%.6f)", len(seeds), combined[0].label, combined[0].score)
}

// ─────────────────────────────────────────────────────────────────────────────
// (c) int64 — sample-based (2^64 exhaustive is infeasible)
// ─────────────────────────────────────────────────────────────────────────────
//
// Production: hashI64WHdet (= wh64det) [PROD]
// Override sample count: SET3_HASH64_SAMPLES=<N> (default 10_000_000).

const hash64SamplesDefault uint64 = 10_000_000

// wh64detLambda wraps wh64det as a func(raw, seed uint64) uint64 for the generic framework.
func wh64detLambda(raw, seed uint64) uint64 {
	return hashing.WH64Det(raw, seed)
}

func candidateSetI64() []rankCandidateM {
	return []rankCandidateM{
		{
			label:  "wh64det RotateLeft32 [PROD]",
			isProd: true,
			hash:   wh64detLambda,
		},
		{
			label: "splitmix64",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ raw)
			},
		},
		{
			label: "wh64-like: XOR-swap (a ^ a>>32, a << 32)",
			hash: func(raw, seed uint64) uint64 {
				a := raw
				b := bits.RotateLeft64(a^(a>>32), 16)
				c := a ^ hashing.P1
				d := b ^ seed
				e := hashing.Mix(c, d)
				return hashing.Mix(hashing.M5^8, e)
			},
		},
	}
}

// TestHashRankI64Sample evaluates all candidates over a large CPRNG sample of int64 values.
// Activate with:
//
//	SET3_HASH_EXHAUSTIVE_MULTI=1 go test -run TestHashRankI64Sample -timeout 60s
//
// Override sample size: SET3_HASH64_SAMPLES=<N>
func TestHashRankI64Sample(t *testing.T) {
	if os.Getenv("SET3_HASH_EXHAUSTIVE_MULTI") != "1" {
		t.Skip("set SET3_HASH_EXHAUSTIVE_MULTI=1 to run")
	}

	sampleCount := envUint64("SET3_HASH64_SAMPLES", hash64SamplesDefault)
	hashSeeds := hashRankSeedsFromEnv()

	// Deterministic sample generation keeps candidate ranking stable between runs.
	sampleSeed := envUint64("SET3_HASH_RANK_SAMPLE_SEED", hashRankSampleSeedDefault)
	rng := rtcompare.NewDPRNG(sampleSeed)
	rawBits := make([]uint64, sampleCount)
	for i := range rawBits {
		rawBits[i] = rng.Uint64()
	}

	candidates := candidateSetI64()
	workerCount := min(envInt("SET3_HASH_MULTI_WORKERS", runtime.GOMAXPROCS(0)), len(candidates))
	t.Logf("int64 sample: %d workers, %d candidates, %d values (sampleSeed=%#x, hashSeeds=%d)", workerCount, len(candidates), sampleCount, sampleSeed, len(hashSeeds))

	seedSums := make([]rankResultM, len(candidates))
	for i, c := range candidates {
		seedSums[i].label = c.label
		seedSums[i].isProd = c.isProd
	}

	for si, seed := range hashSeeds {
		results := evaluateCandidatesMConcurrent(candidates, workerCount, func(c rankCandidateM) rankResultM {
			return evaluateCandidateMFromBits(c, rawBits, seed)
		})

		ranked := append([]rankResultM(nil), results...)
		rankResultsM(ranked)
		printRankingM(fmt.Sprintf("INT64 SAMPLE SEED %d/%d (N=%d DPRNG values, hashSeed=%#x, sampleSeed=%#x)", si+1, len(hashSeeds), sampleCount, seed, sampleSeed), ranked)

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

	combined := make([]rankResultM, len(seedSums))
	div := float64(len(hashSeeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
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
		scoreResultM(&combined[i])
	}

	rankResultsM(combined)
	printRankingM(fmt.Sprintf("INT64 SAMPLE COMBINED (N=%d values, seeds=%d, sampleSeed=%#x)", sampleCount, len(hashSeeds), sampleSeed), combined)
	t.Logf("Best across %d seeds: %s (score=%.6f)", len(hashSeeds), combined[0].label, combined[0].score)
}

// ─────────────────────────────────────────────────────────────────────────────
// (d) float64 — sample-based
// ─────────────────────────────────────────────────────────────────────────────
//
// Production: hashF64SM (= splitmix64(seed ^ bits)) [PROD]
// Values are drawn from canonicalF64Values which emits canonical zero, canonical NaN,
// then CPRNG-sampled non-special bit patterns.
//
// Override sample count: SET3_HASH64_SAMPLES=<N> (default 10_000_000).

func candidateSetF64() []rankCandidateM {
	return []rankCandidateM{
		{
			label:  "splitmix64 + identity (no widening) [PROD]",
			isProd: true,
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ raw)
			},
		},
		{
			label: "splitmix64 + goldenRatio64 pre-multiply",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.GoldenRatio64 * raw))
			},
		},
		{
			label: "splitmix64 + sqrt2_1_64 pre-multiply",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.Sqrt2_1_64 * raw))
			},
		},
		{
			label: "splitmix64 + pie7_64 pre-multiply",
			hash: func(raw, seed uint64) uint64 {
				return hashing.Splitmix64(seed ^ (hashing.Pie7_64 * raw))
			},
		},
		{
			label: "wh64det (no pre-multiply)",
			hash:  wh64detLambda,
		},
		{
			label: "wh64det + goldenRatio64 pre-multiply",
			hash: func(raw, seed uint64) uint64 {
				return wh64detLambda(hashing.GoldenRatio64*raw, seed)
			},
		},
		{
			label: "wh64det + sqrt2_1_64 pre-multiply",
			hash: func(raw, seed uint64) uint64 {
				return wh64detLambda(hashing.Sqrt2_1_64*raw, seed)
			},
		},
	}
}

// TestHashRankF64Sample evaluates all candidates over a sample of canonical float64 values.
// Activate with:
//
//	SET3_HASH_EXHAUSTIVE_MULTI=1 go test -run TestHashRankF64Sample -timeout 60s
//
// Override sample size: SET3_HASH64_SAMPLES=<N>
func TestHashRankF64Sample(t *testing.T) {
	if os.Getenv("SET3_HASH_EXHAUSTIVE_MULTI") != "1" {
		t.Skip("set SET3_HASH_EXHAUSTIVE_MULTI=1 to run")
	}

	sampleCount := envUint64("SET3_HASH64_SAMPLES", hash64SamplesDefault)
	// Shared sample seed with 32-bit ranking tests to make comparisons reproducible.
	sampleSeed := envUint64("SET3_HASH_RANK_SAMPLE_SEED", hashRankSampleSeedDefault)
	hashSeeds := hashRankSeedsFromEnv()
	total := sampleCount + 2 // +2 for canonical zero and canonical NaN

	candidates := candidateSetF64()
	workerCount := min(envInt("SET3_HASH_MULTI_WORKERS", runtime.GOMAXPROCS(0)), len(candidates))
	t.Logf("float64 sample: %d workers, %d candidates, %d values (incl. zero and NaN, sampleSeed=%#x, hashSeeds=%d)", workerCount, len(candidates), total, sampleSeed, len(hashSeeds))

	seedSums := make([]rankResultM, len(candidates))
	for i, c := range candidates {
		seedSums[i].label = c.label
		seedSums[i].isProd = c.isProd
	}

	for si, seed := range hashSeeds {
		results := evaluateCandidatesMConcurrent(candidates, workerCount, func(c rankCandidateM) rankResultM {
			return evaluateCandidateMFromChan(c, canonicalF64Values(sampleCount, sampleSeed), total, seed, t)
		})

		ranked := append([]rankResultM(nil), results...)
		rankResultsM(ranked)
		printRankingM(fmt.Sprintf("FLOAT64 SAMPLE SEED %d/%d (N=%d canonical values, hashSeed=%#x, sampleSeed=%#x)", si+1, len(hashSeeds), total, seed, sampleSeed), ranked)

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

	combined := make([]rankResultM, len(seedSums))
	div := float64(len(hashSeeds))
	for i := range seedSums {
		combined[i] = seedSums[i]
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
		scoreResultM(&combined[i])
	}

	rankResultsM(combined)
	printRankingM(fmt.Sprintf("FLOAT64 SAMPLE COMBINED (N=%d canonical values, seeds=%d, sampleSeed=%#x)", total, len(hashSeeds), sampleSeed), combined)
	t.Logf("Best across %d seeds: %s (score=%.6f)", len(hashSeeds), combined[0].label, combined[0].score)
}
