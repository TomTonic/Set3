package quality

import (
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/hashing/alternatives"
	"github.com/TomTonic/Set3/hashing/benchmark"
	"github.com/TomTonic/rtcompare"
)

const maxAvgGroupLoad = 6.5

type hashFunctionsForType struct {
	typesDescr      string                                                      // types for which the hashfunctions are applicable
	implementations []hashing.HashFunction                                      // available hashfunction implementations for the types
	samples         func(limit uint64) <-chan any                               // function that produces sample values of the types (stream via channel)
	allValues       func(limit uint64) <-chan any                               // up to limit values of the according type in ascending order
	randomValues    func(rng rtcompare.DPRNG, limit uint64) <-chan any          // up to limit values generated from the given random source. the values returned might contain duplicates but the sequence of values is deterministic
	apply           func(h hashing.HashFunction, value any, seed uint64) uint64 // helper function that applies a hashfunction to a value of the types
	cardinality     uint64                                                      // the numeber of distinct values a hashfunction should produce for the samples
}

var hashFunctionsForBool = hashFunctionsForType{
	typesDescr:      "b01",
	implementations: []hashing.HashFunction{hashing.HashBool},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 2)
		go func() {
			ch <- false
			ch <- true
			close(ch)
		}()
		return ch
	},
	allValues: func(limit uint64) <-chan any {
		ch := make(chan any, 2)
		go func() {
			b := false
			for range limit {
				ch <- b
				b = !b
			}
			close(ch)
		}()
		return ch
	},
	randomValues: func(rng rtcompare.DPRNG, limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for range limit {
				b := bool((rng.Uint64() & 1) == 1)
				ch <- b
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(bool)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 2,
}

var hashFunctionsForUint8 = hashFunctionsForType{
	typesDescr:      "i08",
	implementations: []hashing.HashFunction{hashing.SwirlByte, alternatives.XorShift64Star08, alternatives.HashI08SM, alternatives.HashI08WH, alternatives.HashI08MH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for i := range 256 {
				ch <- uint8(i)
			}
			close(ch)
		}()
		return ch
	},
	allValues: func(limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for b := range limit {
				ch <- uint8(b)
			}
			close(ch)
		}()
		return ch
	},
	randomValues: func(rng rtcompare.DPRNG, limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			u := rng.Uint64()
			bp := 0
			for range limit {
				b := uint8(u >> bp)
				ch <- b
				bp += 8
				if bp >= 64 {
					bp = 0
					u = rng.Uint64()
				}
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(uint8)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 256,
}

var hashFunctionsForUint16 = hashFunctionsForType{
	typesDescr:      "i16",
	implementations: []hashing.HashFunction{hashing.HashI16SM, alternatives.HashI16WH, alternatives.HashI16MH, alternatives.XorShift64Star16},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			n := limit
			x := benchmark.XorShift16Star{State: rtcompare.NewCPRNG(16).Uint16() | 1} // non-zero random seed
			if n == 0 || n >= 1<<16 {
				n = 1<<16 - 1
			}
			for range n {
				ch <- x.Uint16() // make sure to produce n distinct uint16 values
			}
			if limit == 0 || limit >= 1<<16 {
				ch <- uint16(0) // include 0 if we need 2^16 distinct values
			}
			close(ch)
		}()
		return ch
	},
	allValues: func(limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for b := range limit {
				ch <- uint16(b)
			}
			close(ch)
		}()
		return ch
	},
	randomValues: func(rng rtcompare.DPRNG, limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			u := rng.Uint64()
			bp := 0
			for range limit {
				b := uint16(u >> bp)
				ch <- b
				bp += 16
				if bp >= 64 {
					bp = 0
					u = rng.Uint64()
				}
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(uint16)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 65536,
}

var hashFunctionsForUint32 = hashFunctionsForType{
	typesDescr:      "i32",
	implementations: []hashing.HashFunction{alternatives.HashI32SM, alternatives.HashI32WH, hashing.HashI32WHdet, alternatives.HashI32MH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			n := limit
			x := benchmark.XorShift32Star{State: rtcompare.NewCPRNG(16).Uint32() | 1} // non-zero random seed
			if n == 0 || n >= 1<<32 {
				n = 1<<32 - 1
			}
			for range n {
				ch <- x.Uint32() // make sure to produce n distinct uint32 values
			}
			if limit == 0 || limit >= 1<<32 {
				ch <- uint32(0) // include 0 if we need 2^32 distinct values
			}
			close(ch)
		}()
		return ch
	},
	allValues: func(limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for b := range limit {
				ch <- uint32(b)
			}
			close(ch)
		}()
		return ch
	},
	randomValues: func(rng rtcompare.DPRNG, limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			u := rng.Uint64()
			bp := 0
			for range limit {
				b := uint32(u >> bp)
				ch <- b
				bp += 32
				if bp >= 64 {
					bp = 0
					u = rng.Uint64()
				}
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(uint32)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 1 << 32,
}

var hashFunctionsForUint64 = hashFunctionsForType{
	typesDescr:      "i64",
	implementations: []hashing.HashFunction{alternatives.HashI64SM, alternatives.HashI64WH, hashing.HashI64WHdet, alternatives.HashI64MH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			n := limit
			d := rtcompare.NewDPRNG() // no param -> random seed
			for range n {
				ch <- d.Uint64() // make sure to produce n distinct uint64 values
			}
			close(ch)
		}()
		return ch
	},
	allValues: func(limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for b := range limit {
				ch <- b
			}
			close(ch)
		}()
		return ch
	},
	randomValues: func(rng rtcompare.DPRNG, limit uint64) <-chan any {
		ch := make(chan any, 32)
		go func() {
			for range limit {
				ch <- rng.Uint64()
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(uint64)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 1<<64 - 1,
}

const f32Cardinality = 1<<32 - 16777214 + 1 - 1 // all bit patterns except NaNs (2⋅(2^23−1)=16,777,214) + 1 canonical NaN - 1 for two representations of 0.0

var hashFunctionsForFloat32 = hashFunctionsForType{
	typesDescr:      "f32",
	implementations: []hashing.HashFunction{hashing.HashF32SM, alternatives.HashF32MH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			n := limit
			x := benchmark.XorShift32Star{State: rtcompare.NewCPRNG(16).Uint32() | 1} // non-zero random seed
			if n == 0 || n > f32Cardinality {
				n = f32Cardinality
			}
			for i := uint64(0); i < n; i++ {
				u := x.Uint32() // make sure to produce n distinct uint32 values
				f := math.Float32frombits(u)
				if math.IsNaN(float64(f)) {
					i--
					continue
				}
				if f == 0 {
					// this can only be -0.0 (0x80000000) as xorshift32Star never produces +0.0 (0x00000000)
					// -> emit normalized +0 instead
					f = 0.0
				}
				ch <- f
			}
			if limit == 0 || limit >= f32Cardinality {
				ch <- float32(math.NaN()) // include canonical NaN if we need f32Cardinality distinct values
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(float32)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: f32Cardinality,
}

var hashFunctionsForFloat64 = hashFunctionsForType{
	typesDescr:      "f64",
	implementations: []hashing.HashFunction{alternatives.HashF64SM, alternatives.HashF64MH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			n := limit
			d := rtcompare.NewDPRNG() // no param -> random seed
			for i := uint64(0); i < n; i++ {
				u := d.Uint64()
				f := math.Float64frombits(u)
				if math.IsNaN(f) {
					i--
					continue
				}
				ch <- f
			}
			close(ch)
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(float64)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 0xFFFFFFFFFFFFFFFF - 9_007_199_254_740_990 + 1, // all bit patterns except NaNs (2⋅(2^52−1)=9,007,199,254,740,990) + 1 for 2^64 is not representable as uint64
}

var hashFunctionsForString = hashFunctionsForType{
	typesDescr:      "str",
	implementations: []hashing.HashFunction{hashing.HashString, alternatives.HashStringWH, alternatives.HashStringMH},
	samples: func(limit uint64) <-chan any {
		ch := make(chan any, 8192)
		go func() {
			defer close(ch)
			if limit == 0 {
				return
			}
			ch <- ""
			count := uint64(1)
			for v := range 256 {
				if count >= limit {
					return
				}
				ch <- string([]byte{byte(v)})
				count++
			}
			if count >= limit {
				return
			}
			ch <- string([]byte{0, 0})
			count++
			cprng := rtcompare.NewCPRNG(4096)
			rng16 := benchmark.XorShift16Star{State: cprng.Uint16() | 1} // non-zero random seed
			for range 1<<16 - 1 {
				if count >= limit {
					return
				}
				u16 := rng16.Uint16()
				ch <- string([]byte{byte(byte(u16 >> 8)), byte(u16 & 0xFF)})
				count++
			}
			if count >= limit {
				return
			}
			ch <- string([]byte{0, 0, 0})
			count++
			rng24 := benchmark.XorShift24Star{State: cprng.Uint32() | 1} // non-zero random seed
			for range 1<<24 - 1 {
				if count >= limit {
					return
				}
				u24 := rng24.Uint24()
				ch <- string([]byte{byte(byte(u24 >> 16)), byte(byte(u24 >> 8)), byte(u24 & 0xFF)})
				count++
			}
			if count >= limit {
				return
			}
			ch <- string([]byte{0, 0, 0, 0})
			count++
			rng32 := benchmark.XorShift32Star{State: cprng.Uint32() | 1} // non-zero random seed
			for range limit - count {
				// length between 0 and 100 inclusive
				additional_length := cprng.Uint32N(96)
				u32 := rng32.Uint32()
				buffer := make([]byte, 4+additional_length)
				buffer[0] = byte(u32 >> 24)
				buffer[1] = byte(u32 >> 16)
				buffer[2] = byte(u32 >> 8)
				buffer[3] = byte(u32 & 0xFF)
				for l := range additional_length {
					b := cprng.Uint8()
					buffer[4+l] = b
				}
				s := string(buffer)
				ch <- s
			}
		}()
		return ch
	},
	apply: func(h hashing.HashFunction, value any, seed uint64) uint64 {
		v := value.(string)
		return h(unsafe.Pointer(&v), seed)
	},
	cardinality: 1 + 1<<8 + 1<<16 + 1<<24 + 1<<32, // all strings of length 0..3 + strings of length 4..100 with first 4 bytes all combinations and remaining bytes random
}

var hashFunctionsForAllTypes = []hashFunctionsForType{
	hashFunctionsForBool,
	hashFunctionsForUint8,
	hashFunctionsForUint16,
	hashFunctionsForUint32,
	hashFunctionsForUint64,
	hashFunctionsForFloat32,
	hashFunctionsForFloat64,
	hashFunctionsForString,
}

func TestDeterminismOfHashFunctionsForAllTypes(t *testing.T) {
	seeds := []uint64{0, 1, 0x8000000000000000, 0x12345678abcdef, 0x7fffffffffffffff, 0xfffffffffffffffe, 0xffffffffffffffff}
	const maxIterationsPerTypeAndFunction = 1 << 20
	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			for _, seed := range seeds {

				// derive short function name from the function pointer
				fn := runtime.FuncForPC(reflect.ValueOf(hf).Pointer()).Name()
				shortname := fn
				if idx := strings.LastIndex(shortname, "."); idx != -1 {
					shortname = shortname[idx+1:]
				}

				t.Run(e.typesDescr+"/"+shortname+"/seed="+fmt.Sprintf("%x", seed), func(t *testing.T) {
					expected := min(e.cardinality, maxIterationsPerTypeAndFunction)
					// determinism
					for s := range e.samples(expected) {
						a := e.apply(hf, s, seed)
						b := e.apply(hf, s, seed)
						if a != b {
							t.Fatalf("%s: non-deterministic result for %s with seed %x: %x != %x", shortname, e.typesDescr, seed, a, b)
						}
					}
				})
			}
		}
	}
}

func TestDistinctOutputsOfHashFunctionsForAllTypes(t *testing.T) {
	seeds := []uint64{0, 1, 0x8000000000000000, 0x12345678abcdef, 0x7fffffffffffffff, 0xfffffffffffffffe, 0xffffffffffffffff}
	const maxIterationsPerTypeAndFunction = 1 << 20
	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			for _, seed := range seeds {

				// derive short function name from the function pointer
				fn := runtime.FuncForPC(reflect.ValueOf(hf).Pointer()).Name()
				shortname := fn
				if idx := strings.LastIndex(shortname, "."); idx != -1 {
					shortname = shortname[idx+1:]
				}

				t.Run(e.typesDescr+"/"+shortname+"/seed="+fmt.Sprintf("%x", seed), func(t *testing.T) {
					expected := min(e.cardinality, maxIterationsPerTypeAndFunction)
					seen := make(map[uint64]struct{}, expected)
					cnt := uint64(0)
					for s := range e.samples(expected) {
						hv := e.apply(hf, s, seed)
						if _, exists := seen[hv]; exists {
							t.Fatalf("%s: duplicate hash value %x for %s after %d samples, seed %x", shortname, hv, e.typesDescr, cnt, seed)
						}
						seen[hv] = struct{}{}
						cnt++
					}
					if cnt != expected {
						t.Fatalf("%s: expected to process %d samples for %s (seed %x), but processed %d", shortname, expected, e.typesDescr, seed, cnt)
					}
					if uint64(len(seen)) != expected {
						t.Fatalf("%s: expected %d distinct hashes for %s (seed %x), got %d", shortname, expected, e.typesDescr, seed, len(seen))
					}
				})
			}
		}
	}
}

func TestUniformityLowest7BitsOfHashFunctionsForAllTypes(t *testing.T) {
	seeds := []uint64{
		0x0000000000000000, 0x0000000000000001, 0x8000000000000000,
		0xffffffffffffffff, 0xfffffffffffffffe, 0x7fffffffffffffff,
		0x1111111111111111, 0x8888888888888888, 0x9999999999999999,
		0x1234567890abcdef, 0x96C0FFEEC0FFEE69, 0x0000DEADC0DE0000, 0xDEAD00000000C0DE,
		0x3f7a1c9e2b4d5f80, 0x8e1a9b2c3d4f5a61, 0xa1b2c3d4e5f60789,
		0x7f6e5d4c3b2a1908, 0x91ab2c3d4e5f6072, 0x5f2e9a1b3c4d6e7f}
	const maxIterationsPerTypeAndFunction = 1 << 19
	const numBuckets = 128
	const stdDevFailureThreshold = 1.0
	const maxRelDevFailureThreshold = 3.0
	resultsRelStdDev, resultsMaxRelDev := prepareResultMaps()

	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			shortname := functionName(hf)
			for _, seed := range seeds {
				t.Run(e.typesDescr+"/"+shortname+"/seed="+fmt.Sprintf("%x", seed), func(t *testing.T) {
					expected := min(e.cardinality, maxIterationsPerTypeAndFunction)
					bucketsUniformityTestBit0ToBit7 := make([]int, numBuckets)
					for s := range e.samples(expected) {
						hv := e.apply(hf, s, seed)
						bucketsUniformityTestBit0ToBit7[hv&0x7F]++
					}

					if expected >= numBuckets {
						mean := float64(expected) / float64(numBuckets)
						var sumsq float64
						var maxDev float64
						maxDevIdx := -1
						for i, c := range bucketsUniformityTestBit0ToBit7 {
							d := float64(c) - mean
							sumsq += d * d
							if math.Abs(d)/mean > maxDev {
								maxDev = math.Abs(d) / mean
								maxDevIdx = i
							}
						}
						std := math.Sqrt(sumsq / float64(numBuckets))
						relStdDev := std / mean
						if relStdDev > stdDevFailureThreshold {
							t.Fatalf("%s: low distribution for %s (seed %x) in bit 0..7 - not uniform enough: relative standard deviation=%.3f", shortname, e.typesDescr, seed, relStdDev)
						}
						// Also ensure no single bucket deviates excessively (e.g. > 45%). This
						// guards against pathological clustering even if the overall stddev is ok.
						if maxDev > maxRelDevFailureThreshold {
							t.Fatalf("%s: a bucket deviated too much from mean: maxRelDev=%.6f, index=%d/%d", shortname, maxDev, maxDevIdx, numBuckets)
						}
						// store results for later
						resultsRelStdDev[shortname] = append(resultsRelStdDev[shortname], relStdDev)
						resultsMaxRelDev[shortname] = append(resultsMaxRelDev[shortname], maxDev)
					}
				})
			}
		}
	}
	logResults(t, resultsRelStdDev, resultsMaxRelDev)
}

func logResults(t *testing.T, resultsRelStdDev map[string][]float64, resultsMaxRelDev map[string][]float64) {
	names := make([]string, 0, len(resultsRelStdDev))
	for name := range resultsRelStdDev {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		stddevs := resultsRelStdDev[name]
		if len(stddevs) == 0 {
			continue
		}
		sort.Float64s(stddevs)
		medRelStdDev := rtcompare.QuickMedian(stddevs)
		t.Logf("%s: relStdDev median=%.6f, values={%.6f..%.6f}", name, medRelStdDev, stddevs[0], stddevs[len(stddevs)-1])
	}
	for _, name := range names {
		maxdevs := resultsMaxRelDev[name]
		if len(maxdevs) == 0 {
			continue
		}
		sort.Float64s(maxdevs)
		medMaxRelDev := rtcompare.QuickMedian(maxdevs)
		t.Logf("%s: maxRelDev median=%.6f, values={%.6f..%.6f}", name, medMaxRelDev, maxdevs[0], maxdevs[len(maxdevs)-1])
	}
}

// prepareResultMaps initializes and returns two maps keyed by short function names.
// The maps are populated with an entry for every hash function implementation
// discovered in hashFunctionsForAllTypes (using functionName(hf) as the key).
// The first returned map is intended to store per test result standard deviations,
// and the second to store per test maximum relative deviations.
func prepareResultMaps() (map[string][]float64, map[string][]float64) {
	resultsRelStdDev := make(map[string][]float64)
	resultsMaxRelDev := make(map[string][]float64)
	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			shortname := functionName(hf)
			resultsRelStdDev[shortname] = []float64{}
			resultsMaxRelDev[shortname] = []float64{}
		}
	}
	return resultsRelStdDev, resultsMaxRelDev
}

// functionName returns the short name of the function value hf.
// It looks up hf's runtime function name via reflect.ValueOf(hf).Pointer()
// and runtime.FuncForPC, then strips the package/path prefix by returning
// the substring after the final '.'; if no '.' is found the full name is returned.
// hf is expected to be a function value of type hashing.HashFunction.
func functionName(hf hashing.HashFunction) string {
	fn := runtime.FuncForPC(reflect.ValueOf(hf).Pointer()).Name()
	shortname := fn
	if idx := strings.LastIndex(shortname, "."); idx != -1 {
		shortname = shortname[idx+1:]
	}
	return shortname
}

func TestBucketMappingOfHashFunctionsForAllTypes(t *testing.T) {
	const numExperimentsPerGroup = 500
	const numGroupSizes = 37
	const groupSizeMultiplier = 512
	const stdDevFailureThreshold = 1.0
	const maxRelDevFailureThreshold = 3.0
	resultsRelStdDev, resultsMaxRelDev := prepareResultMaps()

	for _, hft := range hashFunctionsForAllTypes {
		for _, hf := range hft.implementations {
			shortname := functionName(hf)
			rng := rtcompare.NewDPRNG(123456)
			t.Run(hft.typesDescr+"/"+shortname, func(t *testing.T) {
				for numGroups := range benchmark.FilteredNumbers(numGroupSizes) {
					if float64(numGroups) > float64(hft.cardinality)/maxAvgGroupLoad {
						t.Skipf("Hashing %d elements would never need %d groups or more.", hft.cardinality, numGroups)
					}
					elementsInSet := min(numGroups*groupSizeMultiplier, hft.cardinality)
					for range numExperimentsPerGroup {
						buckets := make([]uint32, numGroups)
						seed := rng.Uint64()
						for s := range hft.samples(elementsInSet) {
							hv := hft.apply(hf, s, seed)
							groupIndex := getGroupIndex(hv, numGroups)
							buckets[groupIndex]++
						}
						relStdDev, maxDev := computeRelStdDevAndMaxDev(buckets)

						if relStdDev > stdDevFailureThreshold {
							t.Fatalf("%s: low distribution for %s with numGroups=%d - not uniform enough: relative standard deviation=%.3f", shortname, hft.typesDescr, numGroups, relStdDev)
						}
						// Also ensure no single bucket deviates excessively (e.g. > 45%). This
						// guards against pathological clustering even if the overall stddev is ok.
						if maxDev > maxRelDevFailureThreshold {
							t.Fatalf("%s: a bucket deviated too much from mean: maxRelDev=%.6f", shortname, maxDev)
						}
						// store results for later
						resultsRelStdDev[shortname] = append(resultsRelStdDev[shortname], relStdDev)
						resultsMaxRelDev[shortname] = append(resultsMaxRelDev[shortname], maxDev)
					}
				}
			})
		}
	}
	logResults(t, resultsRelStdDev, resultsMaxRelDev)
}

func TestAvalancheEffectOfHashFunctionsForAllTypes(t *testing.T) {
	seeds := []uint64{
		0x0000000000000000, 0x0000000000000001, 0x8000000000000000,
		0xffffffffffffffff, 0xfffffffffffffffe, 0x7fffffffffffffff,
		0x1234567890abcdef, 0x96C0FFEEC0FFEE69,
	}
	const maxIterationsPerTypeAndFunction = 1 << 17 // keep runtime reasonable
	const minComparisonsToTest = 4096

	// z=6 => threshold ~= 3/sqrt(n) for per-output-bit flip probability around 0.5
	const z = 6.0
	const minAbsDev = 0.03 // avoid being too strict for smaller n

	resultsMaxAbsDev := make(map[string][]float64)
	resultsAvgFlippedBits := make(map[string][]float64)
	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			shortname := functionName(hf)
			resultsMaxAbsDev[shortname] = nil
			resultsAvgFlippedBits[shortname] = nil
		}
	}

	for _, e := range hashFunctionsForAllTypes {
		for _, hf := range e.implementations {
			shortname := functionName(hf)
			for _, seed := range seeds {
				t.Run(e.typesDescr+"/"+shortname+"/seed="+fmt.Sprintf("%x", seed), func(t *testing.T) {
					expected := min(e.cardinality, maxIterationsPerTypeAndFunction)

					// Determine numOfBitsInType from first encountered sample type.
					numOfBitsInType := -1
					var outBitFlipCounts [64]uint64
					var totalPairs uint64
					var sumFlippedBits uint64

					bitsToFlip := []uint{0} // placeholder; filled once we know bitSize

					for s := range e.samples(maxIterationsPerTypeAndFunction) {

						if bs, ok := bitSizeOfSample(s); ok {
							numOfBitsInType = bs
							bitsToFlip = avalancheInputBits(numOfBitsInType)
						} else {
							t.Skipf("avalanche test: unsupported sample type %T", s)
						}
						if numOfBitsInType < 2 {
							//t.Skip("avalanche test not meaningful for <2 input bits")
							continue
						}

						h0 := e.apply(hf, s, seed)
						for _, bit := range bitsToFlip {
							s2, ok := flipBitSample(s, bit)
							if !ok {
								continue
							}
							h1 := e.apply(hf, s2, seed)
							diff := h0 ^ h1

							totalPairs++
							sumFlippedBits += uint64(bits.OnesCount64(diff))
							for ob := 0; ob < 64; ob++ {
								outBitFlipCounts[ob] += (diff >> ob) & 1
							}
						}
					}

					_ = expected // keep symmetry with other tests; expected influences sample count above.

					if totalPairs < minComparisonsToTest {
						t.Skipf("too few comparisons for avalanche test: %d (need >= %d)", totalPairs, minComparisonsToTest)
					}

					n := float64(totalPairs)
					threshold := math.Max(minAbsDev, (z*0.5)/math.Sqrt(n)) // ~= 3/sqrt(n)

					maxAbsDev := 0.0
					for ob := range 64 {
						p := float64(outBitFlipCounts[ob]) / n
						absDev := math.Abs(p - 0.5)
						if absDev > maxAbsDev {
							maxAbsDev = absDev
						}
						if absDev > threshold {
							t.Logf("weak avalanche: outBit=%d flipProb=%.4f absDev=%.4f threshold=%.4f (pairs=%d)", ob, p, absDev, threshold, totalPairs)
							//t.Fatalf("weak avalanche: outBit=%d flipProb=%.4f absDev=%.4f threshold=%.4f (pairs=%d)", ob, p, absDev, threshold, totalPairs)
						}
					}

					avgFlippedBits := float64(sumFlippedBits) / n
					// Heuristic sanity check: average changed output bits should be near 32.
					// Allow a fairly wide band to avoid flakiness while still catching obvious issues.
					if math.Abs(avgFlippedBits-32.0) > 6.0 {
						t.Fatalf("%s: weak avalanche: avgFlippedBits=%.3f (expected near 32) pairs=%d", shortname, avgFlippedBits, totalPairs)
					}

					resultsMaxAbsDev[shortname] = append(resultsMaxAbsDev[shortname], maxAbsDev)
					resultsAvgFlippedBits[shortname] = append(resultsAvgFlippedBits[shortname], avgFlippedBits)
				})
			}
		}
	}

	logAvalancheResults(t, resultsMaxAbsDev, resultsAvgFlippedBits)
}

// avalancheInputBits returns a small, representative set of input bit indices for avalanche testing.
// The returned slice contains unique uint indices in ascending order, each within [0, bitSize-1].
// If bitSize <= 0 the function returns nil.
// The set includes the low-order bits 0..min(7, bitSize-1), the middle bit (bitSize/2), and the top bit (bitSize-1).
// Duplicate indices are suppressed and the result is sorted in increasing order.
// The goal is to cover a range of input bits at typical positions without excessive test runtime.
func avalancheInputBits(bitSize int) []uint {
	// A small but representative set: low bits plus the top bit.
	if bitSize <= 0 {
		return nil
	}
	seen := make(map[uint]struct{}, 16)
	var out []uint
	add := func(b uint) {
		if int(b) < 0 || int(b) >= bitSize {
			return
		}
		if _, ok := seen[b]; ok {
			return
		}
		seen[b] = struct{}{}
		out = append(out, b)
	}
	for b := 0; b < bitSize && b < 8; b++ {
		add(uint(b))
	}
	add(uint(bitSize / 2))
	add(uint(bitSize - 1))
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func bitSizeOfSample(v any) (int, bool) {
	switch x := v.(type) {
	case bool:
		return 1, true

	// signed ints
	case int8:
		return 8, true
	case int16:
		return 16, true
	case int32:
		return 32, true
	case int64:
		return 64, true
	case int:
		return bits.UintSize, true

	// unsigned ints
	case uint8:
		return 8, true
	case uint16:
		return 16, true
	case uint32:
		return 32, true
	case uint64:
		return 64, true
	case uint:
		return bits.UintSize, true
	case uintptr:
		return bits.UintSize, true

	// floats
	case float32:
		return 32, true
	case float64:
		return 64, true

	// complex
	case complex64:
		return 64, true // 2x float32
	case complex128:
		return 128, true // 2x float64

	// string (bytes)
	case string:
		return len(x) * 8, true

	default:
		return 0, false
	}
}

func flipBitSample(v any, bit uint) (any, bool) {
	switch x := v.(type) {
	case bool:
		if bit != 0 {
			return nil, false
		}
		return !x, true

	// signed ints (flip in two's complement domain)
	case int8:
		if bit >= 8 {
			return nil, false
		}
		u := uint8(x)
		u ^= uint8(1) << bit
		return int8(u), true
	case int16:
		if bit >= 16 {
			return nil, false
		}
		u := uint16(x)
		u ^= uint16(1) << bit
		return int16(u), true
	case int32:
		if bit >= 32 {
			return nil, false
		}
		u := uint32(x)
		u ^= uint32(1) << bit
		return int32(u), true
	case int64:
		if bit >= 64 {
			return nil, false
		}
		u := uint64(x)
		u ^= uint64(1) << bit
		return int64(u), true
	case int:
		if bit >= uint(bits.UintSize) {
			return nil, false
		}
		u := uint(x)
		u ^= uint(1) << bit
		return int(u), true

	// unsigned ints
	case uint8:
		if bit >= 8 {
			return nil, false
		}
		return x ^ (uint8(1) << bit), true
	case uint16:
		if bit >= 16 {
			return nil, false
		}
		return x ^ (uint16(1) << bit), true
	case uint32:
		if bit >= 32 {
			return nil, false
		}
		return x ^ (uint32(1) << bit), true
	case uint64:
		if bit >= 64 {
			return nil, false
		}
		return x ^ (uint64(1) << bit), true
	case uint:
		if bit >= uint(bits.UintSize) {
			return nil, false
		}
		return x ^ (uint(1) << bit), true
	case uintptr:
		if bit >= uint(bits.UintSize) {
			return nil, false
		}
		return x ^ (uintptr(1) << bit), true

	// floats (flip IEEE bits)
	case float32:
		if bit >= 32 {
			return nil, false
		}
		u := math.Float32bits(x)
		u ^= uint32(1) << bit
		return math.Float32frombits(u), true
	case float64:
		if bit >= 64 {
			return nil, false
		}
		u := math.Float64bits(x)
		u ^= uint64(1) << bit
		return math.Float64frombits(u), true

	// complex (map bits across real then imag)
	case complex64:
		if bit >= 64 {
			return nil, false
		}
		r := float32(real(x))
		i := float32(imag(x))
		if bit < 32 {
			ru := math.Float32bits(r) ^ (uint32(1) << bit)
			r = math.Float32frombits(ru)
		} else {
			ib := bit - 32
			iu := math.Float32bits(i) ^ (uint32(1) << ib)
			i = math.Float32frombits(iu)
		}
		return complex(r, i), true
	case complex128:
		if bit >= 128 {
			return nil, false
		}
		r := real(x)
		i := imag(x)
		if bit < 64 {
			ru := math.Float64bits(r) ^ (uint64(1) << bit)
			r = math.Float64frombits(ru)
		} else {
			ib := bit - 64
			iu := math.Float64bits(i) ^ (uint64(1) << ib)
			i = math.Float64frombits(iu)
		}
		return complex(r, i), true

	// string (flip a bit in the UTF-8 bytes, returning a new string)
	case string:
		nBits := uint(len(x) * 8)
		if bit >= nBits {
			return nil, false
		}
		bs := []byte(x)
		byteIdx := bit / 8
		bitInByte := bit % 8
		bs[byteIdx] ^= byte(1) << bitInByte
		return string(bs), true

	default:
		return nil, false
	}
}

func logAvalancheResults(t *testing.T, resultsMaxAbsDev map[string][]float64, resultsAvgFlippedBits map[string][]float64) {
	names := make([]string, 0, len(resultsMaxAbsDev))
	for name := range resultsMaxAbsDev {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		v := resultsAvgFlippedBits[name]
		if len(v) == 0 {
			continue
		}
		sort.Float64s(v)
		t.Logf("%s: avgFlippedBits median=%.6f, values={%.6f..%.6f}", name, rtcompare.QuickMedian(v), v[0], v[len(v)-1])
	}
	for _, name := range names {
		v := resultsMaxAbsDev[name]
		if len(v) == 0 {
			continue
		}
		sort.Float64s(v)
		t.Logf("%s: maxAbsDevFrom0.5 median=%.6f, values={%.6f..%.6f}", name, rtcompare.QuickMedian(v), v[0], v[len(v)-1])
	}
}
