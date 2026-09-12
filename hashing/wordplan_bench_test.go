package hashing

import (
	"fmt"
	"reflect"
	"testing"
	"unsafe"
)

// These benchmarks compare the two closures the generator can build for one
// type: the word-mixing path and the seed-threading chain it replaced. Both
// sides are built by the production builders from the same flattened ops, so
// neither is a hand-written stand-in and the comparison cannot drift from what
// the library actually emits.
//
// The chain remains the shipping path for every type the word plan declines,
// so this is not a museum piece — it is a live comparison of two live paths.

var wordBenchSink uint64

// bothClosures builds the word path and the chain for the same type. It fails
// the benchmark if the type does not qualify for the word path, because then
// the comparison would silently be chain against chain.
func bothClosures(tb testing.TB, ty reflect.Type) (words, chain HashFunction) {
	tb.Helper()
	ops := mergeByteBlocks(flattenTypeOps(ty, 0))
	words = buildWordClosure(ops)
	if words == nil {
		tb.Fatalf("%s does not take the word path, so there is nothing to compare", ty)
	}
	return words, buildClosureFromOps(ops)
}

// benchWordPathVsChain runs both closures over the same value. The value is
// passed as a pointer the caller owns, so nothing here allocates.
func benchWordPathVsChain(b *testing.B, ty reflect.Type, p unsafe.Pointer) {
	words, chain := bothClosures(b, ty)
	const seed = 0x243f6a8885a308d3

	// A cheap equality check first: a benchmark that measured two routines
	// computing different things would be meaningless, and these two are
	// allowed to differ in value — but each must at least depend on the input.
	if words(p, seed) == words(p, seed+1) {
		b.Fatalf("%s word closure ignores the seed", ty)
	}

	b.Run("chain", func(b *testing.B) {
		var acc uint64
		for range b.N {
			acc ^= chain(p, seed)
		}
		wordBenchSink ^= acc
	})
	b.Run("words", func(b *testing.B) {
		var acc uint64
		for range b.N {
			acc ^= words(p, seed)
		}
		wordBenchSink ^= acc
	})
}

func BenchmarkWordPathFloat64Fields(b *testing.B) {
	for _, n := range []int{2, 3, 4, 6, 8} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			// An array of float64 flattens to n float64 ops, exactly as a
			// struct of n float64 fields does, and one type covers every n.
			ty := reflect.ArrayOf(n, reflect.TypeOf(float64(0)))
			v := reflect.New(ty)
			for i := range n {
				v.Elem().Index(i).SetFloat(float64(i) + 1.5)
			}
			benchWordPathVsChain(b, ty, v.UnsafePointer())
		})
	}
}

func BenchmarkWordPathFloat32Fields(b *testing.B) {
	for _, n := range []int{3, 4, 6, 8} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			ty := reflect.ArrayOf(n, reflect.TypeOf(float32(0)))
			v := reflect.New(ty)
			for i := range n {
				v.Elem().Index(i).SetFloat(float64(i) + 0.25)
			}
			benchWordPathVsChain(b, ty, v.UnsafePointer())
		})
	}
}

func BenchmarkWordPathIntAndFloat(b *testing.B) {
	v := wpBlock16Float{A: 11, B: 22, F: 33.5}
	benchWordPathVsChain(b, reflect.TypeOf(v), unsafe.Pointer(&v))
}

func BenchmarkWordPathComplex128(b *testing.B) {
	v := wpComplex2{A: complex(1.5, -2.5), B: complex(3, 4)}
	benchWordPathVsChain(b, reflect.TypeOf(v), unsafe.Pointer(&v))
}

func BenchmarkWordPathFloatsThenString(b *testing.B) {
	v := wpFloatString{Lat: 48.2, Lon: 16.37, Name: "vienna"}
	benchWordPathVsChain(b, reflect.TypeOf(v), unsafe.Pointer(&v))
}

// BenchmarkWordPathThroughTheHasher measures what a caller actually pays: the
// generated closure reached through RuntimeHasher, indirect call included.
func BenchmarkWordPathThroughTheHasher(b *testing.B) {
	h := MakeRuntimeHasher[wpFloat3](0x243f6a8885a308d3)
	v := wpFloat3{1.5, 2.5, 3.5}
	for range b.N {
		wordBenchSink ^= h.Hash(v)
	}
}
