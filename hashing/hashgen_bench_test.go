// nolint:gosec // All unsafe operations below are audited: field offsets are computed via reflect at init time.
package hashing

import (
	"testing"
	"unsafe"
)

// ── Benchmark types ─────────────────────────────────────────────────────────

// benchPaddingStruct has inter-field padding (uint8 → uint64 alignment).
type benchPaddingStruct struct {
	A uint8
	B uint64
}

// benchFloatStruct has a float64 requiring canonicalization.
type benchFloatStruct struct {
	X int32
	F float64
}

// benchStringStruct contains a string field.
type benchStringStruct struct {
	N int32
	S string
}

// benchMixed has scalar, float, string, and complex fields.
type benchMixed struct {
	A uint64
	B float32
	C string
	D complex64
}

// benchLargeStruct has many fields of mixed types.
type benchLargeStruct struct {
	A int64
	B float64
	C int32
	D float32
	E string
	F uint64
	G complex128
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// sink prevents dead-code elimination.
var benchSink uint64

// ── Benchmarks ──────────────────────────────────────────────────────────────

// BenchmarkHashGen_PaddingStruct measures the generated hasher for a struct
// with inter-field padding.
func BenchmarkHashGen_PaddingStruct(b *testing.B) {
	h := MakeRuntimeHasher[benchPaddingStruct](0x1234)
	v := benchPaddingStruct{A: 42, B: 12345678}
	b.ResetTimer()
	for range b.N {
		benchSink = h.Hash(v)
	}
}

// BenchmarkHashFallback_PaddingStruct measures HashFallbackMaphash for the
// same struct to establish a baseline.
func BenchmarkHashFallback_PaddingStruct(b *testing.B) {
	seed := uint64(0x1234)
	v := benchPaddingStruct{A: 42, B: 12345678}
	b.ResetTimer()
	for range b.N {
		benchSink = HashFallbackMaphash[benchPaddingStruct](unsafe.Pointer(&v), seed)
	}
}

// BenchmarkHashGen_FloatStruct measures the generated hasher for a struct
// with a float64 field.
func BenchmarkHashGen_FloatStruct(b *testing.B) {
	h := MakeRuntimeHasher[benchFloatStruct](0x1234)
	v := benchFloatStruct{X: 7, F: 3.14159}
	b.ResetTimer()
	for range b.N {
		benchSink = h.Hash(v)
	}
}

func BenchmarkHashFallback_FloatStruct(b *testing.B) {
	seed := uint64(0x1234)
	v := benchFloatStruct{X: 7, F: 3.14159}
	b.ResetTimer()
	for range b.N {
		benchSink = HashFallbackMaphash[benchFloatStruct](unsafe.Pointer(&v), seed)
	}
}

// BenchmarkHashGen_StringStruct measures the generated hasher for a struct
// with a string field.
func BenchmarkHashGen_StringStruct(b *testing.B) {
	h := MakeRuntimeHasher[benchStringStruct](0x1234)
	v := benchStringStruct{N: 99, S: "hello world benchmark"}
	b.ResetTimer()
	for range b.N {
		benchSink = h.Hash(v)
	}
}

func BenchmarkHashFallback_StringStruct(b *testing.B) {
	seed := uint64(0x1234)
	v := benchStringStruct{N: 99, S: "hello world benchmark"}
	b.ResetTimer()
	for range b.N {
		benchSink = HashFallbackMaphash[benchStringStruct](unsafe.Pointer(&v), seed)
	}
}

// BenchmarkHashGen_Mixed measures the generated hasher for a struct with
// scalar, float, string, and complex fields.
func BenchmarkHashGen_Mixed(b *testing.B) {
	h := MakeRuntimeHasher[benchMixed](0x1234)
	v := benchMixed{A: 1, B: 3.14, C: "test", D: complex(1, 2)}
	b.ResetTimer()
	for range b.N {
		benchSink = h.Hash(v)
	}
}

func BenchmarkHashFallback_Mixed(b *testing.B) {
	seed := uint64(0x1234)
	v := benchMixed{A: 1, B: 3.14, C: "test", D: complex(1, 2)}
	b.ResetTimer()
	for range b.N {
		benchSink = HashFallbackMaphash[benchMixed](unsafe.Pointer(&v), seed)
	}
}

// BenchmarkHashGen_Large measures the generated hasher for a large struct.
func BenchmarkHashGen_Large(b *testing.B) {
	h := MakeRuntimeHasher[benchLargeStruct](0x1234)
	v := benchLargeStruct{A: 1, B: 2.718, C: 3, D: 1.414, E: "benchmark", F: 42, G: complex(1, 2)}
	b.ResetTimer()
	for range b.N {
		benchSink = h.Hash(v)
	}
}

func BenchmarkHashFallback_Large(b *testing.B) {
	seed := uint64(0x1234)
	v := benchLargeStruct{A: 1, B: 2.718, C: 3, D: 1.414, E: "benchmark", F: 42, G: complex(1, 2)}
	b.ResetTimer()
	for range b.N {
		benchSink = HashFallbackMaphash[benchLargeStruct](unsafe.Pointer(&v), seed)
	}
}

// BenchmarkHashGen_NativeMap_FloatStruct uses Go's native map to measure
// the built-in hash function cost (via map insert/lookup overhead).
func BenchmarkHashGen_NativeMap_FloatStruct(b *testing.B) {
	m := make(map[benchFloatStruct]struct{}, 1)
	v := benchFloatStruct{X: 7, F: 3.14159}
	m[v] = struct{}{}
	b.ResetTimer()
	for range b.N {
		_, _ = m[v]
	}
}

// BenchmarkHashGen_NativeMap_Mixed uses Go's native map for the mixed struct.
func BenchmarkHashGen_NativeMap_Mixed(b *testing.B) {
	m := make(map[benchMixed]struct{}, 1)
	v := benchMixed{A: 1, B: 3.14, C: "test", D: complex(1, 2)}
	m[v] = struct{}{}
	b.ResetTimer()
	for range b.N {
		_, _ = m[v]
	}
}

// BenchmarkHashGen_Direct_FloatStruct measures the generated function
// called directly (no RuntimeHasher.Hash overhead/Noescape).
func BenchmarkHashGen_Direct_FloatStruct(b *testing.B) {
	seed := uint64(0x1234)
	v := benchFloatStruct{X: 7, F: 3.14159}
	h := MakeRuntimeHasher[benchFloatStruct](seed)
	b.ResetTimer()
	for range b.N {
		p := unsafe.Pointer(&v)
		benchSink = h.fn(p, seed)
	}
}
