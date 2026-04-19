// nolint:gosec // All unsafe operations below are audited: field offsets are computed via reflect at init time.
package hashing

import (
	"reflect"
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

type benchOps4 = genOps4
type benchOps5 = genOps5
type benchOps6 = genOps6
type benchOps7 = genOps7
type benchOps8 = genOps8
type benchRaw12 = rawEligible12
type benchRaw16 = rawEligible16
type benchRaw20 = rawEligible20
type benchRaw24 = rawEligible24
type benchRaw28 = rawEligible28
type benchRaw32 = rawEligible32

func benchmarkHashFunctionDirect(b *testing.B, fn HashFunction, p unsafe.Pointer, seed uint64) {
	b.Helper()
	b.ResetTimer()
	for range b.N {
		benchSink = fn(p, seed)
	}
}

func mustBuildMergedOpsBenchmark(b *testing.B, sample any) []microOp {
	b.Helper()
	ops := flattenStructOps(reflect.TypeOf(sample), 0)
	if ops == nil {
		b.Fatalf("flattenStructOps returned nil for %T", sample)
	}
	return mergeByteBlocks(ops)
}

func benchmarkUnrolledVsLoop(b *testing.B, sample any, valuePtr unsafe.Pointer) {
	b.Helper()
	ops := mustBuildMergedOpsBenchmark(b, sample)
	seed := uint64(0x1234)
	b.Run("unrolled", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, buildClosureFromOps(ops), valuePtr, seed)
	})
	b.Run("loop-baseline", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, buildLoopClosureFromOpsBaseline(ops), valuePtr, seed)
	})
}

func benchmarkFixedBlockVsGeneric(b *testing.B, size int, specialized HashFunction) {
	b.Helper()
	seed := uint64(0x1234)
	var block [32]byte
	for i := range block {
		block[i] = byte(i*29 + 7)
	}
	p := unsafe.Pointer(&block[0])
	generic := func(p unsafe.Pointer, seed uint64) uint64 {
		b := unsafe.Slice((*byte)(p), size) //nolint:gosec
		return HashBytesBlock(seed, b)
	}
	b.Run("specialized", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, specialized, p, seed)
	})
	b.Run("generic", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, generic, p, seed)
	})
}

func benchmarkHashAsByteArrayVsGeneric[K comparable](b *testing.B, value *K) {
	b.Helper()
	seed := uint64(0x1234)
	p := unsafe.Pointer(value)
	b.Run("specialized", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, HashAsByteArray[K], p, seed)
	})
	b.Run("generic", func(b *testing.B) {
		benchmarkHashFunctionDirect(b, hashAsByteArrayGeneric[K], p, seed)
	})
}

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

// BenchmarkHashGen_Unrolled4Ops compares the new 4-op unrolled closure with
// the old loop-based closure shape.
func BenchmarkHashGen_Unrolled4Ops(b *testing.B) {
	v := &benchOps4{A: 1, B: 2.5, C: 3, D: 4.5}
	benchmarkUnrolledVsLoop(b, benchOps4{}, unsafe.Pointer(v))
}

// BenchmarkHashGen_Unrolled5Ops compares the new 5-op unrolled closure with
// the old loop-based closure shape.
func BenchmarkHashGen_Unrolled5Ops(b *testing.B) {
	v := &benchOps5{A: 1, B: 2.5, C: 3, D: 4.5, E: 5}
	benchmarkUnrolledVsLoop(b, benchOps5{}, unsafe.Pointer(v))
}

// BenchmarkHashGen_Unrolled6Ops compares the new 6-op unrolled closure with
// the old loop-based closure shape.
func BenchmarkHashGen_Unrolled6Ops(b *testing.B) {
	v := &benchOps6{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5}
	benchmarkUnrolledVsLoop(b, benchOps6{}, unsafe.Pointer(v))
}

// BenchmarkHashGen_Unrolled7Ops compares the new 7-op unrolled closure with
// the old loop-based closure shape.
func BenchmarkHashGen_Unrolled7Ops(b *testing.B) {
	v := &benchOps7{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5, G: 7}
	benchmarkUnrolledVsLoop(b, benchOps7{}, unsafe.Pointer(v))
}

// BenchmarkHashGen_Unrolled8Ops compares the new 8-op unrolled closure with
// the old loop-based closure shape.
func BenchmarkHashGen_Unrolled8Ops(b *testing.B) {
	v := &benchOps8{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5, G: 7, H: 8.5}
	benchmarkUnrolledVsLoop(b, benchOps8{}, unsafe.Pointer(v))
}

// BenchmarkHashGen_FixedBlock12 compares the dedicated 12-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock12(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 12, hashByteBlock12)
}

// BenchmarkHashGen_FixedBlock16 compares the dedicated 16-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock16(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 16, hashByteBlock16)
}

// BenchmarkHashGen_FixedBlock20 compares the dedicated 20-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock20(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 20, hashByteBlock20)
}

// BenchmarkHashGen_FixedBlock24 compares the dedicated 24-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock24(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 24, hashByteBlock24)
}

// BenchmarkHashGen_FixedBlock28 compares the dedicated 28-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock28(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 28, hashByteBlock28)
}

// BenchmarkHashGen_FixedBlock32 compares the dedicated 32-byte helper with
// the generic HashBytesBlock path.
func BenchmarkHashGen_FixedBlock32(b *testing.B) {
	benchmarkFixedBlockVsGeneric(b, 32, hashByteBlock32)
}

// BenchmarkHashAsByteArray_Raw12 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 12-byte values.
func BenchmarkHashAsByteArray_Raw12(b *testing.B) {
	v := &benchRaw12{A: 1, B: 2, C: 3}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}

// BenchmarkHashAsByteArray_Raw16 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 16-byte values.
func BenchmarkHashAsByteArray_Raw16(b *testing.B) {
	v := &benchRaw16{A: 1, B: 2}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}

// BenchmarkHashAsByteArray_Raw20 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 20-byte values.
func BenchmarkHashAsByteArray_Raw20(b *testing.B) {
	v := &benchRaw20{A: 1, B: 2, C: 3, D: 4, E: 5}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}

// BenchmarkHashAsByteArray_Raw24 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 24-byte values.
func BenchmarkHashAsByteArray_Raw24(b *testing.B) {
	v := &benchRaw24{A: 1, B: 2, C: 3}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}

// BenchmarkHashAsByteArray_Raw28 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 28-byte values.
func BenchmarkHashAsByteArray_Raw28(b *testing.B) {
	v := &benchRaw28{A: 1, B: 2, C: 3, D: 4, E: 5, F: 6, G: 7}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}

// BenchmarkHashAsByteArray_Raw32 compares the specialized raw-byte fast path
// against the old generic HashAsByteArray implementation for 32-byte values.
func BenchmarkHashAsByteArray_Raw32(b *testing.B) {
	v := &benchRaw32{A: 1, B: 2, C: 3, D: 4}
	benchmarkHashAsByteArrayVsGeneric(b, v)
}
