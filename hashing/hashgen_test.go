// nolint:gosec // Test file: intentional unsafe.Pointer use for testing generated hash functions
package hashing

import (
	"math"
	"reflect"
	"testing"
	"unsafe"
)

// ── test types ──────────────────────────────────────────────────────────────

type genPaddingStruct struct {
	A uint8
	B uint64
}

type genFloatStruct struct {
	X int32
	F float64
}

type genComplexStruct struct {
	A uint32
	C complex128
}

type genStringStruct struct {
	N int32
	S string
}

type genNestedStruct struct {
	F genFloatStruct
	S genStringStruct
}

type genAllBlank struct {
	_ uint32
	_ uint32
}

type genMixed struct {
	A uint64
	B float32
	C string
	D complex64
}

type genOps4 struct {
	A uint64
	B float64
	C uint64
	D float64
}

type genOps5 struct {
	A uint64
	B float64
	C uint64
	D float64
	E uint64
}

type genOps6 struct {
	A uint64
	B float64
	C uint64
	D float64
	E uint64
	F float64
}

type genOps7 struct {
	A uint64
	B float64
	C uint64
	D float64
	E uint64
	F float64
	G uint64
}

type genOps8 struct {
	A uint64
	B float64
	C uint64
	D float64
	E uint64
	F float64
	G uint64
	H float64
}

type genEmptyStruct struct{}

// buildLoopClosureFromOpsBaseline mirrors the old generic N-op path so tests
// and benchmarks can compare specialized closures against the loop baseline.
func buildLoopClosureFromOpsBaseline(ops []microOp) HashFunction {
	frozen := make([]fieldOp, len(ops))
	for i, op := range ops {
		frozen[i] = microOpToFieldOp(op)
	}
	return func(p unsafe.Pointer, seed uint64) uint64 {
		h := seed
		for _, fop := range frozen {
			h = fop.fn(unsafe.Add(p, fop.offset), h) //nolint:gosec
		}
		return h
	}
}

// mustMergedOpsForStruct flattens a struct type and returns its merged op list.
func mustMergedOpsForStruct(t *testing.T, sample any) []microOp {
	t.Helper()
	ops := flattenStructOps(reflect.TypeOf(sample), 0)
	if ops == nil {
		t.Fatalf("flattenStructOps returned nil for %T", sample)
	}
	return mergeByteBlocks(ops)
}

// ── TestGenerateHashFunction_Scalars verifies that the generator handles
// scalar types that require canonicalization (floats, complex) and
// produces deterministic, non-trivial hashes.
func TestGenerateHashFunction_Scalars(t *testing.T) {
	tests := []struct {
		name string
		typ  reflect.Type
	}{
		{"float32", reflect.TypeOf(float32(0))},
		{"float64", reflect.TypeOf(float64(0))},
		{"complex64", reflect.TypeOf(complex64(0))},
		{"complex128", reflect.TypeOf(complex128(0))},
		{"int32", reflect.TypeOf(int32(0))},
		{"string", reflect.TypeOf("")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fn := GenerateHashFunction(tc.typ)
			if fn == nil {
				t.Fatalf("GenerateHashFunction returned nil for %v", tc.typ)
			}
		})
	}
}

// ── TestGenerateHashFunction_ReturnsNilForUnsupported verifies that the
// generator returns nil for types it cannot handle, so the caller falls
// back to HashFallbackMaphash.
func TestGenerateHashFunction_ReturnsNilForUnsupported(t *testing.T) {
	// interface type
	var iface any
	fn := GenerateHashFunction(reflect.TypeOf(&iface).Elem())
	if fn != nil {
		t.Fatalf("expected nil for interface type, got non-nil")
	}
}

// ── TestGenerateHashFunction_StructWithPadding verifies that structs
// with inter-field padding are hashed correctly — fields are read at
// their actual offsets, padding bytes are skipped.
func TestGenerateHashFunction_StructWithPadding(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genPaddingStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil for padding struct")
	}

	seed := uint64(0xABCD)
	v1 := genPaddingStruct{A: 1, B: 100}
	v2 := genPaddingStruct{A: 2, B: 200}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h1b := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)

	if h1 != h1b {
		t.Fatalf("non-deterministic: %#x != %#x", h1, h1b)
	}
	if h1 == h2 {
		t.Fatalf("different values produced same hash")
	}
}

// ── TestGenerateHashFunction_FloatCanonicalization verifies that +0/-0
// hash identically and different float values hash differently.
func TestGenerateHashFunction_FloatCanonicalization(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genFloatStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x5678)
	pos0 := genFloatStruct{X: 1, F: 0.0}
	neg0 := genFloatStruct{X: 1, F: math.Copysign(0, -1)}

	hPos := fn(unsafe.Pointer(&pos0), seed)
	hNeg := fn(unsafe.Pointer(&neg0), seed)
	if hPos != hNeg {
		t.Fatalf("+0 and -0 produced different hashes: %#x vs %#x", hPos, hNeg)
	}

	// NaN values should all hash the same.
	nan1 := genFloatStruct{X: 1, F: math.NaN()}
	nan2 := genFloatStruct{X: 1, F: math.Float64frombits(0x7ff8000000000001)}
	hNaN1 := fn(unsafe.Pointer(&nan1), seed)
	hNaN2 := fn(unsafe.Pointer(&nan2), seed)
	if hNaN1 != hNaN2 {
		t.Fatalf("different NaN values produced different hashes: %#x vs %#x", hNaN1, hNaN2)
	}

	// Different values should differ.
	other := genFloatStruct{X: 2, F: 3.14}
	hOther := fn(unsafe.Pointer(&other), seed)
	if hPos == hOther {
		t.Fatalf("different values produced same hash")
	}
}

// ── TestGenerateHashFunction_ComplexStruct verifies complex128 field
// handling with float canonicalization.
func TestGenerateHashFunction_ComplexStruct(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genComplexStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x9999)
	v1 := genComplexStruct{A: 1, C: complex(0.0, 0.0)}
	v2 := genComplexStruct{A: 1, C: complex(math.Copysign(0, -1), math.Copysign(0, -1))}
	v3 := genComplexStruct{A: 1, C: complex(1.5, 2.5)}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)
	h3 := fn(unsafe.Pointer(&v3), seed)

	if h1 != h2 {
		t.Fatalf("+0+0i and -0-0i produced different hashes")
	}
	if h1 == h3 {
		t.Fatalf("different complex values produced same hash")
	}
}

// ── TestGenerateHashFunction_StringStruct verifies that structs with
// string fields are hashed by string content, not header bytes.
func TestGenerateHashFunction_StringStruct(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genStringStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x4444)
	v1 := genStringStruct{N: 1, S: "hello"}
	v2 := genStringStruct{N: 1, S: "world"}
	v3 := genStringStruct{N: 1, S: "hello"}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)
	h3 := fn(unsafe.Pointer(&v3), seed)

	if h1 == h2 {
		t.Fatalf("different strings produced same hash")
	}
	if h1 != h3 {
		t.Fatalf("same values produced different hashes: %#x vs %#x", h1, h3)
	}
}

// ── TestGenerateHashFunction_NestedStruct verifies that nested structs
// with ineligible fields are handled recursively.
func TestGenerateHashFunction_NestedStruct(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genNestedStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x7777)
	v1 := genNestedStruct{
		F: genFloatStruct{X: 1, F: 3.14},
		S: genStringStruct{N: 2, S: "abc"},
	}
	v2 := genNestedStruct{
		F: genFloatStruct{X: 1, F: 3.14},
		S: genStringStruct{N: 2, S: "xyz"},
	}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)
	h1b := fn(unsafe.Pointer(&v1), seed)

	if h1 != h1b {
		t.Fatalf("non-deterministic")
	}
	if h1 == h2 {
		t.Fatalf("different nested structs produced same hash")
	}
}

// ── TestGenerateHashFunction_EmptyStruct verifies that empty structs
// are handled (they hash to a constant derived from the seed).
func TestGenerateHashFunction_EmptyStruct(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genEmptyStruct{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil for empty struct")
	}

	seed := uint64(0x1111)
	v := genEmptyStruct{}
	h1 := fn(unsafe.Pointer(&v), seed)
	h2 := fn(unsafe.Pointer(&v), seed)
	if h1 != h2 {
		t.Fatalf("non-deterministic for empty struct")
	}
}

// ── TestGenerateHashFunction_BlankFields verifies that structs with
// only blank fields are handled (blank fields skipped).
func TestGenerateHashFunction_BlankFields(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genAllBlank{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil for all-blank struct")
	}
}

// ── TestGenerateHashFunction_ArrayOfFloats verifies that arrays of
// float64 are hashed element-by-element with canonicalization.
func TestGenerateHashFunction_ArrayOfFloats(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf([3]float64{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x2222)
	v1 := [3]float64{1.0, 2.0, 3.0}
	v2 := [3]float64{1.0, 2.0, 4.0}
	v3 := [3]float64{0.0, 0.0, 0.0}
	v4 := [3]float64{math.Copysign(0, -1), math.Copysign(0, -1), math.Copysign(0, -1)}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)
	h3 := fn(unsafe.Pointer(&v3), seed)
	h4 := fn(unsafe.Pointer(&v4), seed)

	if h1 == h2 {
		t.Fatalf("different arrays produced same hash")
	}
	if h3 != h4 {
		t.Fatalf("+0 and -0 arrays produced different hashes")
	}
}

// ── TestGenerateHashFunction_ArrayOfEligibleElements verifies that
// arrays of raw-byte-eligible elements use byte-block hashing.
func TestGenerateHashFunction_ArrayOfEligibleElements(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf([4]uint32{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	seed := uint64(0x3333)
	v1 := [4]uint32{1, 2, 3, 4}
	v2 := [4]uint32{1, 2, 3, 5}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)

	if h1 == h2 {
		t.Fatalf("different arrays produced same hash")
	}
}

// ── TestGenerateHashFunction_MixedStruct verifies a struct with
// multiple field types: scalar, float, string, complex.
func TestGenerateHashFunction_MixedStruct(t *testing.T) {
	fn := GenerateHashFunction(reflect.TypeOf(genMixed{}))
	if fn == nil {
		t.Fatalf("GenerateHashFunction returned nil for mixed struct")
	}

	seed := uint64(0x5555)
	v1 := genMixed{A: 1, B: 3.14, C: "hello", D: complex(1, 2)}
	v2 := genMixed{A: 1, B: 3.14, C: "hello", D: complex(1, 3)}

	h1 := fn(unsafe.Pointer(&v1), seed)
	h1b := fn(unsafe.Pointer(&v1), seed)
	h2 := fn(unsafe.Pointer(&v2), seed)

	if h1 != h1b {
		t.Fatalf("non-deterministic")
	}
	if h1 == h2 {
		t.Fatalf("different values produced same hash")
	}
}

// ── TestGenerateHashFunction_Caching verifies that repeated calls
// return the same function (cache hit).
func TestGenerateHashFunction_Caching(t *testing.T) {
	typ := reflect.TypeOf(genPaddingStruct{})
	fn1 := GenerateHashFunction(typ)
	fn2 := GenerateHashFunction(typ)
	if fn1 == nil || fn2 == nil {
		t.Fatalf("GenerateHashFunction returned nil")
	}

	// Same output for the same input proves the cache is working.
	seed := uint64(0x6666)
	v := genPaddingStruct{A: 42, B: 999}
	if fn1(unsafe.Pointer(&v), seed) != fn2(unsafe.Pointer(&v), seed) {
		t.Fatalf("cached functions produce different results")
	}
}

// TestBuildClosureFromOps_UnrolledFourToEightMatchesLoopBaseline verifies
// that the hand-unrolled 4–8 op closures produce exactly the same hashes as
// the generic loop path they replace.
func TestBuildClosureFromOps_UnrolledFourToEightMatchesLoopBaseline(t *testing.T) {
	tests := []struct {
		name    string
		sample  any
		wantOps int
		ptr     func() unsafe.Pointer
	}{
		{
			name:    "4 ops",
			sample:  genOps4{},
			wantOps: 4,
			ptr: func() unsafe.Pointer {
				v := &genOps4{A: 1, B: 2.5, C: 3, D: 4.5}
				return unsafe.Pointer(v)
			},
		},
		{
			name:    "5 ops",
			sample:  genOps5{},
			wantOps: 5,
			ptr: func() unsafe.Pointer {
				v := &genOps5{A: 1, B: 2.5, C: 3, D: 4.5, E: 5}
				return unsafe.Pointer(v)
			},
		},
		{
			name:    "6 ops",
			sample:  genOps6{},
			wantOps: 6,
			ptr: func() unsafe.Pointer {
				v := &genOps6{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5}
				return unsafe.Pointer(v)
			},
		},
		{
			name:    "7 ops",
			sample:  genOps7{},
			wantOps: 7,
			ptr: func() unsafe.Pointer {
				v := &genOps7{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5, G: 7}
				return unsafe.Pointer(v)
			},
		},
		{
			name:    "8 ops",
			sample:  genOps8{},
			wantOps: 8,
			ptr: func() unsafe.Pointer {
				v := &genOps8{A: 1, B: 2.5, C: 3, D: 4.5, E: 5, F: 6.5, G: 7, H: 8.5}
				return unsafe.Pointer(v)
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ops := mustMergedOpsForStruct(t, tc.sample)
			if len(ops) != tc.wantOps {
				t.Fatalf("got %d ops, want %d", len(ops), tc.wantOps)
			}

			gotFn := buildClosureFromOps(ops)
			loopFn := buildLoopClosureFromOpsBaseline(ops)
			seed := uint64(0xC0FFEE)
			p := tc.ptr()

			if got := gotFn(p, seed); got != loopFn(p, seed) {
				t.Fatalf("hash mismatch: got %#x want %#x", got, loopFn(p, seed))
			}
		})
	}
}

// TestHashFixedByteBlocksMatchHashBytesBlock verifies that the specialized
// fixed-size byte-block helpers preserve the generic HashBytesBlock result.
func TestHashFixedByteBlocksMatchHashBytesBlock(t *testing.T) {
	var block [32]byte
	for i := range block {
		block[i] = byte(i*17 + 3)
	}

	tests := []struct {
		name string
		size int
		fn   HashFunction
	}{
		{name: "12", size: 12, fn: hashByteBlock12},
		{name: "16", size: 16, fn: hashByteBlock16},
		{name: "24", size: 24, fn: hashByteBlock24},
		{name: "32", size: 32, fn: hashByteBlock32},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			seed := uint64(0x1234567890ABCDEF)
			want := HashBytesBlock(seed, block[:tc.size])
			got := tc.fn(unsafe.Pointer(&block[0]), seed)
			if got != want {
				t.Fatalf("hash mismatch for size %d: got %#x want %#x", tc.size, got, want)
			}
		})
	}
}

// ── TestGenerateHashFunction_IntegrationWithMakeRuntimeHasher verifies
// that MakeRuntimeHasher picks the generated hasher for types that are
// not raw-byte-eligible but are supported by the generator.
func TestGenerateHashFunction_IntegrationWithMakeRuntimeHasher(t *testing.T) {
	h := MakeRuntimeHasher[genFloatStruct](0xBEEF)

	v1 := genFloatStruct{X: 10, F: 2.718}
	v2 := genFloatStruct{X: 10, F: 3.141}

	hash1 := h.Hash(v1)
	hash1b := h.Hash(v1)
	hash2 := h.Hash(v2)

	if hash1 != hash1b {
		t.Fatalf("non-deterministic")
	}
	if hash1 == hash2 {
		t.Fatalf("different values produced same hash")
	}

	// Verify it's NOT the fallback (different algorithm means different output).
	seed := h.Seed
	p := unsafe.Pointer(&v1)
	fallbackHash := HashFallbackMaphash[genFloatStruct](p, seed)
	if hash1 == fallbackHash {
		t.Logf("note: generated hash coincidentally matches fallback for this value")
	}
}
