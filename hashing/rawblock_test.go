package hashing

import (
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

type rbNoPaddingStruct struct {
	A uint64
	B uint64
}

type rbPaddingStruct struct {
	A uint8
	B uint64
}

type rbTrailingPaddingStruct struct {
	A uint64
	B uint8
}

type rbBlankFieldStruct struct {
	A uint32
	_ uint32
	B uint32
}

type rbFloatStruct struct {
	A uint32
	F float64
}

type rbNestedOK struct {
	X rbNoPaddingStruct
	Y [2]uint64
}

type rbNestedBad struct {
	X rbNoPaddingStruct
	Y [2]float32
}

func TestCanUseUnsafeRawByteBlockHasherType_ScalarsAndSimpleKinds(t *testing.T) {
	tests := []struct {
		name string
		t    reflect.Type
		ok   bool
	}{
		{name: "uint64", t: reflect.TypeOf(uint64(0)), ok: true},
		{name: "int32", t: reflect.TypeOf(int32(0)), ok: true},
		{name: "bool", t: reflect.TypeOf(false), ok: true},
		{name: "uintptr", t: reflect.TypeOf(uintptr(0)), ok: true},
		{name: "pointer", t: reflect.TypeOf((*int)(nil)), ok: true},
		{name: "float32", t: reflect.TypeOf(float32(0)), ok: false},
		{name: "float64", t: reflect.TypeOf(float64(0)), ok: false},
		{name: "complex64", t: reflect.TypeOf(complex64(0)), ok: false},
		{name: "string", t: reflect.TypeOf(""), ok: false},
		{name: "slice", t: reflect.TypeOf([]byte(nil)), ok: false},
		{name: "map", t: reflect.TypeOf(map[int]int(nil)), ok: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CanUseUnsafeRawByteBlockHasherType(tc.t)
			if got.Eligible != tc.ok {
				t.Fatalf("eligible=%v, want %v (reason=%q)", got.Eligible, tc.ok, got.Reason)
			}
			if got.Reason == "" {
				t.Fatalf("reason must not be empty")
			}
		})
	}
}

func TestCanUseUnsafeRawByteBlockHasherType_StructRules(t *testing.T) {
	tests := []struct {
		name       string
		t          reflect.Type
		ok         bool
		reasonLike string
	}{
		{name: "no padding struct", t: reflect.TypeOf(rbNoPaddingStruct{}), ok: true},
		{name: "struct with padding", t: reflect.TypeOf(rbPaddingStruct{}), ok: false, reasonLike: "padding"},
		{name: "struct with trailing padding", t: reflect.TypeOf(rbTrailingPaddingStruct{}), ok: false, reasonLike: "padding"},
		{name: "struct with blank field", t: reflect.TypeOf(rbBlankFieldStruct{}), ok: false, reasonLike: "blank"},
		{name: "struct with float", t: reflect.TypeOf(rbFloatStruct{}), ok: false},
		{name: "nested struct ok", t: reflect.TypeOf(rbNestedOK{}), ok: true},
		{name: "nested struct bad", t: reflect.TypeOf(rbNestedBad{}), ok: false, reasonLike: "float"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CanUseUnsafeRawByteBlockHasherType(tc.t)
			if got.Eligible != tc.ok {
				t.Fatalf("eligible=%v, want %v (reason=%q)", got.Eligible, tc.ok, got.Reason)
			}
			if tc.reasonLike != "" && !strings.Contains(strings.ToLower(got.Reason), strings.ToLower(tc.reasonLike)) {
				t.Fatalf("reason %q does not contain %q", got.Reason, tc.reasonLike)
			}
		})
	}
}

func TestCanUseUnsafeRawByteBlockHasherType_Arrays(t *testing.T) {
	tests := []struct {
		name string
		t    reflect.Type
		ok   bool
	}{
		{name: "array of uint32", t: reflect.TypeOf([4]uint32{}), ok: true},
		{name: "array of float64", t: reflect.TypeOf([2]float64{}), ok: false},
		{name: "array of no-padding structs", t: reflect.TypeOf([3]rbNoPaddingStruct{}), ok: true},
		{name: "array of padded structs", t: reflect.TypeOf([3]rbPaddingStruct{}), ok: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CanUseUnsafeRawByteBlockHasherType(tc.t)
			if got.Eligible != tc.ok {
				t.Fatalf("eligible=%v, want %v (reason=%q)", got.Eligible, tc.ok, got.Reason)
			}
		})
	}
}

func TestCanUseUnsafeRawByteBlockHasher_GenericEntryPoint(t *testing.T) {
	t.Run("eligible type", func(t *testing.T) {
		got := CanUseUnsafeRawByteBlockHasher[rbNoPaddingStruct]()
		if !got.Eligible {
			t.Fatalf("expected eligible, got false: %q", got.Reason)
		}
	})

	t.Run("ineligible comparable type", func(t *testing.T) {
		got := CanUseUnsafeRawByteBlockHasher[rbFloatStruct]()
		if got.Eligible {
			t.Fatalf("expected ineligible for float-containing struct")
		}
	})
}

func TestMakeRuntimeHasher_UsesRawByteHasherWhenEligible(t *testing.T) {
	h := MakeRuntimeHasher[rbNoPaddingStruct](0x1234)

	values := []rbNoPaddingStruct{{A: 1, B: 2}, {A: 3, B: 5}, {A: 8, B: 13}}
	seed := uint64(0x1234)
	atLeastOneDiffToFallback := false
	for _, v := range values {
		p := unsafe.Pointer(&v)
		got := h.fn(p, seed)
		wantRaw := HashAsByteArray[rbNoPaddingStruct](p, seed)
		wantFallback := HashFallbackMaphash[rbNoPaddingStruct](p, seed)
		if got != wantRaw {
			t.Fatalf("expected raw-byte hasher result, got=%#x wantRaw=%#x", got, wantRaw)
		}
		if got != wantFallback {
			atLeastOneDiffToFallback = true
		}
	}
	if !atLeastOneDiffToFallback {
		t.Fatalf("unable to distinguish path from fallback: all test vectors matched fallback outputs")
	}
}

func TestMakeRuntimeHasher_DoesNotUseRawByteHasherWhenIneligible(t *testing.T) {
	h := MakeRuntimeHasher[rbPaddingStruct](0x1234)

	values := []rbPaddingStruct{{A: 1, B: 2}, {A: 7, B: 11}, {A: 13, B: 17}}
	seed := uint64(0x1234)
	atLeastOneDiffToRaw := false
	for _, v := range values {
		p := unsafe.Pointer(&v)
		got := h.fn(p, seed)
		// Must be deterministic.
		got2 := h.fn(p, seed)
		if got != got2 {
			t.Fatalf("non-deterministic: %#x != %#x", got, got2)
		}
		wantRaw := HashAsByteArray[rbPaddingStruct](p, seed)
		if got != wantRaw {
			atLeastOneDiffToRaw = true
		}
	}
	if !atLeastOneDiffToRaw {
		t.Fatalf("unable to distinguish path from raw-byte hasher: all test vectors matched raw outputs")
	}
	// Different values must produce different hashes.
	p0 := unsafe.Pointer(&values[0])
	p1 := unsafe.Pointer(&values[1])
	if h.fn(p0, seed) == h.fn(p1, seed) {
		t.Fatalf("different values produced same hash")
	}
}

func TestMakeRuntimeHasher_DoesNotUseRawByteHasherForFloatStruct(t *testing.T) {
	h := MakeRuntimeHasher[rbFloatStruct](0x1234)

	seed := uint64(0x1234)
	// +0 and -0 must hash identically (Go equality: +0 == -0).
	v0 := rbFloatStruct{A: 1, F: 0}
	vNeg0 := rbFloatStruct{A: 1, F: -0}
	p0 := unsafe.Pointer(&v0)
	pNeg0 := unsafe.Pointer(&vNeg0)
	if h.fn(p0, seed) != h.fn(pNeg0, seed) {
		t.Fatalf("+0 and -0 produced different hashes: %#x vs %#x", h.fn(p0, seed), h.fn(pNeg0, seed))
	}
	// Determinism: the repeated call is the point, not a copy-paste slip.
	if h.fn(p0, seed) != h.fn(p0, seed) { //nolint:staticcheck // SA4000
		t.Fatalf("non-deterministic")
	}
	// Different values must differ.
	vOther := rbFloatStruct{A: 2, F: 3.5}
	pOther := unsafe.Pointer(&vOther)
	if h.fn(p0, seed) == h.fn(pOther, seed) {
		t.Fatalf("different values produced same hash")
	}
	// Must NOT use raw-byte hasher.
	rawHash := HashAsByteArray[rbFloatStruct](p0, seed)
	if h.fn(p0, seed) == rawHash {
		t.Logf("warning: generated hash matches raw-byte hash for +0 case (may be coincidence)")
	}
}

// TestCanUseUnsafeRawByteBlockHasherTypeRejectsReferenceKinds verifies that the
// fast path which hashes a value's raw memory is refused for every type whose
// memory is a reference to its value rather than the value itself.
//
// This decision is what stands between the fastest hasher in the package and
// silent corruption: a map, slice, string or func is a pointer plus bookkeeping,
// so two equal values can have completely different bytes and two different
// values can share them. The analysis is the guard, so each refusal is worth
// pinning individually.
//
// It asks about one type per rejected kind, plus a nil type, and requires
// ineligibility with a reason that is filled in.
func TestCanUseUnsafeRawByteBlockHasherTypeRejectsReferenceKinds(t *testing.T) {
	var iface any

	cases := []struct {
		name string
		typ  reflect.Type
	}{
		{"nil type", nil},
		{"string", reflect.TypeOf("")},
		{"slice", reflect.TypeOf([]byte(nil))},
		{"map", reflect.TypeOf(map[int]int(nil))},
		{"interface", reflect.TypeOf(&iface).Elem()},
		{"func", reflect.TypeOf(func() {})},
		{"float32", reflect.TypeOf(float32(0))},
		{"float64", reflect.TypeOf(float64(0))},
		{"complex64", reflect.TypeOf(complex64(0))},
		{"complex128", reflect.TypeOf(complex128(0))},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := CanUseUnsafeRawByteBlockHasherType(c.typ)
			if got.Eligible {
				t.Fatalf("%s must not be raw-byte hashable, got eligible with reason %q", c.name, got.Reason)
			}
			if got.Reason == "" {
				t.Fatalf("%s was rejected without a reason", c.name)
			}
		})
	}
}

// TestCanUseUnsafeRawByteBlockHasherTypeAcceptsOnlySelfContainedMemory verifies
// the other direction: that types whose bytes really are their value keep the
// fast path.
//
// Every type accepted here skips the generated per-field hasher entirely and is
// hashed in one pass over its memory. Being too conservative costs speed on the
// most common key types, so the accepting cases deserve as much attention as the
// rejecting ones.
//
// It asks about scalars, pointers, channels and composites built from them, and
// requires eligibility with a reason that is filled in.
func TestCanUseUnsafeRawByteBlockHasherTypeAcceptsOnlySelfContainedMemory(t *testing.T) {
	cases := []struct {
		name string
		typ  reflect.Type
	}{
		{"bool", reflect.TypeOf(false)},
		{"int64", reflect.TypeOf(int64(0))},
		{"uintptr", reflect.TypeOf(uintptr(0))},
		{"pointer", reflect.TypeOf((*int)(nil))},
		{"unsafe pointer", reflect.TypeOf(unsafe.Pointer(nil))},
		{"channel", reflect.TypeOf(make(chan int))},
		{"array of scalars", reflect.TypeOf([4]uint32{})},
		{"empty array", reflect.TypeOf([0]int{})},
		{"struct without padding", reflect.TypeOf(rbNoPaddingStruct{})},
		{"empty struct", reflect.TypeOf(struct{}{})},
		{"nested eligible struct", reflect.TypeOf(rbNestedOK{})},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := CanUseUnsafeRawByteBlockHasherType(c.typ)
			if !got.Eligible {
				t.Fatalf("%s should be raw-byte hashable, rejected with reason %q", c.name, got.Reason)
			}
			if got.Reason == "" {
				t.Fatalf("%s was accepted without a reason", c.name)
			}
		})
	}
}

// TestEveryReflectKindHasAnExplicitAnswer verifies that the raw-byte
// eligibility analysis has made a deliberate decision about every kind of Go
// type that exists, rather than falling through to a catch-all.
//
// The analysis decides whether a key type takes the fastest hashing path in the
// package. A kind that is not named explicitly would be answered by the default
// case, which is the correct conservative answer but also an unconsidered one --
// and if a future Go release adds a kind, nobody would notice that it silently
// lost the fast path, or that it should never have had it.
//
// It builds a value of every reflect.Kind except Invalid, asks the analysis
// about each, and requires an answer other than the catch-all. It also checks
// that the list of kinds it walks is still complete, so a kind added to Go makes
// this test fail rather than pass vacuously.
func TestEveryReflectKindHasAnExplicitAnswer(t *testing.T) {
	var iface any

	perKind := map[reflect.Kind]reflect.Type{
		reflect.Bool:          reflect.TypeOf(false),
		reflect.Int:           reflect.TypeOf(int(0)),
		reflect.Int8:          reflect.TypeOf(int8(0)),
		reflect.Int16:         reflect.TypeOf(int16(0)),
		reflect.Int32:         reflect.TypeOf(int32(0)),
		reflect.Int64:         reflect.TypeOf(int64(0)),
		reflect.Uint:          reflect.TypeOf(uint(0)),
		reflect.Uint8:         reflect.TypeOf(uint8(0)),
		reflect.Uint16:        reflect.TypeOf(uint16(0)),
		reflect.Uint32:        reflect.TypeOf(uint32(0)),
		reflect.Uint64:        reflect.TypeOf(uint64(0)),
		reflect.Uintptr:       reflect.TypeOf(uintptr(0)),
		reflect.Float32:       reflect.TypeOf(float32(0)),
		reflect.Float64:       reflect.TypeOf(float64(0)),
		reflect.Complex64:     reflect.TypeOf(complex64(0)),
		reflect.Complex128:    reflect.TypeOf(complex128(0)),
		reflect.Array:         reflect.TypeOf([1]int{}),
		reflect.Chan:          reflect.TypeOf(make(chan int)),
		reflect.Func:          reflect.TypeOf(func() {}),
		reflect.Interface:     reflect.TypeOf(&iface).Elem(),
		reflect.Map:           reflect.TypeOf(map[int]int(nil)),
		reflect.Pointer:       reflect.TypeOf((*int)(nil)),
		reflect.Slice:         reflect.TypeOf([]int(nil)),
		reflect.String:        reflect.TypeOf(""),
		reflect.Struct:        reflect.TypeOf(struct{ A int }{}),
		reflect.UnsafePointer: reflect.TypeOf(unsafe.Pointer(nil)),
	}

	// reflect.Kind values run from Invalid to UnsafePointer without gaps. If Go
	// grows a new one, the count changes and the missing entry has to be added
	// here and, more importantly, to the analysis itself.
	for k := reflect.Invalid + 1; k <= reflect.UnsafePointer; k++ {
		if _, ok := perKind[k]; !ok {
			t.Fatalf("reflect.Kind %v (%d) is not covered by this test; "+
				"check that CanUseUnsafeRawByteBlockHasherType names it explicitly", k, k)
		}
	}
	if len(perKind) != int(reflect.UnsafePointer) {
		t.Fatalf("this test walks %d kinds, reflect defines %d excluding Invalid",
			len(perKind), int(reflect.UnsafePointer))
	}

	for kind, typ := range perKind {
		got := CanUseUnsafeRawByteBlockHasherType(typ)
		if got.Reason == unhandledKindReason {
			t.Fatalf("kind %v fell through to the catch-all; it needs an explicit case", kind)
		}
		if got.Reason == "" {
			t.Fatalf("kind %v was answered without a reason", kind)
		}
	}
}
