// nolint:gosec // Test file: intentional unsafe.Pointer use for testing byte hashing
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
		wantFallback := HashFallbackMaphash[rbPaddingStruct](p, seed)
		wantRaw := HashAsByteArray[rbPaddingStruct](p, seed)
		if got != wantFallback {
			t.Fatalf("expected fallback hasher result, got=%#x wantFallback=%#x", got, wantFallback)
		}
		if got != wantRaw {
			atLeastOneDiffToRaw = true
		}
	}
	if !atLeastOneDiffToRaw {
		t.Fatalf("unable to distinguish path from raw-byte hasher: all test vectors matched raw outputs")
	}
}

func TestMakeRuntimeHasher_DoesNotUseRawByteHasherForFloatStruct(t *testing.T) {
	h := MakeRuntimeHasher[rbFloatStruct](0x1234)

	values := []rbFloatStruct{
		{A: 1, F: 0},
		{A: 1, F: -0},
		{A: 2, F: 3.5},
	}
	seed := uint64(0x1234)
	for _, v := range values {
		p := unsafe.Pointer(&v)
		got := h.fn(p, seed)
		wantFallback := HashFallbackMaphash[rbFloatStruct](p, seed)
		if got != wantFallback {
			t.Fatalf("expected fallback hasher result for float struct, got=%#x wantFallback=%#x", got, wantFallback)
		}
	}
}
