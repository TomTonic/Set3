package hashing

import (
	"math"
	"testing"
	"unsafe"
)

func TestCanonicalizeFloat64Bits_ZeroAndNaN(t *testing.T) {
	const seed = uint64(0x1234_5678_9abc_def0)

	// +0 and -0 must hash the same (because +0 == -0).
	p0 := 0.0
	n0 := math.Copysign(0, -1)
	hp0 := HashF64WHdet(unsafe.Pointer(&p0), seed)
	hn0 := HashF64WHdet(unsafe.Pointer(&n0), seed)
	if hp0 != hn0 {
		t.Fatalf("expected +0 and -0 to hash equally: %x vs %x", hp0, hn0)
	}

	// Different NaN payloads (and sign) should canonicalize to the same bits.
	nanA := math.Float64frombits(0x7ff0000000000001) // NaN (payload 1)
	nanB := math.Float64frombits(0x7ff8000000000002) // NaN (payload 2)
	nanC := math.Float64frombits(0xfff8000000000003) // NaN (payload 3, sign set)

	ha := HashF64WHdet(unsafe.Pointer(&nanA), seed)
	hb := HashF64WHdet(unsafe.Pointer(&nanB), seed)
	hc := HashF64WHdet(unsafe.Pointer(&nanC), seed)
	if ha != hb || ha != hc {
		t.Fatalf("expected NaNs to hash equally: %x %x %x", ha, hb, hc)
	}
}

func TestCanonicalizeFloat64BitsBranchless_SignPreserved(t *testing.T) {
	const seed = uint64(0x1234_5678_9abc_def0)

	// Sign must be preserved for non-zero numbers; +x and -x are not equal.
	p := 1.0
	n := -1.0
	hp := HashF64WHdet(unsafe.Pointer(&p), seed)
	hn := HashF64WHdet(unsafe.Pointer(&n), seed)
	if hp == hn {
		t.Fatalf("expected +1 and -1 to hash differently, got %x", hp)
	}
}

func TestHashCanonicalizedFloat64_ZeroCanonicalization(t *testing.T) {
	seed := uint64(0xDEADBEEFCAFEBABE)

	pz := 0.0
	nz := math.Copysign(0.0, -1.0)

	hpz := HashF64WHdet(unsafe.Pointer(&pz), seed)
	hnz := HashF64WHdet(unsafe.Pointer(&nz), seed)

	if hpz != hnz {
		t.Fatalf("+0 and -0 must hash equal: hpz=%#x hnz=%#x", hpz, hnz)
	}
	// For zero, bits=0, so hash should be WH64Det(0, seed).
	if want := WH64Det(0, seed); hpz != want {
		t.Fatalf("zero hash mismatch: got=%#x want=%#x", hpz, want)
	}

	// determinism
	if hpz2 := HashF64WHdet(unsafe.Pointer(&pz), seed); hpz2 != hpz {
		t.Fatalf("non-deterministic for +0: got=%#x want=%#x", hpz2, hpz)
	}
	if hnz2 := HashF64WHdet(unsafe.Pointer(&nz), seed); hnz2 != hnz {
		t.Fatalf("non-deterministic for -0: got=%#x want=%#x", hnz2, hnz)
	}
}

func TestHashCanonicalizedFloat64_NaNsCanonicalized(t *testing.T) {
	seed := uint64(0x0123456789ABCDEF)
	const canonNaNBits = 0x7ff8000000000000

	nans := []uint64{
		canonNaNBits,       // canonical qNaN
		0x7ff8000000000001, // qNaN with payload
		0x7ff8000000001234, // qNaN with different payload
		0x7ff0000000000001, // sNaN-like payload
		0xfff8000000000000, // negative qNaN
		0xfff0000000000001, // negative sNaN-like payload
		0x7fffffffffffffff, // exponent all ones, mantissa all ones (NaN)
		0xffffffffffffffff, // negative NaN with all mantissa bits
		0x7ff0123400000000, // NaN (exp all ones, mantissa non-zero)
		0xfff0123400000000, // negative NaN variant
	}

	want := WH64Det(canonNaNBits, seed)

	var first uint64
	for i, bits := range nans {
		v := math.Float64frombits(bits)
		got := HashF64WHdet(unsafe.Pointer(&v), seed)

		if i == 0 {
			first = got
		}
		if got != first {
			t.Fatalf("NaNs must hash equal after canonicalization: i=%d bits=%#x got=%#x first=%#x", i, bits, got, first)
		}
		if got != want {
			t.Fatalf("NaN canonical hash mismatch: i=%d bits=%#x got=%#x want=%#x", i, bits, got, want)
		}

		// determinism
		got2 := HashF64WHdet(unsafe.Pointer(&v), seed)
		if got2 != got {
			t.Fatalf("non-deterministic NaN hash: i=%d bits=%#x got=%#x got2=%#x", i, bits, got, got2)
		}
	}
}

func TestHashCanonicalizedFloat64_NonNaNMatchesWH64DetOfBits(t *testing.T) {
	seed := uint64(0xC0FFEE1234567890)

	vals := []float64{
		1.0,
		-1.0,
		1.5,
		-2.718281828,
		3.1415926535,
		math.SmallestNonzeroFloat64,
		math.MaxFloat64,
		math.Inf(1),
		math.Inf(-1),
	}

	for _, v := range vals {
		got := HashF64WHdet(unsafe.Pointer(&v), seed)
		u := math.Float64bits(v)
		want := WH64Det(u, seed)

		if got != want {
			t.Fatalf("non-NaN hash mismatch for v=%v: got=%#x want=%#x", v, got, want)
		}

		// determinism
		if got2 := HashF64WHdet(unsafe.Pointer(&v), seed); got2 != got {
			t.Fatalf("non-deterministic hash for v=%v: got=%#x got2=%#x", v, got, got2)
		}
	}
}

func TestHashCanonicalizedFloat64_InfIsNotTreatedAsNaN(t *testing.T) {
	seed := uint64(0xBADC0FFEE0DDF00D)

	posInf := math.Inf(1)
	negInf := math.Inf(-1)
	nan := math.NaN()

	hPos := HashF64WHdet(unsafe.Pointer(&posInf), seed)
	hNeg := HashF64WHdet(unsafe.Pointer(&negInf), seed)
	hNaN := HashF64WHdet(unsafe.Pointer(&nan), seed)

	wantPos := WH64Det(math.Float64bits(posInf), seed)
	wantNeg := WH64Det(math.Float64bits(negInf), seed)

	if hPos != wantPos {
		t.Fatalf("+Inf hash mismatch: got=%#x want=%#x", hPos, wantPos)
	}
	if hPos == hNaN {
		t.Fatalf("+Inf and NaN must not hash equal under bit-hash: +Inf=%#x NaN=%#x", hPos, hNaN)
	}
	if hNeg != wantNeg {
		t.Fatalf("-Inf hash mismatch: got=%#x want=%#x", hNeg, wantNeg)
	}
	if hNeg == hNaN {
		t.Fatalf("-Inf and NaN must not hash equal under bit-hash: -Inf=%#x NaN=%#x", hNeg, hNaN)
	}
	if hPos == hNeg {
		t.Fatalf("+Inf and -Inf must not hash equal under bit-hash: +Inf=%#x -Inf=%#x", hPos, hNeg)
	}
}
