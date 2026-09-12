package hashing

import (
	"math"
	"os/exec"
	"reflect"
	"regexp"
	"runtime"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// ── Types that exercise each branch of planWords ───────────────────────────

type wpFloat3 struct{ X, Y, Z float64 }
type wpFloat6 struct{ A, B, C, D, E, F float64 }
type wpFloat32x3 struct{ R, G, B float32 }
type wpIntFloat struct {
	ID    int64
	Score float64
}
type wpFloatString struct {
	Lat, Lon float64
	Name     string
}
type wpComplex2 struct{ A, B complex128 }
type wpComplex64x2 struct{ A, B complex64 }
type wpBlock16Float struct {
	A, B int64
	F    float64
}
type wpMixedWidth struct {
	A float64
	B float32
}
type wpNarrowBlockFloat struct {
	N int32
	F float64
}
type wpStrings3 struct{ A, B, C string }
type wpTwelveFloats struct {
	A, B, C, D, E, F, G, H, I, J, K, L float64
}
type wpEighteenFloats struct {
	A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R float64
}

// ── The oracle ─────────────────────────────────────────────────────────────

// refCanonF64 is the independent reference for float64 canonicalization: it
// compares the float, the way the rest of this package has always done, rather
// than testing the loaded bits.
func refCanonF64(f float64) uint64 {
	switch {
	case f == 0:
		return 0
	case math.IsNaN(f):
		return 0x7ff8000000000000
	default:
		return math.Float64bits(f)
	}
}

func refCanonF32(f float32) uint64 {
	switch {
	case f == 0:
		return 0
	case math.IsNaN(float64(f)):
		return 0x7fc00000
	default:
		return uint64(math.Float32bits(f))
	}
}

// TestWordPathMatchesAnIndependentWordSequence is the load-bearing test of
// wordplan.go.
//
// For each type it spells out, by hand, which words that type should produce
// and in which order, hashes them with the generic byte routine through
// mixWordSlice, and requires the generated closure to return the same value.
// The expected word sequences below do not go through planWords, so a bug in
// the classification, an offset off by eight, a dropped field or a wrong canon
// bit all show up here.
func TestWordPathMatchesAnIndependentWordSequence(t *testing.T) {
	seeds := []uint64{0, 1, 0x9e3779b97f4a7c15, ^uint64(0)}

	t.Run("three float64", func(t *testing.T) {
		v := wpFloat3{1.5, -2.25, 3e300}
		want := func(seed uint64) uint64 {
			return mixWordSlice([]uint64{
				refCanonF64(v.X), refCanonF64(v.Y), refCanonF64(v.Z),
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("six float64", func(t *testing.T) {
		v := wpFloat6{1, 2, 3, 4, 5, 6}
		want := func(seed uint64) uint64 {
			return mixWordSlice([]uint64{
				refCanonF64(v.A), refCanonF64(v.B), refCanonF64(v.C),
				refCanonF64(v.D), refCanonF64(v.E), refCanonF64(v.F),
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("three float32", func(t *testing.T) {
		v := wpFloat32x3{0.25, -0.5, 1e30}
		want := func(seed uint64) uint64 {
			return mixWordSlice([]uint64{
				refCanonF32(v.R), refCanonF32(v.G), refCanonF32(v.B),
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("int64 then float64", func(t *testing.T) {
		v := wpIntFloat{ID: -7, Score: 99.5}
		want := func(seed uint64) uint64 {
			// The int64 is a byte block of exactly eight bytes, so its word is
			// the raw load with no canonicalization.
			return mixWordSlice([]uint64{
				uint64(v.ID), refCanonF64(v.Score), //nolint:gosec
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("two int64 merged then float64", func(t *testing.T) {
		v := wpBlock16Float{A: 11, B: 22, F: 33.5}
		want := func(seed uint64) uint64 {
			// A and B merge into one sixteen-byte block, which becomes two
			// plain words.
			return mixWordSlice([]uint64{
				uint64(v.A), uint64(v.B), refCanonF64(v.F), //nolint:gosec
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("two complex128", func(t *testing.T) {
		v := wpComplex2{A: complex(1.5, -2.5), B: complex(0, 4)}
		want := func(seed uint64) uint64 {
			return mixWordSlice([]uint64{
				refCanonF64(real(v.A)), refCanonF64(imag(v.A)),
				refCanonF64(real(v.B)), refCanonF64(imag(v.B)),
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("two complex64", func(t *testing.T) {
		v := wpComplex64x2{A: complex(1.5, -2.5), B: complex(0, 4)}
		want := func(seed uint64) uint64 {
			return mixWordSlice([]uint64{
				refCanonF32(real(v.A)), refCanonF32(imag(v.A)),
				refCanonF32(real(v.B)), refCanonF32(imag(v.B)),
			}, seed)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("floats then a string", func(t *testing.T) {
		v := wpFloatString{Lat: 48.2, Lon: 16.37, Name: "vienna"}
		want := func(seed uint64) uint64 {
			// The words are hashed first; the string threads the seed on from
			// the value they produce.
			h := mixWordSlice([]uint64{refCanonF64(v.Lat), refCanonF64(v.Lon)}, seed)
			return HashString(unsafe.Pointer(&v.Name), h)
		}
		requireGenerated(t, v, want, seeds)
	})

	t.Run("twelve float64 through the buffer", func(t *testing.T) {
		v := wpTwelveFloats{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		want := func(seed uint64) uint64 {
			w := make([]uint64, 0, 12)
			rv := reflect.ValueOf(v)
			for i := range rv.NumField() {
				w = append(w, refCanonF64(rv.Field(i).Float()))
			}
			return mixWordSlice(w, seed)
		}
		requireGenerated(t, v, want, seeds)
	})
}

// requireGenerated holds the generated closure for the type of v to want.
func requireGenerated[T comparable](t *testing.T, v T, want func(uint64) uint64, seeds []uint64) {
	t.Helper()
	fn := GenerateHashFunction(reflect.TypeOf(v))
	require.NotNil(t, fn, "no hash function generated for %T", v)
	p := unsafe.Pointer(&v)
	for _, seed := range seeds {
		require.Equalf(t, want(seed), fn(p, seed),
			"%T disagrees with its expected word sequence at seed %#x", v, seed)
	}
}

// ── Which types take which path ────────────────────────────────────────────

// TestWhichTypesTakeTheWordPath pins the classification, including every case
// the fast path deliberately or knowingly declines. A change that quietly
// widens or narrows the path fails here, and the table is the documentation of
// what wordplan.go's header claims.
func TestWhichTypesTakeTheWordPath(t *testing.T) {
	cases := []struct {
		name    string
		typ     reflect.Type
		words   int  // 0 means the type keeps the chain
		narrow  bool // four-byte loads
		threads int  // ops still threading the seed
	}{
		{"three float64", reflect.TypeOf(wpFloat3{}), 3, false, 0},
		{"six float64", reflect.TypeOf(wpFloat6{}), 6, false, 0},
		{"three float32", reflect.TypeOf(wpFloat32x3{}), 3, true, 0},
		{"int64 and float64", reflect.TypeOf(wpIntFloat{}), 2, false, 0},
		{"merged block and float64", reflect.TypeOf(wpBlock16Float{}), 3, false, 0},
		{"two complex128", reflect.TypeOf(wpComplex2{}), 4, false, 0},
		{"two complex64", reflect.TypeOf(wpComplex64x2{}), 4, true, 0},
		{"floats and a string", reflect.TypeOf(wpFloatString{}), 2, false, 1},
		{"twelve float64", reflect.TypeOf(wpTwelveFloats{}), 12, false, 0},
		{"array of float64", reflect.TypeOf([4]float64{}), 4, false, 0},

		// Declined, each for a reason the header states.
		{"only strings", reflect.TypeOf(wpStrings3{}), 0, false, 0},
		{"mixed load widths", reflect.TypeOf(wpMixedWidth{}), 0, false, 0},
		{"four-byte block beside a float64", reflect.TypeOf(wpNarrowBlockFloat{}), 0, false, 0},
		{"eighteen float64 exceeds the bound", reflect.TypeOf(wpEighteenFloats{}), 0, false, 0},
		{"a single float64", reflect.TypeOf(struct {
			F float64
			S string
		}{}), 0, false, 0},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ops := mergeByteBlocks(flattenTypeOps(c.typ, 0))
			pl, threaded, ok := planWords(ops)
			if c.words == 0 {
				require.False(t, ok, "expected %s to keep the chain, got a %d-word plan", c.name, len(pl.offs))
				require.Nil(t, buildWordClosure(ops), "buildWordClosure should decline %s", c.name)
				return
			}
			require.True(t, ok, "expected %s to take the word path", c.name)
			require.Equal(t, c.words, len(pl.offs), "word count")
			require.Equal(t, c.narrow, pl.narrow, "load width")
			require.Equal(t, c.threads, len(threaded), "threaded op count")
			require.NotNil(t, buildWordClosure(ops))
		})
	}
}

// TestPlanOffsetsAreInsideTheType checks that every word a plan reads lies
// wholly within the value. A word plan does raw loads at captured offsets, so
// an offset past the end would read another object's memory.
func TestPlanOffsetsAreInsideTheType(t *testing.T) {
	types := []reflect.Type{
		reflect.TypeOf(wpFloat3{}), reflect.TypeOf(wpFloat6{}),
		reflect.TypeOf(wpFloat32x3{}), reflect.TypeOf(wpIntFloat{}),
		reflect.TypeOf(wpBlock16Float{}), reflect.TypeOf(wpComplex2{}),
		reflect.TypeOf(wpComplex64x2{}), reflect.TypeOf(wpFloatString{}),
		reflect.TypeOf(wpTwelveFloats{}), reflect.TypeOf([4]float64{}),
		reflect.TypeOf([8]float32{}),
	}
	for _, ty := range types {
		pl, _, ok := planWords(mergeByteBlocks(flattenTypeOps(ty, 0)))
		if !ok {
			continue
		}
		width := uintptr(8)
		if pl.narrow {
			width = 4
		}
		for i, off := range pl.offs {
			require.LessOrEqualf(t, off+width, ty.Size(),
				"%s word %d reads [%d,%d) but the type is only %d bytes",
				ty, i, off, off+width, ty.Size())
		}
	}
}

// ── Canonicalization through the new path ──────────────────────────────────

// TestWordPathCanonicalizesZeroAndNaN checks the two properties a hash for a
// float key must have, now that canonicalization happens on the loaded bits
// rather than on the float: values that compare equal must hash equal, and all
// NaNs must agree. Infinities must not be swept in with the NaNs.
func TestWordPathCanonicalizesZeroAndNaN(t *testing.T) {
	const seed = 0x243f6a8885a308d3
	fn := GenerateHashFunction(reflect.TypeOf(wpFloat3{}))
	require.NotNil(t, fn)
	h := func(v wpFloat3) uint64 { return fn(unsafe.Pointer(&v), seed) }

	t.Run("negative zero hashes as positive zero", func(t *testing.T) {
		require.Equal(t, h(wpFloat3{0, 0, 0}), h(wpFloat3{math.Copysign(0, -1), 0, 0}))
		require.Equal(t, h(wpFloat3{1, 0, 2}), h(wpFloat3{1, math.Copysign(0, -1), 2}))
		require.Equal(t, h(wpFloat3{1, 2, 0}), h(wpFloat3{1, 2, math.Copysign(0, -1)}))
	})

	t.Run("every NaN hashes alike", func(t *testing.T) {
		nans := []float64{
			math.NaN(),
			math.Float64frombits(0x7ff8000000000000),
			math.Float64frombits(0x7fffffffffffffff),
			math.Float64frombits(0xfff8000000000001),
			math.Float64frombits(0x7ff0000000000001), // signalling
		}
		want := h(wpFloat3{nans[0], 1, 2})
		for _, n := range nans[1:] {
			require.Equalf(t, want, h(wpFloat3{n, 1, 2}),
				"NaN %#x hashed apart from the canonical one", math.Float64bits(n))
		}
	})

	t.Run("infinities are not NaN", func(t *testing.T) {
		inf := h(wpFloat3{math.Inf(1), 1, 2})
		ninf := h(wpFloat3{math.Inf(-1), 1, 2})
		nan := h(wpFloat3{math.NaN(), 1, 2})
		require.NotEqual(t, inf, nan, "+Inf was folded into NaN")
		require.NotEqual(t, ninf, nan, "-Inf was folded into NaN")
		require.NotEqual(t, inf, ninf, "+Inf and -Inf hashed alike")
	})
}

// TestWordPathCanonicalizesFloat32 is the float32 half of the above.
func TestWordPathCanonicalizesFloat32(t *testing.T) {
	const seed = 0x13198a2e03707344
	fn := GenerateHashFunction(reflect.TypeOf(wpFloat32x3{}))
	require.NotNil(t, fn)
	h := func(v wpFloat32x3) uint64 { return fn(unsafe.Pointer(&v), seed) }

	negZero := float32(math.Copysign(0, -1))
	require.Equal(t, h(wpFloat32x3{0, 1, 2}), h(wpFloat32x3{negZero, 1, 2}))
	require.Equal(t, h(wpFloat32x3{1, 0, 2}), h(wpFloat32x3{1, negZero, 2}))

	nans := []float32{
		float32(math.NaN()),
		math.Float32frombits(0x7fc00000),
		math.Float32frombits(0x7fffffff),
		math.Float32frombits(0xffc00001),
		math.Float32frombits(0x7f800001),
	}
	want := h(wpFloat32x3{nans[0], 1, 2})
	for _, n := range nans[1:] {
		require.Equalf(t, want, h(wpFloat32x3{n, 1, 2}),
			"float32 NaN %#x hashed apart", math.Float32bits(n))
	}

	pinf := float32(math.Inf(1))
	ninf := float32(math.Inf(-1))
	require.NotEqual(t, h(wpFloat32x3{pinf, 1, 2}), h(wpFloat32x3{nans[0], 1, 2}))
	require.NotEqual(t, h(wpFloat32x3{pinf, 1, 2}), h(wpFloat32x3{ninf, 1, 2}))
}

// TestCanonFromBitsMatchesTheFloatForm holds the bit-testing canonicalization
// against the float-comparing form this package has always used, over every
// special pattern and a large sweep of ordinary ones. The two must agree
// exactly; the bit form exists only because it is cheaper.
func TestCanonFromBitsMatchesTheFloatForm(t *testing.T) {
	special := []uint64{
		0, 0x8000000000000000, // ±0
		0x3ff0000000000000, 0xbff0000000000000, // ±1
		0x7ff0000000000000, 0xfff0000000000000, // ±Inf
		0x7ff8000000000000, 0xfff8000000000000, // quiet NaN
		0x7ff0000000000001, 0xffffffffffffffff, // signalling NaN, all ones
		0x0000000000000001, 0x8000000000000001, // ±smallest subnormal
		0x000fffffffffffff, 0x0010000000000000, // subnormal/normal boundary
	}
	for _, u := range special {
		require.Equalf(t, refCanonF64(math.Float64frombits(u)), canonF64FromBits(u),
			"canonF64FromBits disagrees on the special pattern %#016x", u)
	}

	// A deterministic sweep across exponents and mantissas.
	for exp := range uint64(2048) {
		for _, mant := range []uint64{0, 1, 0x8000000000000, 0xfffffffffffff} {
			for _, sign := range []uint64{0, 1 << 63} {
				u := sign | exp<<52 | mant
				require.Equalf(t, refCanonF64(math.Float64frombits(u)), canonF64FromBits(u),
					"canonF64FromBits disagrees on %#016x", u)
			}
		}
	}
}

// TestCanonF32FromBitsMatchesTheFloatForm is the float32 half, run over the
// entire exponent range and the same mantissa corners.
func TestCanonF32FromBitsMatchesTheFloatForm(t *testing.T) {
	for exp := range uint32(256) {
		for _, mant := range []uint32{0, 1, 0x400000, 0x7fffff} {
			for _, sign := range []uint32{0, 1 << 31} {
				u := sign | exp<<23 | mant
				want := uint32(refCanonF32(math.Float32frombits(u))) //nolint:gosec
				require.Equalf(t, want, canonF32FromBits(u),
					"canonF32FromBits disagrees on %#08x", u)
			}
		}
	}
}

// ── Structural properties ──────────────────────────────────────────────────

// TestWordPathReadsEveryField flips every bit of every field of every type
// that takes the word path and requires the hash to move. A plan that dropped
// a field, or read one twice instead of reading its neighbour, fails here.
func TestWordPathReadsEveryField(t *testing.T) {
	const seed = 0xa4093822299f31d0

	check := func(t *testing.T, ty reflect.Type, size uintptr, determinate func(uintptr) bool) {
		t.Helper()
		fn := GenerateHashFunction(ty)
		require.NotNil(t, fn)

		buf := make([]byte, size)
		for i := range buf {
			buf[i] = byte(i * 7)
		}
		p := unsafe.Pointer(&buf[0])
		base := fn(p, seed)

		for off := uintptr(0); off < size; off++ {
			if !determinate(off) {
				continue
			}
			for bit := range 8 {
				buf[off] ^= 1 << bit
				got := fn(p, seed)
				buf[off] ^= 1 << bit
				require.NotEqualf(t, base, got,
					"%s ignores bit %d of byte %d", ty, bit, off)
			}
		}
	}

	t.Run("three float64", func(t *testing.T) {
		ty := reflect.TypeOf(wpFloat3{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
	t.Run("six float64", func(t *testing.T) {
		ty := reflect.TypeOf(wpFloat6{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
	t.Run("three float32", func(t *testing.T) {
		ty := reflect.TypeOf(wpFloat32x3{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
	t.Run("two complex128", func(t *testing.T) {
		ty := reflect.TypeOf(wpComplex2{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
	t.Run("twelve float64", func(t *testing.T) {
		ty := reflect.TypeOf(wpTwelveFloats{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
	t.Run("merged block and float64", func(t *testing.T) {
		ty := reflect.TypeOf(wpBlock16Float{})
		check(t, ty, ty.Size(), func(uintptr) bool { return true })
	})
}

// TestWordPathDoesNotReadPadding is the other half of the previous test: a
// byte that is padding must not reach the hash, because its content is
// whatever happened to be in that memory and two equal values would hash
// apart. Only fields the layout actually covers may matter.
func TestWordPathDoesNotReadPadding(t *testing.T) {
	const seed = 0x082efa98ec4e6c89

	// int32 then float64 leaves four bytes of padding at offset 4. This type
	// keeps the chain rather than taking the word path, and must still ignore
	// the padding.
	ty := reflect.TypeOf(wpNarrowBlockFloat{})
	require.Equal(t, uintptr(4), ty.Field(1).Offset-4, "layout assumption changed")

	fn := GenerateHashFunction(ty)
	require.NotNil(t, fn)
	buf := make([]byte, ty.Size())
	p := unsafe.Pointer(&buf[0])
	base := fn(p, seed)
	for off := uintptr(4); off < 8; off++ {
		for bit := range 8 {
			buf[off] ^= 1 << bit
			require.Equalf(t, base, fn(p, seed),
				"bit %d of padding byte %d reached the hash", bit, off)
			buf[off] ^= 1 << bit
		}
	}
}

// TestWordPathOrderMatters requires that moving a value from one field to
// another of the same type changes the hash. Without this, {X: 1, Y: 2} and
// {X: 2, Y: 1} would be one key.
func TestWordPathOrderMatters(t *testing.T) {
	const seed = 0x452821e638d01377
	fn3 := GenerateHashFunction(reflect.TypeOf(wpFloat3{}))
	h3 := func(x, y, z float64) uint64 {
		v := wpFloat3{x, y, z}
		return fn3(unsafe.Pointer(&v), seed)
	}
	require.NotEqual(t, h3(1, 2, 3), h3(2, 1, 3))
	require.NotEqual(t, h3(1, 2, 3), h3(1, 3, 2))
	require.NotEqual(t, h3(1, 2, 3), h3(3, 2, 1))

	fn32 := GenerateHashFunction(reflect.TypeOf(wpFloat32x3{}))
	h32 := func(r, g, b float32) uint64 {
		v := wpFloat32x3{r, g, b}
		return fn32(unsafe.Pointer(&v), seed)
	}
	require.NotEqual(t, h32(1, 2, 3), h32(2, 1, 3))
	require.NotEqual(t, h32(1, 2, 3), h32(1, 3, 2))
}

// TestWordPathRespondsToTheSeed requires distinct hashes across many seeds for
// a fixed value, on every type that takes the path. A plan that lost the seed
// would give every set the same layout, which is the one defect a reseed
// cannot repair.
func TestWordPathRespondsToTheSeed(t *testing.T) {
	// Pointer-free types can be filled with arbitrary bytes. A type holding a
	// string cannot: a fabricated string header is a wild pointer, so those
	// are built as real values below.
	for _, ty := range []reflect.Type{
		reflect.TypeOf(wpFloat3{}), reflect.TypeOf(wpFloat6{}),
		reflect.TypeOf(wpFloat32x3{}), reflect.TypeOf(wpIntFloat{}),
		reflect.TypeOf(wpComplex2{}), reflect.TypeOf(wpComplex64x2{}),
		reflect.TypeOf(wpTwelveFloats{}), reflect.TypeOf([4]float64{}),
	} {
		fn := GenerateHashFunction(ty)
		require.NotNilf(t, fn, "%s", ty)
		buf := make([]byte, ty.Size())
		for i := range buf {
			buf[i] = byte(3*i + 1)
		}
		requireSeedsGiveDistinctHashes(t, ty.String(), fn, unsafe.Pointer(&buf[0]))
	}

	v := wpFloatString{Lat: 48.2, Lon: 16.37, Name: "vienna"}
	fn := GenerateHashFunction(reflect.TypeOf(v))
	require.NotNil(t, fn)
	requireSeedsGiveDistinctHashes(t, "wpFloatString", fn, unsafe.Pointer(&v))
}

func requireSeedsGiveDistinctHashes(t *testing.T, name string, fn HashFunction, p unsafe.Pointer) {
	t.Helper()
	seen := make(map[uint64]uint64, 256)
	for s := range uint64(256) {
		seed := s*0x9e3779b97f4a7c15 + 1
		h := fn(p, seed)
		if prev, dup := seen[h]; dup {
			t.Fatalf("%s gave %#x for seeds %#x and %#x", name, h, prev, seed)
		}
		seen[h] = seed
	}
}

// TestWordPathSpreadsNeighbouringValues checks that values differing in one
// field by one unit do not pile into the same bucket. It is a weak statistical
// property and a strong smoke test: a plan that folded two fields together, or
// forgot to mix at the end, shows up immediately.
func TestWordPathSpreadsNeighbouringValues(t *testing.T) {
	const seed = 0xbe5466cf34e90c6c
	const n = 4096
	const buckets = 256

	fn := GenerateHashFunction(reflect.TypeOf(wpFloat3{}))
	require.NotNil(t, fn)

	counts := make([]int, buckets)
	for i := range n {
		v := wpFloat3{float64(i), 1, 2}
		counts[fn(unsafe.Pointer(&v), seed)%buckets]++
	}

	// chi-square against uniform; the 0.1% upper tail for 255 degrees of
	// freedom is about 341.
	expected := float64(n) / buckets
	var chi2 float64
	for _, c := range counts {
		d := float64(c) - expected
		chi2 += d * d / expected
	}
	require.Lessf(t, chi2, 341.0,
		"consecutive values spread poorly over %d buckets, chi2 = %.1f", buckets, chi2)
}

// ── The two properties the speed depends on ────────────────────────────────

// TestWordReadersStayInlinable asks the compiler whether the word readers are
// still inlinable, because the whole design turns on it. Measured with the
// call left standing, a six-word plan cost 9.2 ns against 3.7 with it inlined:
// every word's result is live at once, so a real call makes the register
// allocator spill all of them, and the result is barely better than the chain
// this replaced.
//
// Both readers sit near the inliner's budget of 80 — around 41 and 43 — so an
// edit that adds a branch or a second canonicalization call can push one over
// silently. That is what this test is for.
func TestWordReadersStayInlinable(t *testing.T) {
	if testing.Short() {
		t.Skip("invokes the compiler")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go tool in PATH")
	}

	cmd := exec.Command("go", "build", "-gcflags=-m=2", ".")
	cmd.Dir = "."
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, "go build failed: %s", out)

	for _, name := range []string{"wordAt8", "wordAt4", "canonF64FromBits", "canonF32FromBits"} {
		canInline := regexp.MustCompile(`can inline ` + name + `\b`)
		cannot := regexp.MustCompile(`cannot inline ` + name + `\b`)
		switch {
		case cannot.Match(out):
			t.Errorf("%s is no longer inlinable; the word path loses most of its "+
				"advantage without it. Compiler said:\n%s", name, firstMatchingLine(out, "cannot inline "+name))
		case !canInline.Match(out):
			t.Errorf("could not find an inlining decision for %s in the compiler's "+
				"output; the test needs updating", name)
		}
	}
}

func firstMatchingLine(out []byte, needle string) string {
	re := regexp.MustCompile(`(?m)^.*` + regexp.QuoteMeta(needle) + `.*$`)
	if m := re.Find(out); m != nil {
		return string(m)
	}
	return "(not found)"
}

// TestWordPlanAllocatesNothing checks that hashing does not allocate, for a
// plan small enough for a straight-line mixer and for one that goes through
// the stack buffer. The buffer is a local array handed to wyBlock as a
// pointer; if anything made it escape, every hash of such a type would
// allocate.
func TestWordPlanAllocatesNothing(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("allocation counting is unreliable here")
	}

	t.Run("straight-line mixer", func(t *testing.T) {
		fn := GenerateHashFunction(reflect.TypeOf(wpFloat6{}))
		v := wpFloat6{1, 2, 3, 4, 5, 6}
		p := unsafe.Pointer(&v)
		requireNoAllocs(t, func() { sinkWP = fn(p, 0x1234) })
	})

	t.Run("stack buffer", func(t *testing.T) {
		fn := GenerateHashFunction(reflect.TypeOf(wpTwelveFloats{}))
		v := wpTwelveFloats{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		p := unsafe.Pointer(&v)
		requireNoAllocs(t, func() { sinkWP = fn(p, 0x1234) })
	})
}

var sinkWP uint64

// TestEveryPlanArityMatchesTheWordSequence walks every arity both closure
// families emit, including the stack-buffer path past eight words.
//
// Each arity is a separate hand-written body — the whole point of the design is
// that the loads are unrolled — so a transcription slip in one of them would be
// invisible to a test that only exercised the arities a few example structs
// happen to produce. This reads the words independently and requires the
// closure to agree with the generic routine over them.
func TestEveryPlanArityMatchesTheWordSequence(t *testing.T) {
	seeds := []uint64{0, 1, 0x9e3779b97f4a7c15, ^uint64(0)}

	t.Run("eight-byte words", func(t *testing.T) {
		// One backing array, read at k different offsets. Interleaving the
		// offsets rather than walking them in order also checks that each body
		// uses the offset belonging to its own word.
		const slots = maxPlanWords + 2
		raw := make([]uint64, slots)
		for i := range raw {
			raw[i] = uint64(i+1) * 0xff51afd7ed558ccd
		}
		p := unsafe.Pointer(&raw[0])

		for k := 2; k <= maxPlanWords; k++ {
			for _, canon := range []uint16{0, 0xffff, 0x5555, 0xaaaa} {
				offs := make([]uintptr, k)
				words := make([]uint64, k)
				for i := range k {
					// A deliberately non-monotone offset order.
					slot := (i*7 + 3) % slots
					offs[i] = uintptr(slot) * 8 //nolint:gosec
					w := raw[slot]
					if canon>>uint(i)&1 != 0 { //nolint:gosec
						w = canonF64FromBits(w)
					}
					words[i] = w
				}
				fn := cheapWordHasher(wordPlan{offs: offs, canon: canon})
				for _, seed := range seeds {
					require.Equalf(t, mixWordSlice(words, seed), fn(p, seed),
						"wide plan of %d words, canon %#04x, seed %#x", k, canon, seed)
				}
			}
		}
	})

	t.Run("four-byte words", func(t *testing.T) {
		const slots = maxPlanWords + 2
		raw := make([]uint32, slots)
		for i := range raw {
			raw[i] = uint32(i+1) * 0x9e3779b1 //nolint:gosec
		}
		p := unsafe.Pointer(&raw[0])

		for k := 2; k <= maxPlanWords; k++ {
			offs := make([]uintptr, k)
			words := make([]uint64, k)
			for i := range k {
				slot := (i*5 + 1) % slots
				offs[i] = uintptr(slot) * 4 //nolint:gosec
				words[i] = uint64(canonF32FromBits(raw[slot]))
			}
			fn := cheapWordHasher(wordPlan{offs: offs, narrow: true})
			for _, seed := range seeds {
				require.Equalf(t, mixWordSlice(words, seed), fn(p, seed),
					"narrow plan of %d words, seed %#x", k, seed)
			}
		}
	})
}

// TestNarrowPlansAreAlwaysCanonicalized pins the invariant wordAt4 relies on:
// a four-byte word is always a float32, so there is no raw four-byte word and
// no branch for one. If a future classification gives byte-stable runs narrow
// words, this fails and says what has to come back.
func TestNarrowPlansAreAlwaysCanonicalized(t *testing.T) {
	types := []reflect.Type{
		reflect.TypeOf(wpFloat32x3{}),
		reflect.TypeOf(wpComplex64x2{}),
		reflect.TypeOf([5]float32{}),
		reflect.TypeOf([3]complex64{}),
		reflect.TypeOf(struct{ A, B, C, D float32 }{}),
	}
	found := false
	for _, ty := range types {
		pl, _, ok := planWords(mergeByteBlocks(flattenTypeOps(ty, 0)))
		if !ok || !pl.narrow {
			continue
		}
		found = true
		full := uint16(1)<<uint(len(pl.offs)) - 1 //nolint:gosec
		require.Equalf(t, full, pl.canon&full,
			"%s produced a narrow plan with an uncanonicalized word; wordAt4 has no branch for that",
			ty)
	}
	require.True(t, found, "no narrow plan was produced, so the invariant went untested")
}

// TestPlanWordsRefusesAnUnknownOpKind covers the guard that matters most for
// anyone extending this package. If a new micro-op kind is added and planWords
// is not taught about it, the field must not silently vanish from the hash —
// two different values would then compare unequal and hash equal, and the set
// would return wrong answers rather than slow ones. The plan has to decline so
// that the chain, which dispatches on the same kinds, keeps the field.
func TestPlanWordsRefusesAnUnknownOpKind(t *testing.T) {
	unknown := opKind(200) // deliberately not one of the declared kinds
	require.Greater(t, unknown, opString, "pick a kind value above every real one")

	ops := []microOp{
		{kind: opFloat64, offset: 0},
		{kind: opFloat64, offset: 8},
		{kind: unknown, offset: 16},
	}
	_, _, ok := planWords(ops)
	require.False(t, ok, "planWords accepted a plan containing an unknown op kind")
	require.Nil(t, buildWordClosure(ops), "buildWordClosure must decline as well")
}

// ── The path a declining type still takes ──────────────────────────────────
//
// The tests below exercise the seed-threading chain, which remains the shipping
// path for every type the word plan declines. They exist because introducing
// the word path took the old tests off that chain: the struct shapes they used
// now qualify for words, so nothing was left driving the chain's float and
// complex canonicalization through its special cases.

// TestChainStillCanonicalizesFloats checks ±0 and NaN on a type that declines
// the word path, so the canonicalization under test is the chain's own — the
// float-comparing form in hashgen.go, not canonF64FromBits.
func TestChainStillCanonicalizesFloats(t *testing.T) {
	const seed = 0x9e3779b97f4a7c15

	// float64 beside float32 is a mixed-width plan, which the word path
	// declines; both fields therefore go through the chain.
	ty := reflect.TypeOf(wpMixedWidth{})
	require.Nil(t, buildWordClosure(mergeByteBlocks(flattenTypeOps(ty, 0))),
		"this test needs a type that declines the word path")
	fn := GenerateHashFunction(ty)
	require.NotNil(t, fn)
	h := func(a float64, b float32) uint64 {
		v := wpMixedWidth{A: a, B: b}
		return fn(unsafe.Pointer(&v), seed)
	}

	negZero64 := math.Copysign(0, -1)
	negZero32 := float32(negZero64)
	require.Equal(t, h(0, 1), h(negZero64, 1), "chain kept -0.0 apart from +0.0 in the float64 field")
	require.Equal(t, h(1, 0), h(1, negZero32), "chain kept -0.0 apart from +0.0 in the float32 field")

	require.Equal(t, h(math.NaN(), 1), h(math.Float64frombits(0x7fffffffffffffff), 1),
		"chain kept two float64 NaNs apart")
	require.Equal(t, h(1, float32(math.NaN())), h(1, math.Float32frombits(0x7fffffff)),
		"chain kept two float32 NaNs apart")

	require.NotEqual(t, h(math.Inf(1), 1), h(math.NaN(), 1), "chain folded +Inf into NaN")
	require.NotEqual(t, h(1, float32(math.Inf(1))), h(1, float32(math.NaN())),
		"chain folded float32 +Inf into NaN")
}

// TestChainStillCanonicalizesComplex is the same for the complex hashers,
// which the word path only declines when the plan as a whole does.
func TestChainStillCanonicalizesComplex(t *testing.T) {
	const seed = 0x452821e638d01377

	// A float32 alongside a complex128 makes the plan mixed-width, and a
	// float64 alongside a complex64 does the same the other way round, so both
	// types go through the chain.
	type mixedComplex128 struct {
		C complex128
		F float32
	}
	type mixedComplex64 struct {
		C complex64
		F float64
	}

	requireDeclines := func(t *testing.T, ty reflect.Type) {
		t.Helper()
		require.Nilf(t, buildWordClosure(mergeByteBlocks(flattenTypeOps(ty, 0))),
			"this test needs %s to decline the word path", ty)
	}

	t.Run("complex128", func(t *testing.T) {
		ty := reflect.TypeOf(mixedComplex128{})
		requireDeclines(t, ty)
		fn := GenerateHashFunction(ty)
		require.NotNil(t, fn)
		h := func(re, im float64) uint64 {
			v := mixedComplex128{C: complex(re, im), F: 1}
			return fn(unsafe.Pointer(&v), seed)
		}
		requireComplexCanonicalization(t, h)
	})

	t.Run("complex64", func(t *testing.T) {
		ty := reflect.TypeOf(mixedComplex64{})
		requireDeclines(t, ty)
		fn := GenerateHashFunction(ty)
		require.NotNil(t, fn)
		h := func(re, im float64) uint64 {
			v := mixedComplex64{C: complex64(complex(re, im)), F: 1}
			return fn(unsafe.Pointer(&v), seed)
		}
		requireComplexCanonicalization(t, h)
	})
}

// requireComplexCanonicalization checks ±0 and NaN in both components.
func requireComplexCanonicalization(t *testing.T, h func(re, im float64) uint64) {
	t.Helper()
	negZero := math.Copysign(0, -1)
	require.Equal(t, h(0, 1), h(negZero, 1), "-0.0 real part hashed apart from +0.0")
	require.Equal(t, h(1, 0), h(1, negZero), "-0.0 imaginary part hashed apart from +0.0")
	require.Equal(t, h(math.NaN(), 1), h(math.Float64frombits(0x7fffffffffffffff), 1),
		"two NaN real parts hashed apart")
	require.Equal(t, h(1, math.NaN()), h(1, math.Float64frombits(0x7fffffffffffffff)),
		"two NaN imaginary parts hashed apart")
	require.NotEqual(t, h(math.Inf(1), 1), h(math.NaN(), 1), "+Inf folded into NaN")
	require.NotEqual(t, h(1, 2), h(2, 1), "the two components are interchangeable")
}

// TestArrayOfStringsTakesTheChain covers the array builder's fallback. An array
// whose elements cannot become words has no plan at all, and the ops must still
// reach a working closure rather than nil.
func TestArrayOfStringsTakesTheChain(t *testing.T) {
	const seed = 0xbe5466cf34e90c6c
	ty := reflect.TypeOf([3]string{})
	require.Nil(t, buildWordClosure(mergeByteBlocks(flattenTypeOps(ty, 0))),
		"an array of strings should produce no word plan")

	fn := GenerateHashFunction(ty)
	require.NotNil(t, fn, "the array builder must fall back to the chain")

	a := [3]string{"alpha", "beta", "gamma"}
	b := [3]string{"alpha", "beta", "gamma"}
	c := [3]string{"beta", "alpha", "gamma"} // same contents, different order
	require.Equal(t, fn(unsafe.Pointer(&a), seed), fn(unsafe.Pointer(&b), seed),
		"equal arrays hashed apart")
	require.NotEqual(t, fn(unsafe.Pointer(&a), seed), fn(unsafe.Pointer(&c), seed),
		"the chain is blind to element order")
}

// ── Hash quality ───────────────────────────────────────────────────────────
//
// The mixing itself needs no separate quality argument: it is wyBlock, held
// bit-for-bit equal to it by TestWordMixersAreTheGenericPath, and that routine's
// quality is measured by the Go authors' SMHasher port in lab/hashquality. What
// is new here is the mapping from fields to words, and what it has to be is
// injective. The tests below measure the composition end to end anyway, because
// an argument is not a measurement.

// TestWordPathAvalanche requires that flipping one input bit flips each output
// bit about half the time.
//
// Only mantissa bits are flipped, and only on ordinary finite values. That is
// not a convenience: flipping an exponent bit can turn a field into a NaN, and
// flipping the sign bit of a zero turns -0.0 into +0.0 — both of which the hash
// is *required* to fold together, so those flips legitimately change nothing. A
// mantissa flip of a normal number always yields a different normal number, so
// every such flip must reach the output.
func TestWordPathAvalanche(t *testing.T) {
	const seed = 0x243f6a8885a308d3
	samples := 1 << 12
	if testing.Short() {
		samples = 1 << 9
	}

	for _, n := range []int{2, 3, 4, 6, 8} {
		ty := reflect.ArrayOf(n, reflect.TypeOf(float64(0)))
		fn := GenerateHashFunction(ty)
		require.NotNilf(t, fn, "%s", ty)

		v := make([]float64, n)
		p := unsafe.Pointer(&v[0])
		raw := unsafe.Slice((*uint64)(p), n)

		var flips [64]uint64
		var pairs uint64
		state := uint64(0x5eed_1a4e)
		next := func() uint64 {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			return state
		}

		for range samples {
			// Ordinary finite values: exponent fixed to something normal, a
			// random mantissa, a random sign.
			for i := range n {
				raw[i] = (next()&1)<<63 | uint64(0x400)<<52 | next()&0x000fffffffffffff
			}
			h0 := fn(p, seed)
			for field := range n {
				for bit := range 52 {
					raw[field] ^= 1 << bit
					diff := h0 ^ fn(p, seed)
					raw[field] ^= 1 << bit
					pairs++
					for ob := range 64 {
						flips[ob] += (diff >> ob) & 1
					}
				}
			}
		}

		np := float64(pairs)
		var maxDev float64
		for _, c := range flips {
			maxDev = math.Max(maxDev, math.Abs(float64(c)/np-0.5))
		}
		// Three standard errors of a fair coin over the pairs measured, with a
		// floor so the bound stays meaningful at large samples.
		threshold := math.Max(0.004, 3/math.Sqrt(np))
		require.Lessf(t, maxDev, threshold,
			"%s: worst output bit deviates %.5f from half over %d flip pairs (threshold %.5f)",
			ty, maxDev, pairs, threshold)
	}
}

// TestWordPathIsCollisionFreeOverManyValues hashes a million distinct values of
// several shapes and requires no collision at all.
//
// A 64-bit hash over 2^20 distinct inputs expects about 2^-25 collisions, so
// one collision is already evidence of structure rather than bad luck. What it
// would catch is the failure mode a word plan can actually have: two fields
// folded into one word, a word read twice instead of reading its neighbour, or
// a lane key shared between positions.
func TestWordPathIsCollisionFreeOverManyValues(t *testing.T) {
	const seed = 0x13198a2e03707344
	count := 1 << 20
	spanA, spanB := 128, 128 // count / (spanA*spanB) values for the last field
	if testing.Short() {
		count = 1 << 16
		spanA, spanB = 32, 32
	}
	// Every field has to vary across its own range, or a plan that dropped the
	// last one would still produce no collision and this test would pass on a
	// broken hash. Verified by injecting exactly that fault.
	require.Greater(t, count/(spanA*spanB), 1, "the last field would be constant")

	t.Run("three float64", func(t *testing.T) {
		fn := GenerateHashFunction(reflect.TypeOf(wpFloat3{}))
		seen := make(map[uint64]wpFloat3, count)
		for i := range count {
			v := wpFloat3{
				X: float64(i % spanA),
				Y: float64((i / spanA) % spanB),
				Z: float64(i / (spanA * spanB)),
			}
			h := fn(unsafe.Pointer(&v), seed)
			if prev, dup := seen[h]; dup {
				t.Fatalf("collision: %+v and %+v both hash to %#x", prev, v, h)
			}
			seen[h] = v
		}
	})

	t.Run("three float32", func(t *testing.T) {
		fn := GenerateHashFunction(reflect.TypeOf(wpFloat32x3{}))
		seen := make(map[uint64]wpFloat32x3, count)
		for i := range count {
			v := wpFloat32x3{
				R: float32(i % spanA),
				G: float32((i / spanA) % spanB),
				B: float32(i / (spanA * spanB)),
			}
			h := fn(unsafe.Pointer(&v), seed)
			if prev, dup := seen[h]; dup {
				t.Fatalf("collision: %+v and %+v both hash to %#x", prev, v, h)
			}
			seen[h] = v
		}
	})

	t.Run("int64 pair and float64", func(t *testing.T) {
		fn := GenerateHashFunction(reflect.TypeOf(wpBlock16Float{}))
		seen := make(map[uint64]wpBlock16Float, count)
		for i := range count {
			v := wpBlock16Float{
				A: int64(i % spanA),           //nolint:gosec
				B: int64((i / spanA) % spanB), //nolint:gosec
				F: float64(i / (spanA * spanB)),
			}
			h := fn(unsafe.Pointer(&v), seed)
			if prev, dup := seen[h]; dup {
				t.Fatalf("collision: %+v and %+v both hash to %#x", prev, v, h)
			}
			seen[h] = v
		}
	})
}

// TestWordPathBucketsAreUniform is the chi-square test the library applies to
// its byte hash, applied to a struct hasher: the low bits are what selects a
// group, so they have to be flat for keys that differ in a structured way.
func TestWordPathBucketsAreUniform(t *testing.T) {
	const seed = 0x082efa98ec4e6c89
	const buckets = 1024
	const n = buckets * 64 // 64 expected per bucket

	shapes := []struct {
		name string
		hash func(i int) uint64
	}{
		{"three float64, one field varying", func(i int) uint64 {
			fn := generatedFor[wpFloat3]()
			v := wpFloat3{X: float64(i), Y: 2.5, Z: -7}
			return fn(unsafe.Pointer(&v), seed)
		}},
		{"three float64, all fields varying", func(i int) uint64 {
			fn := generatedFor[wpFloat3]()
			v := wpFloat3{X: float64(i % 64), Y: float64((i / 64) % 64), Z: float64(i / 4096)}
			return fn(unsafe.Pointer(&v), seed)
		}},
		{"three float32", func(i int) uint64 {
			fn := generatedFor[wpFloat32x3]()
			v := wpFloat32x3{R: float32(i % 64), G: float32((i / 64) % 64), B: float32(i / 4096)}
			return fn(unsafe.Pointer(&v), seed)
		}},
		{"two complex128", func(i int) uint64 {
			fn := generatedFor[wpComplex2]()
			v := wpComplex2{A: complex(float64(i%1024), 1), B: complex(2, float64(i/1024))}
			return fn(unsafe.Pointer(&v), seed)
		}},
	}

	for _, sh := range shapes {
		t.Run(sh.name, func(t *testing.T) {
			counts := make([]int, buckets)
			for i := range n {
				counts[sh.hash(i)%buckets]++
			}
			expected := float64(n) / buckets
			var chi2 float64
			for _, c := range counts {
				d := float64(c) - expected
				chi2 += d * d / expected
			}
			// The 0.1% upper tail for 1023 degrees of freedom is about 1168.
			require.Lessf(t, chi2, 1168.0,
				"chi2 = %.1f over %d buckets, %d samples", chi2, buckets, n)
		})
	}
}

// generatedFor is a cached lookup so the chi-square loops do not rebuild the
// closure per sample. GenerateHashFunction caches by type already; this just
// keeps the call out of the hot loop's line of sight.
func generatedFor[T comparable]() HashFunction {
	var zero T
	return GenerateHashFunction(reflect.TypeOf(zero))
}

// TestWordPathThroughNestedTypes checks the offsets a recursive flatten
// produces. A nested struct's fields are reached at their absolute offset in
// the outer value, which is precisely where an off-by-a-field would hide: the
// hash would still look healthy, still respond to the seed, and still be
// sensitive to most bits, while reading the wrong field.
func TestWordPathThroughNestedTypes(t *testing.T) {
	type inner struct{ A, B float64 }
	type outer struct {
		I inner
		C float64
		J inner
	}

	seeds := []uint64{0, 1, 0x9e3779b97f4a7c15}
	v := outer{I: inner{1.5, 2.5}, C: 3.5, J: inner{4.5, 5.5}}

	// Five float64 words in declaration order, at the offsets the layout gives.
	requireGenerated(t, v, func(seed uint64) uint64 {
		return mixWordSlice([]uint64{
			refCanonF64(v.I.A), refCanonF64(v.I.B), refCanonF64(v.C),
			refCanonF64(v.J.A), refCanonF64(v.J.B),
		}, seed)
	}, seeds)

	// And an array of the nested struct, which goes through flattenArrayOps.
	type pair = inner
	arr := [3]pair{{1, 2}, {3, 4}, {5, 6}}
	requireGenerated(t, arr, func(seed uint64) uint64 {
		w := make([]uint64, 0, 6)
		for _, e := range arr {
			w = append(w, refCanonF64(e.A), refCanonF64(e.B))
		}
		return mixWordSlice(w, seed)
	}, seeds)

	// Every field of the nested shape must reach the hash, at every bit.
	ty := reflect.TypeOf(v)
	fn := GenerateHashFunction(ty)
	require.NotNil(t, fn)
	buf := make([]byte, ty.Size())
	for i := range buf {
		buf[i] = byte(5*i + 2)
	}
	p := unsafe.Pointer(&buf[0])
	base := fn(p, 0x452821e638d01377)
	for off := range ty.Size() {
		for bit := range 8 {
			buf[off] ^= 1 << bit
			got := fn(p, 0x452821e638d01377)
			buf[off] ^= 1 << bit
			require.NotEqualf(t, base, got, "nested shape ignores bit %d of byte %d", bit, off)
		}
	}
}

// TestWordPathReordersWordsBeforeThreadedOps pins the composition order when a
// threaded field comes *first* in the declaration. The words are hashed before
// it, so the emitted closure does not follow declaration order — that is
// allowed, because the reordering is fixed for a type, but it is worth stating
// once and holding, since it is the thing that would change if the composition
// were ever rewritten.
func TestWordPathReordersWordsBeforeThreadedOps(t *testing.T) {
	type nameFirst struct {
		Name     string
		Lat, Lon float64
	}
	v := nameFirst{Name: "vienna", Lat: 48.2, Lon: 16.37}

	requireGenerated(t, v, func(seed uint64) uint64 {
		// The two floats first, then the string threaded from their result,
		// even though the string is declared before them.
		h := mixWordSlice([]uint64{refCanonF64(v.Lat), refCanonF64(v.Lon)}, seed)
		return HashString(unsafe.Pointer(&v.Name), h)
	}, []uint64{0, 1, 0xdeadbeef})

	// The reordering must not make two distinct values collide: a value whose
	// name and coordinates are swapped around is still a different value.
	fn := GenerateHashFunction(reflect.TypeOf(v))
	h := func(name string, lat, lon float64) uint64 {
		x := nameFirst{Name: name, Lat: lat, Lon: lon}
		return fn(unsafe.Pointer(&x), 0x1234)
	}
	require.NotEqual(t, h("vienna", 48.2, 16.37), h("vienna", 16.37, 48.2))
	require.NotEqual(t, h("vienna", 48.2, 16.37), h("berlin", 48.2, 16.37))
}
