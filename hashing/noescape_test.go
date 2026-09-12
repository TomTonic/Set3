package hashing

import (
	"math"
	"runtime"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// This file guards [Noescape]. It is the one piece of the package whose job is
// invisible in the source: it returns its argument unchanged, and everything it
// is there for happens in the compiler.
//
// Two things can break, and only one of them is loud. If Noescape ever returned
// a different address, every hash would be garbage and the rest of the suite
// would fail immediately. If the compiler ever starts seeing through the
// laundering, nothing fails -- [RuntimeHasher.Hash] simply starts moving every
// key it is handed to the heap, and Set3 quietly gets an allocation per lookup.
// A Go release can do that to us without a word, so the allocation behaviour is
// asserted here rather than left to a benchmark nobody reads.

// noescapeSmallStruct has no padding, no floats and no blank fields, so
// MakeRuntimeHasher gives it the raw byte block hasher.
type noescapeSmallStruct struct {
	A uint64
	B uint64
}

// noescapeFloatStruct contains a float, so it is routed through
// GenerateHashFunction and gets a built closure instead.
type noescapeFloatStruct struct {
	A uint64
	F float64
}

// noescapeFallbackStruct has an interface field, which the generator refuses,
// so it lands on the maphash fallback path.
type noescapeFallbackStruct struct {
	A uint64
	B any
}

// TestNoescapeReturnsTheSamePointer verifies the part of Noescape that is
// visible in the source: it is the identity function on pointers.
//
// Everything in the hashing package reads the value through the pointer
// Noescape hands back, so an address that differs by even one byte would mean
// every hash is computed over the wrong memory.
//
// It checks pointers into a local, into the heap, into the middle of a slice,
// and one past nothing at all, and requires the returned pointer to compare
// equal to the input every time.
func TestNoescapeReturnsTheSamePointer(t *testing.T) {
	local := uint64(0x0123456789ABCDEF)
	heap := new(uint64)
	slice := make([]byte, 64)
	str := "the quick brown fox"

	cases := []struct {
		name string
		p    unsafe.Pointer
	}{
		{"stack local", unsafe.Pointer(&local)},
		{"heap allocation", unsafe.Pointer(heap)},
		{"slice start", unsafe.Pointer(&slice[0])},
		{"slice interior", unsafe.Pointer(&slice[37])},
		{"string header", unsafe.Pointer(&str)},
		{"nil", nil},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.p, Noescape(c.p), "Noescape must return its argument unchanged")
		})
	}
}

// TestNoescapeKeepsTheValueReadable verifies that memory reached through a
// laundered pointer still holds what the caller put there.
//
// This is the contract every HashFunction in the package relies on: it is
// handed the pointer Noescape returned and reads the key's representation
// straight out of it.
//
// It writes a known bit pattern, passes the address through Noescape, reads it
// back through the returned pointer, and requires the value to be unchanged --
// before and after a garbage collection.
func TestNoescapeKeepsTheValueReadable(t *testing.T) {
	const want = uint64(0xDEADBEEFCAFEBABE)
	v := want

	got := *(*uint64)(Noescape(unsafe.Pointer(&v)))
	require.Equal(t, want, got)

	runtime.GC()

	got = *(*uint64)(Noescape(unsafe.Pointer(&v)))
	require.Equal(t, want, got, "value must survive a GC cycle")
	runtime.KeepAlive(&v)
}

// TestRuntimeHasherHashDoesNotAllocate is the regression guard for the reason
// Noescape exists at all.
//
// Hashing a key must not cost an allocation. RuntimeHasher.Hash calls its hash
// function through a function value, which the compiler cannot look inside, so
// without Noescape the address of the key parameter escapes and every single
// Add, Contains or Remove in Set3 heap-allocates a copy of the key. That is a
// silent, permanent slowdown, not a failure -- so it is asserted, not measured.
//
// It runs Hash for one key type per dispatch path in MakeRuntimeHasher and
// requires zero allocations per call. If a future Go release sees through the
// pointer laundering in Noescape, this is the test that says so.
func TestRuntimeHasherHashDoesNotAllocate(t *testing.T) {
	var sink uint64

	t.Run("uint64 (primitive path)", func(t *testing.T) {
		h := MakeRuntimeHasher[uint64](0x1234)
		requireNoAllocs(t, func() { sink += h.Hash(0x0123456789ABCDEF) })
	})

	t.Run("string", func(t *testing.T) {
		h := MakeRuntimeHasher[string](0x1234)
		key := "the quick brown fox jumps over the lazy dog"
		requireNoAllocs(t, func() { sink += h.Hash(key) })
	})

	t.Run("struct (raw byte block path)", func(t *testing.T) {
		h := MakeRuntimeHasher[noescapeSmallStruct](0x1234)
		key := noescapeSmallStruct{A: 1, B: 2}
		requireNoAllocs(t, func() { sink += h.Hash(key) })
	})

	t.Run("struct with float (generated closure path)", func(t *testing.T) {
		h := MakeRuntimeHasher[noescapeFloatStruct](0x1234)
		key := noescapeFloatStruct{A: 1, F: 3.14159}
		requireNoAllocs(t, func() { sink += h.Hash(key) })
	})

	t.Run("struct with interface field (maphash fallback path)", func(t *testing.T) {
		h := MakeRuntimeHasher[noescapeFallbackStruct](0x1234)
		key := noescapeFallbackStruct{A: 1, B: "boxed once, outside the loop"}
		requireNoAllocs(t, func() { sink += h.Hash(key) })
	})

	runtime.KeepAlive(sink)
}

// TestNoescapeHidesTheKeyFromEscapeAnalysis pins down what the previous test
// measures, by showing the same call site allocating once Noescape is taken out
// of the picture.
//
// Without this, a Go release that made RuntimeHasher.Hash allocation-free for
// some unrelated reason would leave TestRuntimeHasherHashDoesNotAllocate
// passing while proving nothing.
//
// It runs the identical hash call twice -- once through Hash, which launders
// the key pointer, and once through a local copy of Hash that does not -- and
// requires the unlaundered one to allocate. If this ever stops allocating,
// escape analysis has become strong enough to handle the indirect call on its
// own, and Noescape has become unnecessary rather than broken.
func TestNoescapeHidesTheKeyFromEscapeAnalysis(t *testing.T) {
	h := MakeRuntimeHasher[noescapeSmallStruct](0x1234)
	key := noescapeSmallStruct{A: 1, B: 2}
	var sink uint64

	requireNoAllocs(t, func() { sink += h.Hash(key) })

	withoutNoescape := testing.AllocsPerRun(200, func() {
		sink += hashWithoutNoescape(h, key)
	})
	require.Positive(t, withoutNoescape,
		"the same call must allocate without Noescape, otherwise the zero-allocation "+
			"assertion above proves nothing; if escape analysis now handles the indirect "+
			"call on its own, Noescape is obsolete and can go")

	runtime.KeepAlive(sink)
}

// hashWithoutNoescape is RuntimeHasher.Hash with the pointer laundering left
// out. It exists purely as the control case for
// TestNoescapeHidesTheKeyFromEscapeAnalysis and must never be used in
// production code.
//
//go:noinline
func hashWithoutNoescape[K comparable](h RuntimeHasher[K], k K) uint64 {
	return h.fn(unsafe.Pointer(&k), h.Seed)
}

// requireNoAllocs fails the test unless fn performs no heap allocation at all.
//
// It exists because testing.AllocsPerRun returns an average as a float64 and
// the useful assertion is "exactly zero"; reporting the measured average makes
// a failure diagnosable rather than just red.
//
// It takes the minimum over several measurements rather than trusting one.
// AllocsPerRun counts allocations process-wide, so a goroutine from another
// test package running in parallel can attribute its own work here; the
// question being asked is "can this allocate at all", and the minimum is the
// statistic that answers it.
func requireNoAllocs(t *testing.T, fn func()) {
	t.Helper()
	best := math.Inf(1)
	for range 5 {
		if avg := testing.AllocsPerRun(200, fn); avg < best {
			best = avg
		}
	}
	require.Zero(t, best, "expected no allocations per call, measured %.2f", best)
}
