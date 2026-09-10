package hashing

import (
	"reflect"
	"unsafe"
)

// HashFunction is the runtime function used to hash values. It receives a
// pointer to the value and a uint64 seed; it returns a uint64 hash.
// Implementations must treat the memory at the pointer as the concrete
// representation of the value and incorporate the seed to allow
// deterministic re-seeding.
//
// Implementations must satisfy two contracts:
//
//   - Equal values must hash equally. If a == b then the function must
//     return the same value for both, for every seed.
//   - The pointer must not be retained. [RuntimeHasher.Hash] hides it from
//     escape analysis with [Noescape] so that the key stays on the caller's
//     stack. An implementation that stores p, or anything derived from it,
//     beyond the duration of the call creates a dangling pointer that the
//     compiler cannot catch. Read through p, return, keep nothing.
//
// [HashBool] deliberately ignores the seed: with only two possible values a
// seeded hash gains nothing, so reseeding a set of bools is a no-op.
type HashFunction func(unsafe.Pointer, uint64) uint64

// RuntimeHasher holds a per-type runtime hash function and a seed.
// It is intended to be created by [MakeRuntimeHasher] and called by the
// generic Set implementation to compute element hashes efficiently.
type RuntimeHasher[K comparable] struct {
	Seed uint64
	fn   HashFunction
}

// Hash computes the hash for key k using the stored runtime function and seed.
//
// k is taken by value, and its address is passed to the hash function. Because
// the call goes through a function value the compiler cannot look inside, that
// address would escape and every hash would heap-allocate a copy of the key --
// [Noescape] is what prevents it. TestRuntimeHasherHashDoesNotAllocate holds
// that guarantee to account.
//
// Returns the hash of k under the seed this hasher was built with. There is no
// error case; every value of a comparable type is hashable.
//
// Keep this a single expression. Splitting the call into "p := Noescape(...)"
// followed by "return h.fn(p, h.Seed)" pushes the inline cost of the shape
// instantiation past the compiler's budget of 80, and Hash then costs the
// caller a real call frame on top of the indirect hash call. It measures:
// Set3[uint64].Contains 3.67 ns inlined against 4.41 ns not, on Go 1.26.8.
// BenchmarkPlainU64 is the narrow version of the same measurement.
func (h RuntimeHasher[K]) Hash(k K) uint64 {
	return h.fn(Noescape(unsafe.Pointer(&k)), h.Seed) //nolint:gosec
}

// MakeRuntimeHasher chooses an efficient per-type hash function for the
// generic type parameter K. The returned RuntimeHasher contains the provided
// seed and the selected hash function.
//
// Selection happens once, at construction time, in this order:
//
//  1. Interface types (reflect.TypeOf yields nil) use the maphash fallback,
//     which resolves the dynamic type per value.
//  2. Types whose kind is a primitive get a dedicated hasher. Dispatch is by
//     kind rather than by concrete type, so named types such as
//     "type UserID uint64" take the same fast path as uint64.
//  3. Layouts that are safe to hash as a raw byte block (no padding, no
//     floats, no blank fields) use [HashAsByteArray].
//  4. Everything else is handed to [GenerateHashFunction], which builds a
//     reflection-free closure for structs and arrays containing floats,
//     strings or padding.
//  5. Types that generator cannot handle, such as structs with interface
//     fields, use the maphash fallback.
//
// No reflection happens at hash time on any of these paths.
func MakeRuntimeHasher[K comparable](seed uint64) RuntimeHasher[K] {
	h := RuntimeHasher[K]{Seed: seed}
	var zero K

	t := reflect.TypeOf(zero)
	if t == nil {
		// K is an interface type. Since Go 1.20 ordinary interface types
		// satisfy the comparable constraint, so K may legitimately be any or
		// a named interface. The zero value is a nil interface, for which
		// reflect.TypeOf returns nil, so no static layout analysis is
		// possible: dispatch has to happen per value at hash time. maphash
		// handles this (and panics for dynamic types that are not
		// comparable, matching the behaviour of ==).
		h.fn = HashFallbackMaphash[K]
		return h
	}

	// Dispatch on the kind rather than on the concrete type. A type switch
	// over concrete types only matches exact types, so a named type such as
	// "type UserID uint64" would miss every fast path and fall through to the
	// generic byte-block hasher. Switching on the kind gives named types the
	// same specialized hasher as their underlying type.
	switch t.Kind() {
	case reflect.Uint8, reflect.Int8:
		h.fn = SwirlByte
	case reflect.Bool:
		h.fn = HashBool
	case reflect.Uint16, reflect.Int16:
		h.fn = HashI16SM
	case reflect.Uint32, reflect.Int32:
		h.fn = HashI32WHdet
	case reflect.Uint64, reflect.Int64:
		h.fn = HashI64WHdet
	case reflect.Uint, reflect.Int:
		h.fn = HashInt
	case reflect.Uintptr, reflect.Pointer, reflect.UnsafePointer, reflect.Chan:
		// All pointer-shaped scalars; equality is over the pointer value
		// itself, which is exactly what HashPtr mixes.
		h.fn = HashPtr
	case reflect.Float32:
		h.fn = HashF32SM
	case reflect.Float64:
		h.fn = HashF64WHdet
	case reflect.String:
		h.fn = HashString
	default:
		switch {
		case CanUseUnsafeRawByteBlockHasherType(t).Eligible:
			// Fast path for layouts that are safe for raw byte-block hashing
			// according to Go equality semantics.
			h.fn = HashAsByteArray[K]
		default:
			// Reflection-based generator produces a fast, type-specific hash
			// closure (e.g. for structs with padding, floats, strings, or
			// complex fields). No reflection happens at hash time. It returns
			// nil for types it cannot handle, e.g. structs with interface
			// fields, which fall back to maphash.
			if fn := GenerateHashFunction(t); fn != nil {
				h.fn = fn
			} else {
				h.fn = HashFallbackMaphash[K]
			}
		}
	}
	return h
}

// Noescape hides the pointer p from escape analysis, preventing it
// from escaping to the heap. It compiles down to nothing.
//
// WARNING: This is very subtle to use correctly. The caller must
// ensure that it's truly safe for p to not escape to the heap by
// maintaining runtime pointer invariants (for example, that globals
// and the heap may not generally point into a stack).
//
// The body is copied verbatim from runtime.noescape (runtime/stubs.go, see
// also internal/abi/escape.go). That is deliberate and worth keeping: the Go
// runtime hashes maps through this exact expression, so no Go release can
// teach escape analysis to see through it without breaking itself first. Other
// spellings of the same laundering -- reading the uintptr back through a
// pointer, for instance -- work today and carry no such guarantee.
//
// The round trip through uintptr is what go vet's unsafeptr check reports, and
// what it is designed to report; there is no way to write this function without
// tripping it. golangci-lint has the one line suppressed with a reason.
// noescape_test.go asserts both halves of the contract: that the pointer comes
// back unchanged, and that callers stay allocation-free.
//
//go:nosplit
//go:nocheckptr
func Noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0) //nolint:govet,gosec,staticcheck // the pointer laundering is the whole point
}
