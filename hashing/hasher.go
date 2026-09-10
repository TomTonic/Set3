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
type HashFunction func(unsafe.Pointer, uint64) uint64

// RuntimeHasher holds a per-type runtime hash function and a seed.
// It is intended to be created by [MakeRuntimeHasher] and called by the
// generic Set implementation to compute element hashes efficiently.
type RuntimeHasher[K comparable] struct {
	Seed uint64
	fn   HashFunction
}

// Hash computes the hash for key k using the stored runtime function
// and seed. The key pointer is wrapped with [Noescape] to avoid heap
// allocation during hashing.
//
//go:inline
func (h RuntimeHasher[K]) Hash(k K) uint64 {
	p := Noescape(unsafe.Pointer(&k)) //nolint:gosec
	return h.fn(p, h.Seed)
}

// MakeRuntimeHasher chooses an efficient per-type hash function for the
// generic type parameter K. It first matches common concrete types in a
// type switch (fast path) and falls back to reflect-based inspection for
// named slices/arrays. The returned RuntimeHasher contains the provided
// seed and the selected hash function.
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
// see internal/abi/escape.go
//
//go:nosplit
//go:nocheckptr
func Noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0) //nolint:gosec
}
