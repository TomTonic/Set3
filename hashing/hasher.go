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
func (h *RuntimeHasher[K]) Hash(k K) uint64 {
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

	switch any(zero).(type) {
	case uint8:
		h.fn = SwirlByte
	case int8:
		h.fn = SwirlByte
	case bool:
		h.fn = HashBool
	case uint16:
		h.fn = HashI16SM
	case int16:
		h.fn = HashI16SM
	case uint32:
		h.fn = HashI32WHdet
	case int32:
		h.fn = HashI32WHdet
	case uint64:
		h.fn = HashI64WHdet
	case int64:
		h.fn = HashI64WHdet
	case uint:
		h.fn = HashInt
	case int:
		h.fn = HashInt
	case uintptr:
		h.fn = HashPtr
	case float32:
		h.fn = HashF32SM
	case float64:
		h.fn = HashF64WHdet
	case string:
		h.fn = HashString
	case []byte, []int8:
		// []byte and []uint8 are identical types; both use slice handler
		h.fn = HashByteSlice
	case []int, []uint, []int16, []uint16, []int32, []uint32,
		[]int64, []uint64:
		// h.fn = hashAnySliceAsByteSlice[K]
		panic("slices of non-byte int/uint types were not 'comparable' at the time of writing this code, so no tests were possible")
	default:
		// fall back to reflect-based inspection for more cases
		t := reflect.TypeOf(zero)
		switch {
		case t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8:
			// subtle difference: []byte (but with different declared element type) -> use slice handler
			h.fn = HashByteSlice
		case CanUseUnsafeRawByteBlockHasherType(t).Eligible:
			// Fast path for layouts that are safe for raw byte-block hashing
			// according to Go equality semantics.
			h.fn = HashAsByteArray[K]
		case GenerateHashFunction(t) != nil:
			// Reflection-based generator produced a fast, type-specific
			// hash closure (e.g. for structs with padding, floats, strings,
			// or complex fields). No reflection happens at hash time.
			h.fn = GenerateHashFunction(t)
		default:
			// generic approach: use internal hash function from SwissMapType
			h.fn = HashFallbackMaphash[K]
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
