package hashing

import (
	"hash/maphash"
	"unsafe"
)

// HashBytesBlock hashes a byte slice.
//
// It is the variable-length entry point into the lane-parallel body in
// lanehash.go: short inputs are straight-line code, and only inputs past 32
// bytes enter a loop. See that file for why the shape is what it is.
//
// The hash depends on the platform's endianness — it is a hash table's hash,
// not a wire format — and it is deterministic within a process family: the same
// bytes and the same seed always give the same value.
func HashBytesBlock(seed uint64, b []byte) uint64 {
	if len(b) == 0 {
		return hashLaneBytes(nil, 0, seed)
	}
	return hashLaneBytes(unsafe.Pointer(&b[0]), len(b), seed) //nolint:gosec
}

// HashAsByteArray handles fixed-size raw-byte-eligible values (for example
// [N]byte or structs with byte-stable equality semantics) by viewing their
// memory as a []byte slice and hashing it directly. Hot fixed sizes are
// dispatched to straight-line helpers; all other sizes fall back to the
// generic byte-block hasher. It works for N==0 as unsafe.Slice with length 0
// is valid.
func HashAsByteArray[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// Safely view the array memory as a byte slice using unsafe.Slice.
	// The size in bytes of array type K equals the number of uint8 elements
	// since element size is 1. Audited: size calculation is safe.
	size := int(unsafe.Sizeof(*(*K)(p))) //nolint:gosec
	if specialized := fixedSizeByteBlockHasher(size); specialized != nil {
		return specialized(p, seed)
	}
	b := unsafe.Slice((*byte)(p), size) //nolint:gosec
	return HashBytesBlock(seed, b)
}

// HashString hashes a Go string by reading its bytes directly through pointer
// arithmetic, avoiding slice creation.
//
// It shares its body with HashBytesBlock, so a string and a byte slice holding
// the same bytes hash to the same value.
func HashString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	n := len(s)
	if n == 0 {
		return hashLaneBytes(nil, 0, seed)
	}
	return hashLaneBytes(unsafe.Pointer(unsafe.StringData(s)), n, seed) //nolint:gosec
}

// HashFallbackMaphash is the generic fallback hasher which uses stdlib
// `hash/maphash` to hash arbitrary comparable types by calling
// `maphash.Comparable`. This is slower than the specialized routines but
// works for any K, including interface types whose dynamic type is only
// known per value.
//
// It panics if K is an interface type holding a value that is not
// comparable, which mirrors the behaviour of == on such a value.
func HashFallbackMaphash[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// Safely dereference the comparable type K from the pointer.
	// Audited: p points to a valid K instance.
	k := *(*K)(p) //nolint:gosec
	return maphash.Comparable(SeedToMaphashSeed(seed), k)
}

// SeedToMaphashSeed derives a deterministic maphash.Seed from a
// uint64 seed. This performs an unsafe copy of the 8 bytes into the
// maphash.Seed value; that relies on the concrete layout of Seed and
// is pragmatic but not guaranteed by the language spec.
func SeedToMaphashSeed(seed uint64) maphash.Seed {
	// Derive maphash.Seed deterministically from the provided uint64 seed
	// by copying the 8 bytes into the Seed value. We avoid calling
	// maphash.MakeSeed() here so that a seed value of 0 remains
	// deterministic (useful for reproducible tests and deterministic
	// behavior across runs).
	// Ensure we never produce the all-zero Seed value, which the
	// stdlib treats as uninitialized. For a uint64 input of 0 we map it
	// to a fixed non-zero constant so behavior remains deterministic.
	if seed == 0 {
		seed = 0x9E3779B97F4A7C15
	}
	// Copy 8 bytes from the uint64 seed into maphash.Seed.
	// Audited: seed and Seed layout are well-understood; unsafe copy is safe.
	var sd maphash.Seed
	p := unsafe.Pointer(&sd)                   //nolint:gosec
	buf := (*[8]byte)(p)                       //nolint:gosec
	*(*uint64)(unsafe.Pointer(&buf[0])) = seed //nolint:gosec // for a hashset the byte order does not matter
	return sd
}
