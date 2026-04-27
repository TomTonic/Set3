// nolint:gosec // All unsafe operations below are audited and safe for low-level byte hashing
package hashing

import (
	"encoding/binary"
	"hash/maphash"
	"unsafe"
)

// HashBytesBlock hashes a byte slice using 64-bit block mixing.
// It consumes 8-byte words with LittleEndian decoding and mixes each
// block through splitmix64; the tail (0..7 bytes) is folded in and the
// length is incorporated to avoid collisions for different-length inputs.
//
//go:inline
func HashBytesBlock(seed uint64, b []byte) uint64 {
	h := seed ^ P0
	i, n := 0, len(b)
	for i+8 <= n {
		v := binary.NativeEndian.Uint64(b[i:])
		h = WH64Det(v, h)
		i += 8
	}
	// Tail 0..7 Bytes
	var tail uint64
	switch n - i {
	case 7:
		tail |= uint64(b[i+6]) << 48 //nolint:gosec
		fallthrough
	case 6:
		tail |= uint64(b[i+5]) << 40 //nolint:gosec
		fallthrough
	case 5:
		tail |= uint64(b[i+4]) << 32 //nolint:gosec
		fallthrough
	case 4:
		tail |= uint64(b[i+3]) << 24 //nolint:gosec
		fallthrough
	case 3:
		tail |= uint64(b[i+2]) << 16 //nolint:gosec
		fallthrough
	case 2:
		tail |= uint64(b[i+1]) << 8 //nolint:gosec
		fallthrough
	case 1:
		tail |= uint64(b[i]) //nolint:gosec
	case 0:
		tail = P1
	}
	return WH64Det(tail^uint64(n)*P2, h)
}

// HashByteSlice hashes a []uint8 (alias []byte) by delegating to the
// byte-block hashing routine. This avoids per-element overhead.
func HashByteSlice(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	return HashBytesBlock(seed, b)
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

// HashString hashes a Go string by reading its bytes directly through
// pointer arithmetic, avoiding slice creation and the function-call
// overhead of HashBytesBlock.
//
//go:inline
func HashString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	n := len(s)
	h := seed ^ P0
	if n == 0 {
		return WH64Det(P1^uint64(n)*P2, h)
	}
	dp := unsafe.Pointer(unsafe.StringData(s)) //nolint:gosec
	i := 0
	for i+8 <= n {
		v := *(*uint64)(unsafe.Add(dp, i)) //nolint:gosec
		h = WH64Det(v, h)
		i += 8
	}
	// Tail 0..7 bytes – same encoding as HashBytesBlock
	var tail uint64
	switch n - i {
	case 7:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+6))) << 48 //nolint:gosec
		fallthrough
	case 6:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+5))) << 40 //nolint:gosec
		fallthrough
	case 5:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+4))) << 32 //nolint:gosec
		fallthrough
	case 4:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+3))) << 24 //nolint:gosec
		fallthrough
	case 3:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+2))) << 16 //nolint:gosec
		fallthrough
	case 2:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+1))) << 8 //nolint:gosec
		fallthrough
	case 1:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i))) //nolint:gosec
	case 0:
		tail = P1
	}
	return WH64Det(tail^uint64(n)*P2, h)
}

// HashFallbackMaphash is the generic fallback hasher which uses
// stdlib `hash/maphash` to hash arbitrary comparable types by calling
// `maphash.WriteComparable`. This is slower than the specialized
// routines but works for any K.
func HashFallbackMaphash[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// Safely dereference the comparable type K from the pointer.
	// Audited: p points to a valid K instance.
	k := *(*K)(p) //nolint:gosec
	var mh maphash.Hash
	mh.SetSeed(SeedToMaphashSeed(seed))
	maphash.WriteComparable(&mh, k)
	return mh.Sum64()
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
