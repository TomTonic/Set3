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
		tail |= uint64(b[i+6]) << 48
		fallthrough
	case 6:
		tail |= uint64(b[i+5]) << 40
		fallthrough
	case 5:
		tail |= uint64(b[i+4]) << 32
		fallthrough
	case 4:
		tail |= uint64(b[i+3]) << 24
		fallthrough
	case 3:
		tail |= uint64(b[i+2]) << 16
		fallthrough
	case 2:
		tail |= uint64(b[i+1]) << 8
		fallthrough
	case 1:
		tail |= uint64(b[i])
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

// HashAsByteArray handles fixed-size arrays (e.g. [N]byte) by viewing the
// array memory as a []byte slice and reusing the byte-block hasher. It
// works for N==0 as unsafe.Slice with length 0 is valid.
func HashAsByteArray[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// The size in bytes of the array type K equals the number of uint8 elements,
	// since element size is 1. Use unsafe.Sizeof on the dereferenced value to get it.
	size := int(unsafe.Sizeof(*(*K)(p)))
	b := unsafe.Slice((*byte)(p), size)
	return HashBytesBlock(seed, b)
}

// HashString hashes a Go string by obtaining a byte view of the string
// data (without allocations) and delegating to the byte-block hasher.
func HashString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	// Go 1.20+: unsafe.StringData(s) → *byte
	b := unsafe.Slice(unsafe.StringData(s), len(s))
	return HashBytesBlock(seed, b)
}

// HashFallbackMaphash is the generic fallback hasher which uses
// stdlib `hash/maphash` to hash arbitrary comparable types by calling
// `maphash.WriteComparable`. This is slower than the specialized
// routines but works for any K.
func HashFallbackMaphash[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	k := *(*K)(p)
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
	var sd maphash.Seed
	p := unsafe.Pointer(&sd)
	buf := (*[8]byte)(p)
	*(*uint64)(unsafe.Pointer(&buf[0])) = seed // for a hashset the byte order does not matter
	return sd
}
