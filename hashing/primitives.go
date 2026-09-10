package hashing

import (
	"math"
	"math/bits"
	"unsafe"
)

// Splitmix64 is a fast 64-bit mixing function used to
// scramble input words. It is not cryptographic but provides good
// dispersion for hash table use.
func Splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// SwirlByte mixes a single byte value with the seed by broadcasting.
// not a real hash function, but it is salted and makes sure to distribute
// bits well over the 64-bit output. A special focus is put on the
// lower bits to ensure a) that lower 7 bits are well distributed and
// b) that the most significant bit influences the lower bits to
// increase the chance that adjacent byte values are hashed to
// different buckets.
func SwirlByte(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*byte)(p)
	u := 0x0101010101010101 * uint64(b) // broadcast byte value to all 8 bytes
	v := bits.RotateLeft64(u, -3)       // rotate right by 3 - increase probabability that adjacent values are hashed to different buckets
	w := v ^ seed                       // mix in seed
	return w
}

// HashBool always returns one of two fixed hash values for false and true.
// This is not a good general hash function, but it is well suited the
// specific use case of hashing boolean values in a hash set.
func HashBool(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*bool)(p)
	if b {
		return 0x1111_1111_1111_1111
	}
	return 0x0000_0000_0000_0000
}

// HashI16SM hashes a uint16 value by promoting to uint64 and applying splitmix64.
func HashI16SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	v := 0x0001000100010001 * uint64(u) // broadcast to all 8 bytes
	return Splitmix64(seed ^ v)
}

// HashI32WHdet hashes a uint32 using the deterministic variant of wh32.
func HashI32WHdet(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	return WH32DetGR(u, seed)
}

// HashI64WHdet hashes a uint64 by using the deterministic variant of wh64.
func HashI64WHdet(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return WH64Det(u, seed)
}

// HashInt handles the platform-sized unsigned integer type.
func HashInt(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint)(p)
	return Splitmix64(seed ^ uint64(u))
}

// HashPtr hashes a uintptr value by casting it to uint64 and mixing.
func HashPtr(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uintptr)(p)
	return Splitmix64(seed ^ uint64(u))
}

// HashF32SM hashes a float32 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func HashF32SM(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float32)(p)
	var bits uint32
	switch {
	case f == 0:
		bits = 0
	case math.IsNaN(float64(f)):
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7fc00000 // canonical representation of NaN for float32
	default:
		bits = math.Float32bits(f)
	}
	v := 0x0000000100000001 * uint64(bits) // broadcast to all 8 bytes, replication performed best in our benchmarks for the full float32 range
	return Splitmix64(seed ^ v)
}

// HashF64WHdet hashes a float64 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func HashF64WHdet(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float64)(p)
	var bits uint64
	switch {
	case f == 0:
		bits = 0
	case math.IsNaN(f):
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7ff8000000000000 // canonical representation of NaN for float64
	default:
		bits = math.Float64bits(f)
	}
	return WH64Det(bits, seed)
}
