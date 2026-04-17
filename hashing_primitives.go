package set3

import (
	"hash/maphash"
	"math"
	"math/bits"
	"unsafe"
)

const goldenRatio64 = 0x9E3779B97F4A7C15 // (sqrt(5)-1)/2 * 2^64
const goldenRatio56 = 0x009E3779B97F4A7D // (sqrt(5)-1)/2 * 2^56
const goldenRatio48 = 0x00009E3779B97F4B // (sqrt(5)-1)/2 * 2^48
const goldenRatio40 = 0x0000009E3779B97F // (sqrt(5)-1)/2 * 2^40
const goldenRatio32 = 0x000000009E3779B9 // (sqrt(5)-1)/2 * 2^32
const goldenRatio24 = 0x00000000009E3779 // (sqrt(5)-1)/2 * 2^24
const goldenRatio16 = 0x0000000000009E37 // (sqrt(5)-1)/2 * 2^16
const goldenRatio08 = 0x000000000000009E // (sqrt(5)-1)/2 * 2^8

const sqrt2_1_64 = 0x6A09E667F3BCC909 // (sqrt(2)-1) * 2^64
const sqrt2_1_56 = 0x006A09E667F3BCC9 // (sqrt(2)-1) * 2^56
const sqrt2_1_48 = 0x00006A09E667F3BD // (sqrt(2)-1) * 2^48
const sqrt2_1_40 = 0x0000006A09E667F3 // (sqrt(2)-1) * 2^40
const sqrt2_1_32 = 0x000000006A09E667 // (sqrt(2)-1) * 2^32
const sqrt2_1_24 = 0x00000000006A09E7 // (sqrt(2)-1) * 2^24
const sqrt2_1_16 = 0x0000000000006A09 // (sqrt(2)-1) * 2^16
const sqrt2_1_08 = 0x000000000000006B // (sqrt(2)-1) * 2^8

const pie7_64 = 0xD64DD1B3DDCB7509 // (pi+e)/7 * 2^64
const pie7_56 = pie7_64>>8 | 1     // (pi+e)/7 * 2^56 & make sure the number is odd
const pie7_48 = pie7_64>>16 | 1    // (pi+e)/7 * 2^48 & make sure the number is odd
const pie7_40 = pie7_64>>24 | 1    // (pi+e)/7 * 2^40 & make sure the number is odd
const pie7_32 = pie7_64>>32 | 1    // (pi+e)/7 * 2^32 & make sure the number is odd
const pie7_24 = pie7_64>>40 | 1    // (pi+e)/7 * 2^24 & make sure the number is odd
const pie7_16 = pie7_64>>48 | 1    // (pi+e)/7 * 2^16 & make sure the number is odd
const pie7_08 = pie7_64>>56 | 1    // (pi+e)/7 * 2^8 & make sure the number is odd

const spread16to64 = pie7_48       // tests show that a multiplication with pie7_48 yields in the best distribution of 16-bit hashvalues to groups when using SplitMix64. See TestHashingCompare16BitConstantsForSplitMixGroupCountBuckets
const spread32to64 = goldenRatio32 // tests show that a multiplication with goldenRatio32 yields in the best distribution of 32-bit hashvalues to groups when using SplitMix64. See TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets

// hashBool always returns one of two fixed hash values for false and true.
// This is not a good general hash function, but it is well suited the
// specific use case of hashing boolean values in a hash set.
func hashBool(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*bool)(p)
	if b {
		return 0x1111_1111_1111_1111
	}
	return 0x0000_0000_0000_0000
}

// hashI08SM hashes a single uint8 by xoring it into the seed and
// applying splitmix64.
func hashI08SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	//v := 0x0101010101010101 * uint64(u) // broadcast to all 8 bytes
	//v := 0x0101010075310589 * uint64(u) // better dispersion for uint8 values: multiply with the largest prime p such that p*255 < 2^64
	v := goldenRatio56 * uint64(u) // better dispersion for uint8 values: multiply byte value with Knuth's golden ratio constant for 56 bits
	return splitmix64(seed ^ v)
}

// hashI08WH hashes a single uint8 by applying wh8.
func hashI08WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	return wh8(u, seed)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashI08MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	var mh maphash.Hash
	mh.SetSeed(seedToMaphashSeed(seed))
	_ = mh.WriteByte(u)
	return mh.Sum64()
}

// hashI16SM hashes a uint16 value by promoting to uint64 and applying splitmix64.
func hashI16SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	//v := 0x0001000100010001 * uint64(u) // broadcast to all 8 bytes
	//v := 0x000100010000FFD1 * uint64(u) // better dispersion for small uint16 values: multiply with the largest prime p such that p*65535 < 2^64
	v := spread16to64 * uint64(u) // better dispersion for uint16 values: multiply with Knuth's golden ratio constant for 48 bits
	return splitmix64(seed ^ v)
}

// hashI16WH hashes a uint16 value by applying wh16.
func hashI16WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	return wh16(u, seed)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashI16MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	b := (*[2]byte)(unsafe.Pointer(&u))[:] // view on the 2 Bytes of u
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// hashI32SM hashes a uint32 by promoting to uint64 and mixing.
func hashI32SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	//v := 0x0000000100000001 * uint64(u) // broadcast to all 8 bytes
	//v := 0x00000000FFFFFFFB * uint64(u) // better dispersion for small uint32 values: multiply with the largest prime p such that p*4294967295 < 2^64
	v := spread32to64 * uint64(u) // better dispersion for uint32 values
	return splitmix64(seed ^ v)
}

// hashI32WH hashes a uint32 using the wh32 function.
func hashI32WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	return wh32(u, seed)
}

// hashI32WHdet hashes a uint32 using the deterministic variant of wh32.
func hashI32WHdet(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	return wh32det(u, seed)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashI32MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	b := (*[4]byte)(unsafe.Pointer(&u))[:] // view on the 4 Bytes of u
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// hashI64SM hashes a uint64 by mixing it with the seed using splitmix64.
func hashI64SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return splitmix64(seed ^ u)
}

// hashI64WH hashes a uint64 by using wh64.
func hashI64WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return wh64(u, seed)
}

// hashI64WHdet hashes a uint64 by using the deterministic variant of wh64.
func hashI64WHdet(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return wh64det(u, seed)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashI64MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	b := (*[8]byte)(unsafe.Pointer(&u))[:] // view on the 8 Bytes of u
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// hashInt handles the platform-sized unsigned integer type.
func hashInt(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashPtr hashes a uintptr value by casting it to uint64 and mixing.
func hashPtr(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uintptr)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashF32SM hashes a float32 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func hashF32SM(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float32)(p)
	var bits uint32
	if f == 0 {
		bits = 0
	} else if math.IsNaN(float64(f)) {
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7fc00000 // canonical representation of NaN for float32
	} else {
		bits = math.Float32bits(f)
	}
	//v := 0x0000000100000001 * uint64(u) // broadcast to all 8 bytes
	//v := 0x00000000FFFFFFFB * uint64(bits) // better dispersion for small uint32 values: multiply with the largest prime p such that p*4294967295 < 2^64
	v := spread32to64 * uint64(bits) // better dispersion for 32-bit values
	return splitmix64(seed ^ v)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashF32MH(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float32)(p)
	var bits uint32
	if f == 0 {
		bits = 0
	} else if math.IsNaN(float64(f)) {
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7fc00000 // canonical representation of NaN for float32
	} else {
		bits = math.Float32bits(f)
	}
	b := (*[4]byte)(unsafe.Pointer(&bits))[:] // view on the 4 Bytes of bits
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// hashF64SM hashes a float64 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func hashF64SM(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float64)(p)
	var bits uint64
	if f == 0 {
		bits = 0
	} else if math.IsNaN(f) {
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7ff8000000000000 // canonical representation of NaN for float64
	} else {
		bits = math.Float64bits(f)
	}
	return splitmix64(seed ^ bits)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// float64.
func hashF64MH(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float64)(p)
	var bits uint64
	if f == 0 {
		bits = 0
	} else if math.IsNaN(f) {
		// even though we can never "find" NaN in a set (as NaN != NaN),
		// we still want all NaN values to hash to the same value
		// to avoid unnecessary collisions with other values
		bits = 0x7ff8000000000000 // canonical representation of NaN for float64
	} else {
		bits = math.Float64bits(f)
	}
	b := (*[8]byte)(unsafe.Pointer(&bits))[:] // view on the 8 Bytes of bits
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// splitmix64 is a fast 64-bit mixing function used to
// scramble input words. It is not cryptographic but provides good
// dispersion for hash table use.
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// swirlByte mixes a single byte value with the seed by broadcasting.
// not a real hash function, but it is salted and makes sure to distribute
// bits well over the 64-bit output. A special focus is put on the
// lower bits to ensure a) that lower 7 bits are well distributed and
// b) that the most significant bit influences the lower bits to
// increase the chance that adjacent byte values are hashed to
// different buckets.
func swirlByte(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*byte)(p)
	u := 0x0101010101010101 * uint64(b) // broadcast byte value to all 8 bytes
	v := bits.RotateLeft64(u, -3)       // rotate right by 3 - increase probabability that adjacent values are hashed to different buckets
	w := v ^ seed                       // mix in seed
	return w
}

func xorshift64star08(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*byte)(p)
	//u := 0x0100100100101001 * uint64(b) // irregularely scatter byte value over all 8 state bytes
	//u := 0x0101010075310589 * uint64(b) // better dispersion for small uint8 values: multiply with the largest prime p such that p*255 < 2^64
	u := goldenRatio56 * uint64(b) // better dispersion for uint8 values: multiply byte value with Knuth's golden ratio constant for 56 bits
	x := u ^ seed
	// this is the xorshift64star algorithm, see https://en.wikipedia.org/wiki/Xorshift#xorshift*
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	//y := x * 0x2545F4914F6CDD1D
	y := x * pie7_64              // use different multiplier to avoid correlation with other xorshift64star variants
	z := bits.RotateLeft64(y, 32) // swap upper and lower 32 bits, as xorshift64* is weak in the lower bits
	return z
}

func xorshift64star16(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*uint16)(p)
	//u := 0x0001_0000_0010_0001 * uint64(b) // irregularely scatter 16 bit value over all 8 state bytes
	//u := 0x000100010000FFD1 * uint64(b) // better dispersion for small uint16 values: multiply with the largest prime p such that p*65535 < 2^64
	u := goldenRatio48 * uint64(b) // better dispersion for uint16 values: multiply with Knuth's golden ratio constant for 48 bits
	x := u ^ seed
	// this is the xorshift64star algorithm, see https://en.wikipedia.org/wiki/Xorshift#xorshift*
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	//y := x * 0x2545F4914F6CDD1D
	y := x * pie7_64              // use different multiplier to avoid correlation with other xorshift64star variants
	z := bits.RotateLeft64(y, 32) // swap upper and lower 32 bits, as xorshift64* is weak in the lower bits
	return z
}
