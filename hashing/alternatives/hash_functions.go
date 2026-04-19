package alternatives

import (
	"hash/maphash"
	"math"
	"math/bits"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

// --- uint8 alternatives ---

// HashI08SM hashes a single uint8 by xoring it into the seed and
// applying splitmix64.
func HashI08SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	//v := 0x0101010101010101 * uint64(u) // broadcast to all 8 bytes
	//v := 0x0101010075310589 * uint64(u) // better dispersion for uint8 values: multiply with the largest prime p such that p*255 < 2^64
	v := hashing.GoldenRatio56 * uint64(u) // better dispersion for uint8 values: multiply byte value with Knuth's golden ratio constant for 56 bits
	return hashing.Splitmix64(seed ^ v)
}

// HashI08WH hashes a single uint8 by applying wh8.
func HashI08WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	return WH8(u, seed)
}

// HashI08MH uses the hasher from stdlib hash/maphash to hash uint8.
func HashI08MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	var mh maphash.Hash
	mh.SetSeed(hashing.SeedToMaphashSeed(seed))
	_ = mh.WriteByte(u)
	return mh.Sum64()
}

// --- uint16 alternatives ---

// HashI16WH hashes a uint16 value by applying wh16.
func HashI16WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	return WH16(u, seed)
}

// HashI16MH uses the hasher from stdlib hash/maphash to hash uint16.
func HashI16MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	b := (*[2]byte)(unsafe.Pointer(&u))[:] // view on the 2 Bytes of u
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- uint32 alternatives ---

// HashI32SM hashes a uint32 by promoting to uint64 and mixing.
func HashI32SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	//v := 0x0000000100000001 * uint64(u) // broadcast to all 8 bytes
	//v := 0x00000000FFFFFFFB * uint64(u) // better dispersion for small uint32 values: multiply with the largest prime p such that p*4294967295 < 2^64
	v := hashing.Spread32to64 * uint64(u) // better dispersion for uint32 values
	return hashing.Splitmix64(seed ^ v)
}

// HashI32WH hashes a uint32 using the non-deterministic wh32 function.
func HashI32WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	return WH32(u, seed)
}

// HashI32MH uses the hasher from stdlib hash/maphash to hash uint32.
func HashI32MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	b := (*[4]byte)(unsafe.Pointer(&u))[:] // view on the 4 Bytes of u
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- uint64 alternatives ---

// HashI64SM hashes a uint64 by mixing it with the seed using splitmix64.
func HashI64SM(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return hashing.Splitmix64(seed ^ u)
}

// HashI64WH hashes a uint64 by using the non-deterministic wh64.
func HashI64WH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return WH64(u, seed)
}

// HashI64MH uses the hasher from stdlib hash/maphash to hash uint64.
func HashI64MH(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	b := (*[8]byte)(unsafe.Pointer(&u))[:] // view on the 8 Bytes of u
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- float32 alternatives ---

// HashF32WHdet hashes a float32 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation using the deterministic wh32.
func HashF32WHdet(p unsafe.Pointer, seed uint64) uint64 {
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
	v := hashing.WH32DetGR(bits, seed)
	return v
}

// HashF32MH uses the hasher from stdlib hash/maphash to hash float32.
func HashF32MH(p unsafe.Pointer, seed uint64) uint64 {
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
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- float64 alternatives ---

// HashF64SM hashes a float64 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation using splitmix64.
func HashF64SM(p unsafe.Pointer, seed uint64) uint64 {
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
	return hashing.Splitmix64(seed ^ bits)
}

// HashF64MH uses the hasher from stdlib hash/maphash to hash float64.
func HashF64MH(p unsafe.Pointer, seed uint64) uint64 {
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
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- string alternatives ---

// HashStringMH uses the hasher from stdlib hash/maphash to hash Go strings.
func HashStringMH(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.String(mhs, s)
	return h
}

// HashStringWH hashes a Go string using the non-deterministic whX byte-block hasher.
func HashStringWH(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	// Go 1.20+: unsafe.StringData(s) → *byte
	b := unsafe.Slice(unsafe.StringData(s), len(s))
	return WhX(b, seed)
}

// --- byte slice alternatives ---

// HashByteSliceWH hashes a []uint8 (alias []byte) using the non-deterministic
// whX byte-block hasher.
func HashByteSliceWH(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	return WhX(b, seed)
}

// HashBytesMH uses the hasher from stdlib hash/maphash to hash byte slices.
func HashBytesMH(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	mhs := hashing.SeedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// --- xorshift variants for byte/uint16 hashing ---

// XorShift64Star08 hashes a single uint8 using the xorshift64* algorithm.
func XorShift64Star08(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*byte)(p)
	//u := 0x0100100100101001 * uint64(b) // irregularely scatter byte value over all 8 state bytes
	//u := 0x0101010075310589 * uint64(b) // better dispersion for small uint8 values: multiply with the largest prime p such that p*255 < 2^64
	u := hashing.GoldenRatio56 * uint64(b) // better dispersion for uint8 values: multiply byte value with Knuth's golden ratio constant for 56 bits
	x := u ^ seed
	// this is the xorshift64star algorithm, see https://en.wikipedia.org/wiki/Xorshift#xorshift*
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	//y := x * 0x2545F4914F6CDD1D
	y := x * hashing.Pie7_64      // use different multiplier to avoid correlation with other xorshift64star variants
	z := bits.RotateLeft64(y, 32) // swap upper and lower 32 bits, as xorshift64* is weak in the lower bits
	return z
}

// XorShift64Star16 hashes a uint16 using the xorshift64* algorithm.
func XorShift64Star16(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*uint16)(p)
	//u := 0x0001_0000_0010_0001 * uint64(b) // irregularely scatter 16 bit value over all 8 state bytes
	//u := 0x000100010000FFD1 * uint64(b) // better dispersion for small uint16 values: multiply with the largest prime p such that p*65535 < 2^64
	u := hashing.GoldenRatio48 * uint64(b) // better dispersion for uint16 values: multiply with Knuth's golden ratio constant for 48 bits
	x := u ^ seed
	// this is the xorshift64star algorithm, see https://en.wikipedia.org/wiki/Xorshift#xorshift*
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	//y := x * 0x2545F4914F6CDD1D
	y := x * hashing.Pie7_64      // use different multiplier to avoid correlation with other xorshift64star variants
	z := bits.RotateLeft64(y, 32) // swap upper and lower 32 bits, as xorshift64* is weak in the lower bits
	return z
}

// --- slice-as-byte-array utilities ---

// AnySliceAsByteSlice reinterprets an arbitrary slice of comparable elements as
// a []byte by viewing its backing array as raw memory. The length is
// len(slice) * sizeof(element). Returns nil for empty slices.
func AnySliceAsByteSlice[T comparable](p unsafe.Pointer) []byte {
	anySlice := *(*[]T)(p)
	if len(anySlice) == 0 {
		return nil
	}
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	l := len(anySlice) * elemSize
	result := unsafe.Slice((*byte)(unsafe.Pointer(&anySlice[0])), l)
	return result
}

// HashAnySliceAsByteSlice hashes a slice of any comparable type by
// reinterpreting it as []byte and using the byte-block hasher.
func HashAnySliceAsByteSlice[T comparable](p unsafe.Pointer, seed uint64) uint64 {
	b := AnySliceAsByteSlice[T](p)
	return hashing.HashBytesBlock(seed, b)
}
