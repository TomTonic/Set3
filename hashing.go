package set3

import (
	"encoding/binary"
	"hash/maphash"
	"math"
	"reflect"
	"unsafe"
)

// hashfunction is the runtime function used to hash values. It receives a
// pointer to the value and a uint64 seed; it returns a uint64 hash.
// hashfunction is the runtime function used to hash values. It receives a
// pointer to the value and a uint64 seed; it returns a uint64 hash.
// Implementations must treat the memory at the pointer as the concrete
// representation of the value and incorporate the seed to allow
// deterministic re-seeding.
type hashfunction func(unsafe.Pointer, uint64) uint64

// maphashhashfunction is the type of the internal maphash hasher function. It receives
// a pointer to the value and a uintptr seed; it returns a uintptr hash.
// maphashhashfunction is the signature used by low-level maphash-based
// hashers. It takes a pointer to a value and a uintptr seed and returns
// a uintptr-sized hash. This mirrors the internal form used by the
// stdlib/third-party implementations we interoperate with.
type maphashhashfunction func(unsafe.Pointer, uintptr) uintptr

// splitmix64 is a fast, high-quality 64-bit mixing function used to
// scramble input words. It is not cryptographic but provides good
// dispersion for hash table use.
// see https://en.wikipedia.org/wiki/SplitMix64
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// hashBytesBlock hashes a byte slice using 64-bit block mixing.
// It consumes 8-byte words with LittleEndian decoding and mixes each
// block through splitmix64; the tail (0..7 bytes) is folded in and the
// length is incorporated to avoid collisions for different-length inputs.
func hashBytesBlock(seed uint64, b []byte) uint64 {
	h := seed ^ 0x9E3779B97F4A7C15
	i, n := 0, len(b)
	for i+8 <= n {
		v := binary.LittleEndian.Uint64(b[i:])
		h = splitmix64(h ^ v)
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
	}
	return splitmix64(h ^ tail ^ uint64(n))
}

// RuntimeHasher holds a per-type runtime hash function and a seed.
// It is intended to be created by MakeRuntimeHasher and called by the
// generic Set implementation to compute element hashes efficiently.
type RuntimeHasher[K comparable] struct {
	Seed uint64
	fn   hashfunction
}

// Hash computes the hash for key k using the stored runtime function
// and seed. The key pointer is wrapped with noescape to avoid heap
// allocation during hashing.
func (h *RuntimeHasher[K]) Hash(k K) uint64 {
	p := noescape(unsafe.Pointer(&k))
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
		h.fn = hashUint8
	case int8:
		h.fn = hashInt8
	case bool:
		h.fn = hashBool
	case uint16:
		h.fn = hashUint16
	case int16:
		h.fn = hashInt16
	case uint32:
		h.fn = hashUint32
	case int32:
		h.fn = hashInt32
	case uint64:
		h.fn = hashUint64
	case int64:
		h.fn = hashInt64
	case uint:
		h.fn = hashUint
	case int:
		h.fn = hashInt
	case uintptr:
		h.fn = hashUintptr
	case float32:
		h.fn = hashFloat32
	case float64:
		h.fn = hashFloat64
	case string:
		h.fn = hashString
	case []byte:
		// []byte and []uint8 are identical types; both use slice handler
		h.fn = hashUint8Slice
	case []int8:
		h.fn = hashUint8Slice
	default:
		// fall back to reflect-based inspection for more cases
		t := reflect.TypeOf(zero)
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			// subtle difference: []byte (but with different declared element type) -> use slice handler
			h.fn = hashUint8Slice
		} else if t.Kind() == reflect.Array && t.Elem().Kind() == reflect.Uint8 {
			// [N]byte -> treat as raw bytes
			h.fn = hashUint8Array[K]
		} else {
			// generic approach: use internal hash function from SwissMapType
			// api package internal/abi not allowed
			//var m map[K]struct{}
			//mTyp := abi.TypeOf(m)
			//maphashhasher := (*abi.SwissMapType)(unsafe.Pointer(mTyp)).Hasher
			//h.fn = maphashhashfunctionWrapper(maphashhasher)
			h.fn = hashFallbackMaphash[K]
		}
	}
	return h
}

// hashBool hashes a boolean value by mapping false->0 and true->1 and
// mixing with the seed via splitmix64.
func hashBool(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*bool)(p)
	var v uint64
	if b {
		v = 1
	}
	return splitmix64(seed ^ v)
}

// hashUint8 hashes a single uint8 by xoring it into the seed and
// applying splitmix64.
func hashUint8(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint8)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashUint8Slice hashes a []uint8 (alias []byte) by delegating to the
// byte-block hashing routine. This avoids per-element overhead.
func hashUint8Slice(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	return hashBytesBlock(seed, b)
}

// hashUint8Array handles fixed-size arrays like [N]byte by treating the
// array memory as a []byte and reusing the byte-block hash implementation.
// hashUint8Array handles fixed-size arrays (e.g. [N]byte) by viewing the
// array memory as a []byte slice and reusing the byte-block hasher. It
// works for N==0 as unsafe.Slice with length 0 is valid.
func hashUint8Array[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// The size in bytes of the array type K equals the number of uint8 elements,
	// since element size is 1. Use unsafe.Sizeof on the dereferenced value to get it.
	size := int(unsafe.Sizeof(*(*K)(p)))
	b := unsafe.Slice((*byte)(p), size)
	return hashBytesBlock(seed, b)
}

// hashInt8 hashes a single int8 by reinterpreting its bits as uint8 and
// mixing with the seed.
func hashInt8(p unsafe.Pointer, seed uint64) uint64 {
	v := *(*int8)(p)
	return splitmix64(seed ^ uint64(uint8(v)))
}

// hashUint16 hashes a uint16 value by promoting to uint64 and applying splitmix64.
func hashUint16(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint16)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashInt16 hashes an int16 value by reinterpreting its bits and mixing.
func hashInt16(p unsafe.Pointer, seed uint64) uint64 {
	v := *(*int16)(p)
	return splitmix64(seed ^ uint64(uint16(v)))
}

// hashUint32 hashes a uint32 by promoting to uint64 and mixing.
func hashUint32(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint32)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashInt32 hashes an int32 by reinterpreting bits as uint32 and mixing.
func hashInt32(p unsafe.Pointer, seed uint64) uint64 {
	v := *(*int32)(p)
	return splitmix64(seed ^ uint64(uint32(v)))
}

// hashUint64 hashes a uint64 by mixing it with the seed using splitmix64.
func hashUint64(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint64)(p)
	return splitmix64(seed ^ u)
}

// hashInt64 hashes an int64 by reinterpreting its bit pattern and mixing.
func hashInt64(p unsafe.Pointer, seed uint64) uint64 {
	v := *(*int64)(p)
	return splitmix64(seed ^ uint64(v))
}

// hashUint handles the platform-sized unsigned integer type.
func hashUint(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uint)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashInt handles the platform-sized signed integer type.
func hashInt(p unsafe.Pointer, seed uint64) uint64 {
	v := *(*int)(p)
	return splitmix64(seed ^ uint64(v))
}

// hashUintptr hashes a uintptr value by casting it to uint64 and mixing.
func hashUintptr(p unsafe.Pointer, seed uint64) uint64 {
	u := *(*uintptr)(p)
	return splitmix64(seed ^ uint64(u))
}

// hashFloat32 hashes a float32 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func hashFloat32(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float32)(p)
	var bits uint64
	if f == 0 {
		bits = 0
	} else if math.IsNaN(float64(f)) {
		bits = 0x7fc00000 // canonical quiet NaN for float32
	} else {
		bits = uint64(math.Float32bits(f))
	}
	return splitmix64(seed ^ bits)
}

// hashFloat64 hashes a float64 by canonicalizing zero and NaN and then
// hashing the IEEE-754 bit representation.
func hashFloat64(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float64)(p)
	var bits uint64
	if f == 0 {
		bits = 0
	} else if math.IsNaN(f) {
		bits = 0x7ff8000000000000 // canonical quiet NaN for float64
	} else {
		bits = math.Float64bits(f)
	}
	return splitmix64(seed ^ bits)
}

// hashString hashes a Go string by obtaining a byte view of the string
// data (without allocations) and delegating to the byte-block hasher.
func hashString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	// Go 1.20+: unsafe.StringData(s) → *byte
	b := unsafe.Slice(unsafe.StringData(s), len(s))
	return hashBytesBlock(seed, b)
}

// maphashhashfunctionWrapper adapts a maphash-style function (taking a
// uintptr seed and returning a uintptr) into our internal hashfunction
// type which uses uint64 seeds and results.
func maphashhashfunctionWrapper(fn maphashhashfunction) hashfunction {
	return func(p unsafe.Pointer, seed uint64) uint64 {
		return uint64(fn(p, uintptr(seed)))
	}
}

// hashFallbackMaphash is the generic fallback hasher which uses
// stdlib `hash/maphash` to hash arbitrary comparable types by calling
// `maphash.WriteComparable`. This is slower than the specialized
// routines but works for any K.
func hashFallbackMaphash[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	k := *(*K)(p)
	var mh maphash.Hash
	mh.SetSeed(seedToMaphashSeed(seed))
	maphash.WriteComparable(&mh, k)
	return mh.Sum64()
}

// seedToMaphashSeed derives a deterministic maphash.Seed from a
// uint64 seed. This performs an unsafe copy of the 8 bytes into the
// maphash.Seed value; that relies on the concrete layout of Seed and
// is pragmatic but not guaranteed by the language spec.
func seedToMaphashSeed(seed uint64) maphash.Seed {
	if seed == 0 {
		return maphash.MakeSeed()
	}
	var sd maphash.Seed
	p := unsafe.Pointer(&sd)
	buf := (*[8]byte)(p)
	binary.LittleEndian.PutUint64(buf[0:8], seed)
	return sd
}

// noescape hides the pointer p from escape analysis, preventing it
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
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}
