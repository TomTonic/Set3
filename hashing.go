package set3

import (
	"encoding/binary"
	"hash/maphash"
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
		h.fn = swirlByte
	case int8:
		h.fn = swirlByte
	case bool:
		h.fn = hashBool
	case uint16:
		h.fn = hashI16SM
	case int16:
		h.fn = hashI16SM
	case uint32:
		h.fn = hashI32WHdet
	case int32:
		h.fn = hashI32WHdet
	case uint64:
		h.fn = hashI64WHdet
	case int64:
		h.fn = hashI64WHdet
	case uint:
		h.fn = hashInt
	case int:
		h.fn = hashInt
	case uintptr:
		h.fn = hashPtr
	case float32:
		h.fn = hashF32SM
	case float64:
		h.fn = hashF64SM
	case string:
		h.fn = hashStringSM
	case []byte, []int8:
		// []byte and []uint8 are identical types; both use slice handler
		h.fn = hashByteSlice
	case []int, []uint, []int16, []uint16, []int32, []uint32,
		[]int64, []uint64:
		//h.fn = hashAnySliceAsByteSlice[K]
		panic("slices of non-byte int/uint types were not 'comparable' at the time of writing this code, so no tests were possible")
	default:
		// fall back to reflect-based inspection for more cases
		t := reflect.TypeOf(zero)
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			// subtle difference: []byte (but with different declared element type) -> use slice handler
			h.fn = hashByteSlice
		} else if t.Kind() == reflect.Array && func() bool {
			// switch über das Element-Kind: für alle int- / uint-Typen true zurückgeben
			switch t.Elem().Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
				reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				return true
			default:
				return false
			}
		}() {
			// [N]byte, [N]uint16, etc. -> treat as raw bytes
			h.fn = hashAsByteArray[K]
		} else {
			// generic approach: use internal hash function from SwissMapType
			h.fn = hashFallbackMaphash[K]

			// alternative api package internal/abi is not allowed:
			//var m map[K]struct{}
			//mTyp := abi.TypeOf(m)
			//maphashhasher := (*abi.SwissMapType)(unsafe.Pointer(mTyp)).Hasher
			//h.fn = func(p unsafe.Pointer, seed uint64) uint64 {
			//	return uint64(maphashhasher(p, uintptr(seed)))
			//}
		}
	}
	return h
}

// hashBytesBlock hashes a byte slice using 64-bit block mixing.
// It consumes 8-byte words with LittleEndian decoding and mixes each
// block through splitmix64; the tail (0..7 bytes) is folded in and the
// length is incorporated to avoid collisions for different-length inputs.
func hashBytesBlock(seed uint64, b []byte) uint64 {
	h := seed ^ p0
	i, n := 0, len(b)
	for i+8 <= n {
		v := binary.NativeEndian.Uint64(b[i:])
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
	case 0:
		tail = p1
	}
	return splitmix64(h ^ tail ^ uint64(n)*p2)
}

// hashByteSlice hashes a []uint8 (alias []byte) by delegating to the
// byte-block hashing routine. This avoids per-element overhead.
func hashByteSlice(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	return hashBytesBlock(seed, b)
}

func anySliceAsByteSlice[T comparable](p unsafe.Pointer) []byte {
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

// hashByteSlice hashes a []uint8 (alias []byte) by delegating to the
// byte-block hashing routine. This avoids per-element overhead.
func hashAnySliceAsByteSlice[T comparable](p unsafe.Pointer, seed uint64) uint64 {
	b := anySliceAsByteSlice[T](p)
	return hashBytesBlock(seed, b)
}

// hashAsByteArray handles fixed-size arrays like [N]byte by treating the
// array memory as a []byte and reusing the byte-block hash implementation.
// hashAsByteArray handles fixed-size arrays (e.g. [N]byte) by viewing the
// array memory as a []byte slice and reusing the byte-block hasher. It
// works for N==0 as unsafe.Slice with length 0 is valid.
func hashAsByteArray[K comparable](p unsafe.Pointer, seed uint64) uint64 {
	// The size in bytes of the array type K equals the number of uint8 elements,
	// since element size is 1. Use unsafe.Sizeof on the dereferenced value to get it.
	size := int(unsafe.Sizeof(*(*K)(p)))
	b := unsafe.Slice((*byte)(p), size)
	return hashBytesBlock(seed, b)
}

// hashStringSM hashes a Go string by obtaining a byte view of the string
// data (without allocations) and delegating to the byte-block hasher.
func hashStringSM(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	// Go 1.20+: unsafe.StringData(s) → *byte
	b := unsafe.Slice(unsafe.StringData(s), len(s))
	return hashBytesBlock(seed, b)
}

// hashStringMH uses the hasher from  stdlib `hash/maphash` to hash
// Go strings.
func hashStringMH(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	mhs := seedToMaphashSeed(seed)
	h := maphash.String(mhs, s)
	return h
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

// hashBytesMH uses the hasher from  stdlib `hash/maphash` to hash
// byte slices
func hashBytesMH(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	mhs := seedToMaphashSeed(seed)
	h := maphash.Bytes(mhs, b)
	return h
}

// seedToMaphashSeed derives a deterministic maphash.Seed from a
// uint64 seed. This performs an unsafe copy of the 8 bytes into the
// maphash.Seed value; that relies on the concrete layout of Seed and
// is pragmatic but not guaranteed by the language spec.
func seedToMaphashSeed(seed uint64) maphash.Seed {
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
