package set3

import (
	"crypto/rand"
	"encoding/binary"
	"math/bits"
	"unsafe"
)

const m5 = 0x1d8e4e27c47d124f

var hashkey [4]uint64

func init() {
	b := unsafe.Slice((*byte)(unsafe.Pointer(&hashkey[0])), 32)
	rand.Read(b)
}

// This function is heavily inspired by wyhash's mixing functions, especially
// the implementations from hash/maphash/maphash_purego.go from the Go
// standard library. It is an optimized version especially crafted for hashing
// 64 bit values for this hashset implementation. Please note that it is NOT bit
// compatible with wyhash; e.g., it depends on the endianness of the platform.
func wh64(val uint64, seed uint64) uint64 {
	seed ^= hashkey[0]
	a := val
	b := bits.RotateLeft64(a, 32) // swap the upper and lower 32 bits
	c := a ^ hashkey[1]
	d := b ^ seed
	e := mix(c, d)
	f := mix(m5^8, e) // m5^8 is from wyhash, but its a constant here, so just keep it
	return f
}

// wh64det is a deterministic variant of wh64 that does not use random keys.
// It is 30% faster than wh64 due to the lack of extra key usage.
func wh64det(val uint64, seed uint64) uint64 {
	a := val
	b := bits.RotateLeft64(a, 32) // swap the upper and lower 32 bits
	c := a ^ p1
	d := b ^ seed
	e := mix(c, d)
	f := mix(m5^8, e) // m5^8 is from wyhash, but its a constant here, so just keep it
	return f
}

func wh32(val uint32, seed uint64) uint64 {
	seed ^= hashkey[0]
	a := 0x0000_0001_0000_0001 * uint64(val)
	b := a ^ hashkey[1]
	c := a ^ seed
	d := mix(b, c)
	e := mix(m5^4, d) // m5^4 is from wyhash, but its a constant here, so just keep it
	return e
}

// wh32det is a deterministic variant of wh32 that does not use random keys.
func wh32det(val uint32, seed uint64) uint64 {
	a := 0x0000_0001_0000_0001 * uint64(val)
	b := a ^ p1
	c := a ^ seed
	d := mix(b, c)
	e := mix(m5^4, d) // m5^4 is from wyhash, but its a constant here, so just keep it
	return e
}

func wh16(val uint16, seed uint64) uint64 {
	seed ^= hashkey[0]
	// a := (uint64(val&0xff00) << 8) | (uint64(val&0x00ff) << 8) | uint64(val&0x00ff)  // original wyhash logic
	a := 0x0001_0001_0001_0001 * uint64(val)
	b := a ^ hashkey[1]
	c := a ^ seed
	d := mix(b, c)
	e := mix(m5^2, d) // m5^2 is from wyhash, but its a constant here, so just keep it
	return e
}

// does not actually perform better than swirlByte for our purpose but kept for now
func wh8(val byte, seed uint64) uint64 {
	seed ^= hashkey[0]
	// a := (uint64(val) << 16) | (uint64(val) << 8) | uint64(val) // original wyhash logic
	a := 0x0001_0001_0001_0001 * uint64(val)
	b := a ^ hashkey[1]
	c := a ^ seed
	d := mix(b, c)
	e := mix(m5^1, d) // m5^1 is from wyhash, but its a constant here, so just keep it
	return e
}

// hashStringWH hashes a Go string by obtaining a byte view of the string
// data (without allocations) and delegating to the byte-block hasher.
func hashStringWH(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	// Go 1.20+: unsafe.StringData(s) → *byte
	b := unsafe.Slice(unsafe.StringData(s), len(s))
	return whX(b, seed)
}

// hashByteSlice hashes a []uint8 (alias []byte) by delegating to the
// byte-block hashing routine. This avoids per-element overhead.
func hashByteSliceWH(p unsafe.Pointer, seed uint64) uint64 {
	b := *(*[]uint8)(p)
	return whX(b, seed)
}

func mix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return hi ^ lo
}

// 4 primes between 2^63 and 2^64
const p0 = 0xbefff2bb2ab34c2b
const p1 = 0xf20a3e5e0b7b9731
const p2 = 0xe834f7294373147d
const p3 = 0xf33958e37006e5ed

// derived from func memhashFallback() from runtime/hash64.go
func whX(data []byte, seed uint64) uint64 {
	var a, b uint64
	numBytes := uint64(len(data))
	seed ^= p0
	switch {
	case numBytes == 0:
		return seed
	case numBytes < 4:
		a = uint64(data[0]) | (uint64(data[numBytes>>1]) << 8) | (uint64(data[numBytes-1]) << 16)
		b = 0
	case numBytes < 8:
		a = uint64(binary.NativeEndian.Uint32(data))
		b = uint64(binary.NativeEndian.Uint32(data[numBytes-4:])) // 0 <= numBytes-4 < 4 -> data[0:]..data[3:]
	case numBytes < 16:
		a = binary.NativeEndian.Uint64(data)
		b = binary.NativeEndian.Uint64(data[numBytes-8:]) // 0 <= numBytes-8 < 8 -> data[0:]..data[7:]
	default:
		p := data
		remaining := numBytes
		if remaining > 48 {
			seed1 := seed
			seed2 := seed
			for ; remaining > 48; remaining -= 48 {
				seed = mix(binary.NativeEndian.Uint64(p)^p1, binary.NativeEndian.Uint64(p[8:])^seed)
				seed1 = mix(binary.NativeEndian.Uint64(p[16:])^p2, binary.NativeEndian.Uint64(p[24:])^seed1)
				seed2 = mix(binary.NativeEndian.Uint64(p[32:])^p3, binary.NativeEndian.Uint64(p[40:])^seed2)
				p = p[48:]
			}
			seed ^= seed1 ^ seed2
		}
		for ; remaining > 16; remaining -= 16 {
			seed = mix(binary.NativeEndian.Uint64(p)^p1, binary.NativeEndian.Uint64(p[8:])^seed)
			p = p[16:]
		}
		a = binary.NativeEndian.Uint64(data[numBytes-16:])
		b = binary.NativeEndian.Uint64(data[numBytes-8:])
	}
	return mix(m5^numBytes, mix(a^p1, b^seed))
}

// ======= Below are functions copied and adapted from go library for testing

// memhashFallbackPort is a deterministic variant of func memhashFallback() from runtime/hash64.go
// only used for testing purposes.
func memhashFallbackPort(p unsafe.Pointer, numBytes, seed uint64) uint64 {
	var a, b uint64
	seed ^= p0
	switch {
	case numBytes == 0:
		return seed
	case numBytes < 4:
		a = uint64(*(*byte)(p))
		a |= uint64(*(*byte)(add(p, numBytes>>1))) << 8
		a |= uint64(*(*byte)(add(p, numBytes-1))) << 16
	case numBytes == 4:
		a = r4p(p)
		b = a
	case numBytes < 8:
		a = r4p(p)
		b = r4p(add(p, numBytes-4))
	case numBytes == 8:
		a = r8p(p)
		b = a
	case numBytes <= 16:
		a = r8p(p)
		b = r8p(add(p, numBytes-8))
	default:
		l := numBytes
		if l > 48 {
			seed1 := seed
			seed2 := seed
			for ; l > 48; l -= 48 {
				seed = mix(r8p(p)^p1, r8p(add(p, 8))^seed)
				seed1 = mix(r8p(add(p, 16))^p2, r8p(add(p, 24))^seed1)
				seed2 = mix(r8p(add(p, 32))^p3, r8p(add(p, 40))^seed2)
				p = add(p, 48)
			}
			seed ^= seed1 ^ seed2
		}
		for ; l > 16; l -= 16 {
			seed = mix(r8p(p)^p1, r8p(add(p, 8))^seed)
			p = add(p, 16)
		}
		a = r8p(add(p, l-16))
		b = r8p(add(p, l-8))
	}
	return mix(m5^numBytes, mix(a^p1, b^seed))
}

func readUnaligned32(p unsafe.Pointer) uint32 { return *(*uint32)(p) }
func readUnaligned64(p unsafe.Pointer) uint64 { return *(*uint64)(p) }

func r4p(p unsafe.Pointer) uint64 {
	return uint64(readUnaligned32(p))
}

func r8p(p unsafe.Pointer) uint64 {
	return uint64(readUnaligned64(p))
}

// Should be a built-in for unsafe.Pointer?
//
// add should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - fortio.org/log
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname add
//go:nosplit
func add(p unsafe.Pointer, x uint64) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(x))
}
