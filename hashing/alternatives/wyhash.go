package alternatives

import (
	"crypto/rand"
	"encoding/binary"
	"math/bits"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

// Hashkey holds four random 64-bit keys initialized at process startup.
// These are used by the non-deterministic WyHash variants (WH64, WH32,
// WH16, WH8) to provide per-process randomisation.
var Hashkey [4]uint64

func init() {
	b := unsafe.Slice((*byte)(unsafe.Pointer(&Hashkey[0])), 32)
	_, _ = rand.Read(b)
}

// WH64 is a non-deterministic WyHash variant for 64-bit values.
// This function is heavily inspired by wyhash's mixing functions, especially
// the implementations from hash/maphash/maphash_purego.go from the Go
// standard library. It is an optimized version especially crafted for hashing
// 64 bit values for this hashset implementation. Please note that it is NOT bit
// compatible with wyhash; e.g., it depends on the endianness of the platform.
func WH64(val uint64, seed uint64) uint64 {
	seed ^= Hashkey[0]
	a := val
	b := bits.RotateLeft64(a, 32) // swap the upper and lower 32 bits
	c := a ^ Hashkey[1]
	d := b ^ seed
	e := hashing.Mix(c, d)
	f := hashing.Mix(hashing.M5^8, e) // m5^8 is from wyhash, but its a constant here, so just keep it
	return f
}

// WH32 is a non-deterministic WyHash variant for 32-bit values.
func WH32(val uint32, seed uint64) uint64 {
	seed ^= Hashkey[0]
	a := 0x0000_0001_0000_0001 * uint64(val)
	b := a ^ Hashkey[1]
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^4, d) // m5^4 is from wyhash, but its a constant here, so just keep it
	return e
}

// WH32DetExtMul is a deterministic variant of WH32 that does not use random keys.
// The widening multiplication must be done BEFORE this function is called, it is for benchmarking purposes only.
func WH32DetExtMul(val uint64, seed uint64) uint64 {
	a := val
	b := a ^ hashing.P1
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^4, d) // m5^4 is from wyhash, but its a constant here, so just keep it
	return e
}

// WH16DetExtMul is the external-widening variant of WH16Det, analogous to WH32DetExtMul.
// The widening multiplication must be done BEFORE this function is called;
// this function exists to allow testing different widening constants.
func WH16DetExtMul(val uint64, seed uint64) uint64 {
	a := val
	b := a ^ hashing.P1
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^2, d) // m5^2 mirrors the wh16 constant (2 bytes of input)
	return e
}

// WH16Det is a deterministic variant of WH16 that does not use random keys.
func WH16Det(val uint16, seed uint64) uint64 {
	a := 0x0001_0001_0001_0001 * uint64(val)
	b := a ^ hashing.P1
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^2, d) // m5^2 mirrors the wh16 constant
	return e
}

// WH16 is a non-deterministic WyHash variant for 16-bit values.
func WH16(val uint16, seed uint64) uint64 {
	seed ^= Hashkey[0]
	// a := (uint64(val&0xff00) << 8) | (uint64(val&0x00ff) << 8) | uint64(val&0x00ff)  // original wyhash logic
	a := 0x0001_0001_0001_0001 * uint64(val)
	b := a ^ Hashkey[1]
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^2, d) // m5^2 is from wyhash, but its a constant here, so just keep it
	return e
}

// WH8 is a non-deterministic WyHash variant for byte values.
// Does not actually perform better than SwirlByte for our purpose but kept for now.
func WH8(val byte, seed uint64) uint64 {
	seed ^= Hashkey[0]
	// a := (uint64(val) << 16) | (uint64(val) << 8) | uint64(val) // original wyhash logic
	a := 0x0001_0001_0001_0001 * uint64(val)
	b := a ^ Hashkey[1]
	c := a ^ seed
	d := hashing.Mix(b, c)
	e := hashing.Mix(hashing.M5^1, d) // m5^1 is from wyhash, but its a constant here, so just keep it
	return e
}

// WhX is a byte-block hash function derived from func memhashFallback() from
// runtime/hash64.go. It uses the WyHash mixing primitives.
func WhX(data []byte, seed uint64) uint64 {
	var a, b uint64
	numBytes := uint64(len(data))
	seed ^= hashing.P0
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
				seed = hashing.Mix(binary.NativeEndian.Uint64(p)^hashing.P1, binary.NativeEndian.Uint64(p[8:])^seed)
				seed1 = hashing.Mix(binary.NativeEndian.Uint64(p[16:])^hashing.P2, binary.NativeEndian.Uint64(p[24:])^seed1)
				seed2 = hashing.Mix(binary.NativeEndian.Uint64(p[32:])^hashing.P3, binary.NativeEndian.Uint64(p[40:])^seed2)
				p = p[48:]
			}
			seed ^= seed1 ^ seed2
		}
		for ; remaining > 16; remaining -= 16 {
			seed = hashing.Mix(binary.NativeEndian.Uint64(p)^hashing.P1, binary.NativeEndian.Uint64(p[8:])^seed)
			p = p[16:]
		}
		a = binary.NativeEndian.Uint64(data[numBytes-16:])
		b = binary.NativeEndian.Uint64(data[numBytes-8:])
	}
	return hashing.Mix(hashing.M5^numBytes, hashing.Mix(a^hashing.P1, b^seed))
}

// ======= Below are functions copied and adapted from go library for testing

// MemhashFallbackPort is a deterministic variant of func memhashFallback() from runtime/hash64.go
// only used for testing purposes.
func MemhashFallbackPort(p unsafe.Pointer, numBytes, seed uint64) uint64 {
	var a, b uint64
	seed ^= hashing.P0
	switch {
	case numBytes == 0:
		return seed
	case numBytes < 4:
		a = uint64(*(*byte)(p))
		a |= uint64(*(*byte)(Add(p, numBytes>>1))) << 8
		a |= uint64(*(*byte)(Add(p, numBytes-1))) << 16
	case numBytes == 4:
		a = R4p(p)
		b = a
	case numBytes < 8:
		a = R4p(p)
		b = R4p(Add(p, numBytes-4))
	case numBytes == 8:
		a = R8p(p)
		b = a
	case numBytes <= 16:
		a = R8p(p)
		b = R8p(Add(p, numBytes-8))
	default:
		l := numBytes
		if l > 48 {
			seed1 := seed
			seed2 := seed
			for ; l > 48; l -= 48 {
				seed = hashing.Mix(R8p(p)^hashing.P1, R8p(Add(p, 8))^seed)
				seed1 = hashing.Mix(R8p(Add(p, 16))^hashing.P2, R8p(Add(p, 24))^seed1)
				seed2 = hashing.Mix(R8p(Add(p, 32))^hashing.P3, R8p(Add(p, 40))^seed2)
				p = Add(p, 48)
			}
			seed ^= seed1 ^ seed2
		}
		for ; l > 16; l -= 16 {
			seed = hashing.Mix(R8p(p)^hashing.P1, R8p(Add(p, 8))^seed)
			p = Add(p, 16)
		}
		a = R8p(Add(p, l-16))
		b = R8p(Add(p, l-8))
	}
	return hashing.Mix(hashing.M5^numBytes, hashing.Mix(a^hashing.P1, b^seed))
}

func readUnaligned32(p unsafe.Pointer) uint32 { return *(*uint32)(p) }
func readUnaligned64(p unsafe.Pointer) uint64 { return *(*uint64)(p) }

// R4p reads 4 bytes from the given pointer and returns them as a uint64.
func R4p(p unsafe.Pointer) uint64 {
	return uint64(readUnaligned32(p))
}

// R8p reads 8 bytes from the given pointer and returns them as a uint64.
func R8p(p unsafe.Pointer) uint64 {
	return uint64(readUnaligned64(p))
}

// Add performs pointer arithmetic: it returns p + x.
//
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
//go:nosplit
func Add(p unsafe.Pointer, x uint64) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(x))
}
