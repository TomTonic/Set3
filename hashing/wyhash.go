package hashing

import "math/bits"

// ── WyHash internal constants ──────────────────────────────────────────────

// M5 is a WyHash internal mixing constant derived from the reference
// implementation. It is combined with the input length via XOR in the
// final mixing step.
const M5 = 0x1d8e4e27c47d124f

// P0, P1, P2, P3 are four primes between 2^63 and 2^64 used as mixing
// keys in the WyHash family of hash functions.
const P0 = 0xbefff2bb2ab34c2b
const P1 = 0xf20a3e5e0b7b9731
const P2 = 0xe834f7294373147d
const P3 = 0xf33958e37006e5ed

// Mix performs a 128-bit widening multiplication of a and b and returns
// the XOR of the high and low 64-bit halves. This is the core mixing
// step of the WyHash family.
func Mix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return hi ^ lo
}

// WH64Det is a deterministic variant of the WyHash-inspired 64-bit hash.
// It does not use random keys, making it faster (≈30 %) and fully
// reproducible across program runs.
//
// This function is heavily inspired by wyhash's mixing functions, especially
// the implementations from hash/maphash/maphash_purego.go from the Go
// standard library. It is an optimized version especially crafted for hashing
// 64 bit values for this hashset implementation. Please note that it is NOT bit
// compatible with wyhash; e.g., it depends on the endianness of the platform.
func WH64Det(val uint64, seed uint64) uint64 {
	a := val
	b := bits.RotateLeft64(a, 32) // swap the upper and lower 32 bits
	c := a ^ P1
	d := b ^ seed
	e := Mix(c, d)
	f := Mix(M5^8, e) // M5^8 is from wyhash, but its a constant here, so just keep it
	return f
}

// WH32DetGR is a deterministic variant of wh32 that does not use random keys.
// Widening multiplication is done inside the function using the golden ratio constant,
// which performed best in our benchmarks for the full 2^32 input range.
func WH32DetGR(val uint32, seed uint64) uint64 {
	a := GoldenRatio32 * uint64(val)
	b := a ^ P1
	c := a ^ seed
	d := Mix(b, c)
	e := Mix(M5^4, d) // M5^4 is from wyhash, but its a constant here, so just keep it
	return e
}
