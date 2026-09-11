package hashing

import (
	"math/bits"
	"unsafe"
)

// Lane-parallel byte hashing.
//
// # Why the shape is what it is
//
// The obvious way to hash N words is a chain: h = mix(word0, h); h = mix(word1,
// h); and so on. It is simple, it mixes well, and it is slow for a reason that
// no amount of loop unrolling fixes — each mix is a widening multiply that
// cannot start until the previous one has retired. On a current x86 core that
// is three cycles per multiply, two multiplies per word, and nothing the
// processor can overlap. A 24-byte key costs eight multiplies in series.
//
// These routines consume words in independent lanes instead. Two accumulators
// that do not depend on one another issue together, and a single multiply
// combines them at the end. A 24-byte key is then two levels deep rather than
// eight, and a long input costs one multiply per 24 bytes per lane rather than
// two per 8. The structure is wyhash's own; wyhash runs three lanes over
// 48-byte blocks for exactly this reason.
//
// # Two properties that are easy to lose here, and the tests that hold them
//
// Every input byte must reach the hash. The lanes read overlapping windows
// rather than walking a tail byte by byte, and an overlap that is off by a few
// bytes leaves a hole that no statistical test will find: uniformity and
// avalanche are measured over random inputs, and an ignored byte is still
// uniform. TestEveryInputBitChangesTheHash flips every bit of every input up to
// a length well past the loop stride and requires the hash to move.
//
// Mix is a multiply, so it is zero whenever either operand is zero, and every
// operand here is an input word XORed with something. When that something is a
// public constant, an input word can be chosen to zero a lane and erase the
// word next to it — one such choice in an earlier draft of this file erased
// eight bytes of a 16-byte key for every seed. Two things answer that: laneMix
// has no operand that erases the other, and every operand is masked with the
// seed, so nothing is forceable by someone who does not know it.
// TestDegenerateOperandsDoNotEraseInput constructs the inputs that would
// trigger both.

// laneMix mixes two words without an operand that can erase the other one.
//
// Mix alone is a multiply, so it returns zero whenever either operand is zero,
// and a lane that reaches zero has forgotten everything fed into it. Folding
// both operands back in fixes that — but the fold has to be chosen with care,
// because Mix(a,1) == a, so an XOR fold would cancel exactly and leave the
// constant 1 for every a. That is the same total erasure in a different place,
// and it cost an earlier draft of this file a 2^64 collision family for any
// table whose seed happened to be one.
//
// Addition has neither degenerate case. Mix(a,b) + a + b is b when a is zero
// and a when b is zero, and for the two operands where the multiply collapses
// to the identity it is 2a+1 and 2b+1 — which lose one bit, not sixty-four.
// There is no c for which a is erased, because the XOR-fold of a widening
// multiply is not an affine function of a, and only an affine one could cancel
// against the added a.
//
// The fold is safe here because a lane's output is never the hash: it always
// passes through one more plain Mix, so no input word reaches the result
// linearly. The two adds sit off the multiply's critical path in all but the
// final cycle.
func laneMix(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	return (hi ^ lo) + a + b
}

// read8 and read4 load an unaligned word at an offset. Every call site is
// bounded by the caller's length check; the accompanying tests walk each
// boundary.
func read8(p unsafe.Pointer, off int) uint64 {
	return *(*uint64)(unsafe.Add(p, off)) //nolint:gosec
}

func read4(p unsafe.Pointer, off int) uint64 {
	return uint64(*(*uint32)(unsafe.Add(p, off))) //nolint:gosec
}

// hashLaneBytes hashes n bytes at p. It is the single body behind every
// byte-oriented entry point in this package, so a string, a byte slice and a
// fixed-size raw block holding the same bytes cannot disagree.
func hashLaneBytes(p unsafe.Pointer, n int, seed uint64) uint64 {
	switch {
	case n <= 16:
		return laneShort(p, n, seed)
	case n <= 32:
		return laneMedium(p, n, seed)
	default:
		return laneLong(p, n, seed)
	}
}

// laneCore2 and laneCore4 are the mixing bodies, separated from the reads that
// feed them.
//
// The split keeps each piece readable and lets the mixing inline into every
// caller: what varies between the length classes is which bytes are read, not
// how they are combined. Every entry point in the package ends in one of these
// two, so a string, a byte slice and a fixed-size raw block holding the same
// bytes cannot disagree about a hash value.

// laneCore2 finishes a key that fits in two words.
func laneCore2(a, b uint64, n int, seed uint64) uint64 {
	return Mix(laneMix(a^seed^P1, b^seed^P2)^P0, M5^uint64(n)) //nolint:gosec
}

// laneCore4 finishes a key covered by four words through two lanes that issue
// together.
func laneCore4(a, b, c, d uint64, n int, seed uint64) uint64 {
	x := laneMix(a^seed^P1, b^seed^P2)
	y := laneMix(c^seed^P3, d^seed^P0)
	return Mix(x^y^P0, M5^uint64(n)) //nolint:gosec
}

// laneShort hashes 0 to 16 bytes with no loop and no byte-at-a-time tail.
//
// It reads the first word and the last word and lets them overlap. For any
// length in [8,16] those two reads together cover every byte, so there is
// nothing left for a tail to assemble. Below eight bytes the same holds for
// four-byte reads, and below four the first, middle and last byte distinguish
// every length and every content at those sizes.
func laneShort(p unsafe.Pointer, n int, seed uint64) uint64 {
	var a, b uint64
	switch {
	case n >= 8:
		a = read8(p, 0)
		b = read8(p, n-8)
	case n >= 4:
		a = read4(p, 0)
		b = read4(p, n-4)
	case n > 0:
		first := uint64(*(*byte)(p))
		mid := uint64(*(*byte)(unsafe.Add(p, n>>1))) //nolint:gosec
		last := uint64(*(*byte)(unsafe.Add(p, n-1))) //nolint:gosec
		// The same three bytes go into both operands. Leaving the second one
		// constant would give the input a single multiply against a fixed
		// value, and the avalanche measured over the whole 256-value space of
		// a one-byte key was visibly short of even.
		a = first<<16 | mid<<8 | last
		b = a
	default:
		// The empty input carries no bytes, so the seed masks alone are the
		// operand pair. That is deliberate: the superseded routine hashed the
		// empty input to zero for every seed, and Set3 reseeds precisely to
		// break up a collision pattern.
		a, b = 0, 0
	}
	return laneCore2(a, b, n, seed)
}

// laneMedium hashes 17 to 32 bytes through two lanes that issue together.
//
// Four reads cover the input: the first sixteen bytes and the last sixteen,
// overlapping in the middle. Overlap costs one extra load and removes the tail
// entirely.
func laneMedium(p unsafe.Pointer, n int, seed uint64) uint64 {
	return laneCore4(read8(p, 0), read8(p, 8), read8(p, n-16), read8(p, n-8), n, seed)
}

// laneLong hashes more than 32 bytes through three lanes over 24-byte strides,
// so the chain per lane is one multiply per 24 bytes rather than two per 8.
//
// The loop stops while at least 24 bytes remain and the final round reads the
// last 24, overlapping back into what the loop already consumed. The strict
// inequality is the whole correctness argument: the loop leaves between 1 and
// 24 bytes, and a 24-byte window anchored at the end covers all of them. An
// earlier draft stopped at `rest >= 24` and closed with a 16-byte window, which
// left up to seven bytes of the input unread for 29% of all lengths — bytes
// that could be changed freely without changing the hash.
func laneLong(p unsafe.Pointer, n int, seed uint64) uint64 {
	s1, s2, s3 := seed^P1, seed^P2, seed^P3
	l1, l2, l3 := s1, s2, s3
	off, rest := 0, n
	for rest > 24 {
		l1 = laneMix(read8(p, off)^s1, l1)
		l2 = laneMix(read8(p, off+8)^s2, l2)
		l3 = laneMix(read8(p, off+16)^s3, l3)
		off += 24
		rest -= 24
	}
	l1 = laneMix(read8(p, n-24)^s1, l1)
	l2 = laneMix(read8(p, n-16)^s2, l2)
	l3 = laneMix(read8(p, n-8)^s3, l3)
	return Mix(l1^l2^l3^P0, M5^uint64(n)) //nolint:gosec
}
