package hashing

import (
	"unsafe"
)

// The byte-oriented hash, ported from the wyhash-derived routine in the Go
// runtime.
//
// # Where this comes from
//
// It is a specialization of memHashFallback in the standard library's
// internal/runtime/maps (runtime_fallback.go), the routine that hashes every
// key of every Go map on platforms without an assembly hasher. That routine is
// in turn the Go authors' adaptation of wyhash
// (https://github.com/wangyi-fudan/wyhash). The arithmetic below is theirs; the
// length dispatch, the loop structure, the overlapping reads and the final two
// mixing steps are all unchanged, and M5, P0..P3 are wyhash's constants.
//
// The specializations are:
//
//   - Native endianness only. The library hashes in memory for one process
//     family; nothing here is a wire format, and byte-order handling would put
//     a branch in the hot path for no benefit.
//   - The four secret words come from Set3's per-set seed as seed^P0..seed^P3
//     rather than from a process-global random array. See "the secret" below —
//     this is the one place where the substitution matters.
//   - The hot fixed sizes have straight-line entry points in hashgen.go. They
//     are the same arithmetic with the length pinned, and a test holds each of
//     them equal to what the generic path computes.
//
// An earlier version of this file was a hash of the author's own design that
// borrowed wyhash's shape without being wyhash. It was faster still, and it was
// wrong in three ways that the tests in wybytes_test.go now pin permanently.
// A library's hash should be one that many people have already tried to break.
//
// # The secret, and why it is derived from the seed
//
// Mix is a widening multiply, so it returns zero whenever either operand is
// zero. Every operand here is an input word XORed with a secret word, which
// means a key whose first word equals that secret zeroes the first mix, and the
// final mix of zero is zero — every such key hashes alike no matter what the
// rest of it holds. wyhash has this property, the Go runtime's routine has it,
// and this one has it.
//
// What makes it harmless is that the secret is not public. The Go runtime draws
// hashkey[] from the operating system at startup; Set3 draws a random seed per
// set and draws a fresh one whenever a rehash is triggered by a collision
// pattern. Deriving the four secret words from that seed keeps the property
// that matters — an attacker who does not know the seed cannot construct the
// erasing key, and one who guesses right for one set is wrong for the next.
//
// The derivation is affine: the four words differ from each other by public
// constants. Four independent random words would be stronger, and the reason
// this does not use them is that a HashFunction receives one uint64 and
// widening that signature would touch every routine in the package. The
// difference is bounded by the fact that nothing here claims to resist an
// attacker who already knows the seed.
//
// TestNoSeedIndependentErasure and TestErasureDoesNotSurviveAReseed hold the
// part of this that is actually load-bearing.

// read8 and read4 load an unaligned word at an offset. Go's own map hashing
// reads unaligned on every architecture it supports, which is what makes this
// portable in practice. Every call site is bounded by the caller's length
// check, and TestHashReadsExactlyTheInput walks each boundary.
func read8(p unsafe.Pointer, off int) uint64 {
	return *(*uint64)(unsafe.Add(p, off)) //nolint:gosec
}

func read4(p unsafe.Pointer, off int) uint64 {
	return uint64(*(*uint32)(unsafe.Add(p, off))) //nolint:gosec
}

// wyFinal is the two mixing steps every length class ends in:
// mix(m5^s, mix(a^secret1, b^seed)).
func wyFinal(a, b uint64, s int, k1, seed uint64) uint64 {
	return Mix(M5^uint64(s), Mix(a^k1, b^seed)) //nolint:gosec
}

// wyBlock hashes s bytes at p. It is the single body behind every byte-oriented
// entry point in this package, so a string, a byte slice and a fixed-size raw
// block holding the same bytes cannot disagree.
func wyBlock(p unsafe.Pointer, s int, seed uint64) uint64 {
	var a, b uint64
	k1 := seed ^ P1
	seed ^= P0

	switch {
	case s == 0:
		// The reference returns the seeded accumulator untouched. There is one
		// empty key, and what matters is that its hash moves with the seed:
		// the routine this replaced returned zero for every seed, which is
		// precisely what a reseed cannot fix.
		return seed
	case s < 4:
		a = uint64(*(*byte)(p))
		a |= uint64(*(*byte)(unsafe.Add(p, s>>1))) << 8 //nolint:gosec
		a |= uint64(*(*byte)(unsafe.Add(p, s-1))) << 16 //nolint:gosec
	case s == 4:
		a = read4(p, 0)
		b = a
	case s < 8:
		a = read4(p, 0)
		b = read4(p, s-4)
	case s == 8:
		a = read8(p, 0)
		b = a
	case s <= 16:
		// The first word and the last word, overlapping in the middle. For any
		// length in [8,16] the two together cover every byte, which is what
		// removes the byte-at-a-time tail.
		a = read8(p, 0)
		b = read8(p, s-8)
	default:
		l := s
		if l > 48 {
			// Three independent lanes over 48-byte strides. This is the part
			// that keeps long keys off a serial dependency chain: the three
			// mixes in a round do not depend on one another.
			// k1^P1 is the original seed, so these are seed^P2 and seed^P3
			// with the XOR of the two constants folded at compile time.
			k2, k3 := k1^P1^P2, k1^P1^P3
			seed1, seed2 := seed, seed
			for ; l > 48; l -= 48 {
				seed = Mix(read8(p, 0)^k1, read8(p, 8)^seed)
				seed1 = Mix(read8(p, 16)^k2, read8(p, 24)^seed1)
				seed2 = Mix(read8(p, 32)^k3, read8(p, 40)^seed2)
				p = unsafe.Add(p, 48) //nolint:gosec
			}
			seed ^= seed1 ^ seed2
		}
		for ; l > 16; l -= 16 {
			seed = Mix(read8(p, 0)^k1, read8(p, 8)^seed)
			p = unsafe.Add(p, 16) //nolint:gosec
		}
		// Whatever the loop left is covered by the last sixteen bytes, which
		// overlap back into what it already read.
		a = read8(p, l-16)
		b = read8(p, l-8)
	}

	return wyFinal(a, b, s, k1, seed)
}
