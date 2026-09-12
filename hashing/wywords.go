package hashing

import "unsafe"

// Word mixing: [wyBlock] with its input handed over in registers instead of
// through memory.
//
// # Why this exists
//
// A struct that is not raw-byte-eligible cannot be hashed as a block of
// memory. Its floats need ±0 and NaN folded together, its strings are
// pointers to content rather than content, and its padding holds whatever was
// there before. The generator therefore turns such a struct into a sequence of
// canonical 64-bit *words* — one per field, or one per eight bytes of a merged
// run of byte-stable fields.
//
// Those words are then hashed exactly as [wyBlock] would hash the 8k bytes
// they would occupy if they were laid out contiguously in memory. Not
// "similarly": the functions below are transcriptions of that routine with the
// length pinned to a multiple of eight and the loads replaced by their
// register operands. mixWordsK(w0..wk-1, seed) and wyBlock over a buffer
// holding those same words return the same value, and
// TestWordMixersAreTheGenericPath holds every one of them to it.
//
// # What that buys
//
// The shape this replaces threaded the seed through one full hash per field:
// h = f0(field0, seed); h = f1(field1, h); and so on. Every field's hash
// therefore waited for the previous field's hash to finish, and each of those
// is two dependent widening multiplies. An N-field struct cost 2N multiplies
// that the processor could not overlap.
//
// wyBlock's shape consumes two words per multiply and, past six words, splits
// the work across three independent lanes. Measured on six float64 fields,
// hand-written against hand-written: 8.42 ns for the chain, 3.89 ns for this.
//
// # Why the words, and not the bytes
//
// Because a word is not the same thing as a load. A float64's word is its
// canonicalized bit pattern, so that -0.0 and +0.0 — which compare equal —
// cannot hash apart, and so that every NaN hashes alike. Padding never becomes
// a word at all, which is what makes two equal values hash equal. The word
// sequence is what the *type* means, and wyBlock is then applied to it
// unchanged.
//
// Not every field can become a word. A string is a pointer to content of an
// unknown length, and a byte run that is not a whole number of words has a
// tail no single load covers; both keep threading the seed through their own
// hash instead. wordplan.go decides which is which and says why.
//
// Note that the word count is fixed per type, so the length that goes into
// wyFinal is a constant for any given struct. Two different struct types that
// produce the same words therefore hash alike, which costs nothing: a set
// holds one type.

// mixWords2 hashes two words as wyBlock hashes sixteen bytes. Two is the
// smallest plan worth emitting: one word costs the same either way, since the
// chain's per-field hash is also two mixing steps.
func mixWords2(w0, w1, seed uint64) uint64 {
	k1 := seed ^ P1
	// The s <= 16 case: the first word and the last word. At exactly sixteen
	// bytes they are the two words themselves, with no overlap.
	return wyFinal(w0, w1, 16, k1, seed^P0)
}

// mixWords3 hashes three words as wyBlock hashes twenty-four bytes.
//
// Twenty-four bytes take the reference's default branch: one sixteen-byte
// round, and then a closing pair that overlaps back into what the round
// already consumed. That overlap is why w1 appears twice; it is the
// reference's behaviour, not a transcription slip.
func mixWords3(w0, w1, w2, seed uint64) uint64 {
	k1 := seed ^ P1
	s := Mix(w0^k1, w1^(seed^P0))
	return wyFinal(w1, w2, 24, k1, s)
}

// mixWords4 hashes four words as wyBlock hashes thirty-two bytes.
func mixWords4(w0, w1, w2, w3, seed uint64) uint64 {
	k1 := seed ^ P1
	s := Mix(w0^k1, w1^(seed^P0))
	return wyFinal(w2, w3, 32, k1, s)
}

// mixWords5 hashes five words as wyBlock hashes forty bytes.
func mixWords5(w0, w1, w2, w3, w4, seed uint64) uint64 {
	k1 := seed ^ P1
	s := Mix(w0^k1, w1^(seed^P0))
	s = Mix(w2^k1, w3^s)
	return wyFinal(w3, w4, 40, k1, s)
}

// mixWords6 hashes six words as wyBlock hashes forty-eight bytes.
func mixWords6(w0, w1, w2, w3, w4, w5, seed uint64) uint64 {
	k1 := seed ^ P1
	s := Mix(w0^k1, w1^(seed^P0))
	s = Mix(w2^k1, w3^s)
	return wyFinal(w4, w5, 48, k1, s)
}

// mixWords7 hashes seven words as wyBlock hashes fifty-six bytes.
//
// Past forty-eight bytes the reference splits into three independent lanes, so
// the three multiplies below do not depend on one another. This is the shape
// the whole file exists for.
func mixWords7(w0, w1, w2, w3, w4, w5, w6, seed uint64) uint64 {
	k1 := seed ^ P1
	s0 := seed ^ P0
	k2, k3 := k1^P1^P2, k1^P1^P3
	a := Mix(w0^k1, w1^s0)
	b := Mix(w2^k2, w3^s0)
	c := Mix(w4^k3, w5^s0)
	return wyFinal(w5, w6, 56, k1, a^b^c)
}

// mixWords8 hashes eight words as wyBlock hashes sixty-four bytes.
func mixWords8(w0, w1, w2, w3, w4, w5, w6, w7, seed uint64) uint64 {
	k1 := seed ^ P1
	s0 := seed ^ P0
	k2, k3 := k1^P1^P2, k1^P1^P3
	a := Mix(w0^k1, w1^s0)
	b := Mix(w2^k2, w3^s0)
	c := Mix(w4^k3, w5^s0)
	return wyFinal(w6, w7, 64, k1, a^b^c)
}

// mixWordSlice hashes any number of words by handing wyBlock the memory they
// already sit in. It is the path for word counts the straight-line mixers
// above do not cover; a slice of k words is k*8 contiguous bytes, so this is
// the same hash the mixers compute, reached the ordinary way.
func mixWordSlice(w []uint64, seed uint64) uint64 {
	if len(w) == 0 {
		return wyBlock(nil, 0, seed)
	}
	return wyBlock(unsafe.Pointer(&w[0]), len(w)*8, seed) //nolint:gosec
}
