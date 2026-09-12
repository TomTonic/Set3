package hashing

import "unsafe"

// Reading a struct's canonical word sequence.
//
// [mixWords2] and its siblings hash a sequence of 64-bit words. This file
// decides which words a type produces and emits the closure that reads them.
// See wywords.go for what the mixing is and why it replaced a chain of one
// full hash per field.
//
// # What a word can be
//
// A word is one load — eight bytes or four — optionally passed through float
// canonicalization. That vocabulary is deliberately narrow, and the narrowness
// is the whole reason this path is fast. Four richer designs were built and
// measured against it on a six-field float64 struct, where the chain this
// replaces costs 9.77 ns through the generated closure. The first row is what
// shipped:
//
//	one load width, one canonicalization branch per word   3.65 ns
//	both load widths, so two branches per word             7.22 ns
//	a switch over a per-word kind byte                     7.73 ns
//	a reader chosen by type parameter                      7.88 ns
//	word producers captured as closures                   15.68 ns
//
// One predicted branch per word is nearly free; a second one costs most of the
// win. The type parameter does not help because Go stencils by GC shape and
// the method call goes through a dictionary. Captured closures are the worst
// option by far, and not because of call overhead: every word's result is live
// at once, so the register allocator spills all of them.
//
// A plan is therefore uniform in load width, and a type whose words would be
// of mixed width keeps the chain. What the fast path covers:
//
//   - float64 and complex128 fields, canonicalized so that -0.0 and +0.0
//     cannot hash apart and every NaN hashes alike
//   - float32 and complex64 fields, likewise
//   - runs of byte-stable fields that merged into a block whose length is a
//     multiple of eight, as that many plain loads
//
// What keeps the chain, and why it is no loss:
//
//   - strings. HashString is a real call, so the calls serialize whatever the
//     data dependencies say; threading the seed through them is free and an
//     extra mixing round is pure overhead. Measured on three string fields,
//     lane mixing came out 6% slower than the chain.
//   - byte blocks whose length is not a multiple of eight. wyBlock is already
//     the right routine for them.
//   - types mixing four- and eight-byte words, such as an int32 beside a
//     float64. This is a real gap rather than a considered exclusion: it would
//     need a per-word width, which is the second branch measured above.
//   - fewer than two words, where there is nothing to overlap.

// maxPlanWords bounds a word plan. Up to eight words there is a straight-line
// mixer; beyond that the words go through a stack buffer, and past sixteen the
// buffer stops being worth it against simply threading the seed.
const maxPlanWords = 16

// wordPlan says how to read a type's canonical word sequence: the byte offset
// of each word, and whether the words are four-byte or eight-byte loads.
//
// canon carries one bit per word, bit i for word i, and applies to eight-byte
// plans only: those mix float64 fields with plain loads. A four-byte word is
// always a float32, so a narrow plan needs no mask — see [wordAt4].
type wordPlan struct {
	offs   []uintptr
	canon  uint16
	narrow bool
}

// canonF64FromBits folds the float64 bit patterns that must not hash apart
// into one representative: -0.0 becomes +0.0, because the two compare equal,
// and every NaN becomes the same quiet NaN.
//
// It works on the loaded bits rather than on a float64 so that the value never
// enters a floating-point register. That measured faster than comparing the
// float, and it is what keeps [wordAt8] inside the inliner's budget.
//
// The NaN test is one comparison rather than two. Shifting left by one drops
// the sign bit, and a left shift preserves order below 2^63, so a > +Inf<<1
// holds exactly for the patterns with an all-ones exponent and a non-zero
// mantissa.
//
// NaN is canonicalized even though a set can never find a NaN again — NaN !=
// NaN means Contains always misses. The point is that the NaNs a caller does
// insert do not spread across buckets by bit pattern.
func canonF64FromBits(u uint64) uint64 {
	a := u << 1 // the magnitude, shifted; the sign bit is gone
	if a == 0 {
		return 0 // +0.0 and -0.0
	}
	if a > 0xffe0000000000000 { // +Inf << 1
		return 0x7ff8000000000000 // every NaN
	}
	return u
}

// canonF32FromBits is [canonF64FromBits] for float32.
func canonF32FromBits(u uint32) uint32 {
	a := u << 1
	if a == 0 {
		return 0
	}
	if a > 0xff000000 { // +Inf << 1
		return 0x7fc00000
	}
	return u
}

// wordAt8 reads one eight-byte word, canonicalizing it when the low bit of
// canon is set.
//
// The load may be unaligned — a plan's offsets are field offsets, and a block
// carved into eight-byte words can start wherever its first byte-stable field
// sits. That is the same stance [read8] takes, for the same reason: Go's own
// map hashing reads unaligned on every architecture it supports.
//
// It must stay inlinable. Measured with the call left standing, six words cost
// 9.2 ns against 3.7 with it inlined, because every word's result is live at
// once and the spills eat the whole gain.
// TestWordReadersStayInlinable holds it.
func wordAt8(p unsafe.Pointer, off uintptr, canon uint16) uint64 {
	u := *(*uint64)(unsafe.Add(p, off)) //nolint:gosec
	if canon&1 != 0 {
		return canonF64FromBits(u)
	}
	return u
}

// wordAt4 reads one four-byte word as a canonicalized float32, zero-extended.
//
// It takes no canon flag because every four-byte word is a float32 — a float32
// field, or half of a complex64. Byte-stable runs only ever become eight-byte
// words, so a raw four-byte word cannot arise, and a branch for one would be
// untestable code in the hottest loop there is.
// TestNarrowPlansAreAlwaysCanonicalized holds that invariant.
//
// It must stay inlinable, for the reason given on [wordAt8].
func wordAt4(p unsafe.Pointer, off uintptr) uint64 {
	return uint64(canonF32FromBits(*(*uint32)(unsafe.Add(p, off)))) //nolint:gosec
}

// cheapWordHasher emits the closure for a plan.
func cheapWordHasher(pl wordPlan) HashFunction {
	if pl.narrow {
		return narrowWordHasher(pl.offs)
	}
	return wideWordHasher(pl.offs, pl.canon)
}

// wideWordHasher emits the closure for a plan whose words are all eight-byte loads.
func wideWordHasher(o []uintptr, canon uint16) HashFunction {
	switch len(o) {
	case 2:
		o0, o1 := o[0], o[1]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords2(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				seed)
		}
	case 3:
		o0, o1, o2 := o[0], o[1], o[2]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords3(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				seed)
		}
	case 4:
		o0, o1, o2, o3 := o[0], o[1], o[2], o[3]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords4(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				wordAt8(p, o3, canon>>3),
				seed)
		}
	case 5:
		o0, o1, o2, o3, o4 := o[0], o[1], o[2], o[3], o[4]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords5(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				wordAt8(p, o3, canon>>3),
				wordAt8(p, o4, canon>>4),
				seed)
		}
	case 6:
		o0, o1, o2, o3, o4, o5 := o[0], o[1], o[2], o[3], o[4], o[5]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords6(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				wordAt8(p, o3, canon>>3),
				wordAt8(p, o4, canon>>4),
				wordAt8(p, o5, canon>>5),
				seed)
		}
	case 7:
		o0, o1, o2, o3, o4, o5, o6 := o[0], o[1], o[2], o[3], o[4], o[5], o[6]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords7(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				wordAt8(p, o3, canon>>3),
				wordAt8(p, o4, canon>>4),
				wordAt8(p, o5, canon>>5),
				wordAt8(p, o6, canon>>6),
				seed)
		}
	case 8:
		o0, o1, o2, o3, o4, o5, o6, o7 := o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords8(
				wordAt8(p, o0, canon),
				wordAt8(p, o1, canon>>1),
				wordAt8(p, o2, canon>>2),
				wordAt8(p, o3, canon>>3),
				wordAt8(p, o4, canon>>4),
				wordAt8(p, o5, canon>>5),
				wordAt8(p, o6, canon>>6),
				wordAt8(p, o7, canon>>7),
				seed)
		}
	default:
		frozen := make([]uintptr, len(o))
		copy(frozen, o)
		return func(p unsafe.Pointer, seed uint64) uint64 {
			var buf [maxPlanWords]uint64
			w := buf[:len(frozen)]
			for i, off := range frozen {
				w[i] = wordAt8(p, off, canon>>uint(i)) //nolint:gosec
			}
			return mixWordSlice(w, seed)
		}
	}
}

// narrowWordHasher emits the closure for a plan whose words are all four-byte loads.
func narrowWordHasher(o []uintptr) HashFunction {
	switch len(o) {
	case 2:
		o0, o1 := o[0], o[1]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords2(
				wordAt4(p, o0),
				wordAt4(p, o1),
				seed)
		}
	case 3:
		o0, o1, o2 := o[0], o[1], o[2]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords3(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				seed)
		}
	case 4:
		o0, o1, o2, o3 := o[0], o[1], o[2], o[3]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords4(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				wordAt4(p, o3),
				seed)
		}
	case 5:
		o0, o1, o2, o3, o4 := o[0], o[1], o[2], o[3], o[4]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords5(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				wordAt4(p, o3),
				wordAt4(p, o4),
				seed)
		}
	case 6:
		o0, o1, o2, o3, o4, o5 := o[0], o[1], o[2], o[3], o[4], o[5]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords6(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				wordAt4(p, o3),
				wordAt4(p, o4),
				wordAt4(p, o5),
				seed)
		}
	case 7:
		o0, o1, o2, o3, o4, o5, o6 := o[0], o[1], o[2], o[3], o[4], o[5], o[6]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords7(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				wordAt4(p, o3),
				wordAt4(p, o4),
				wordAt4(p, o5),
				wordAt4(p, o6),
				seed)
		}
	case 8:
		o0, o1, o2, o3, o4, o5, o6, o7 := o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return mixWords8(
				wordAt4(p, o0),
				wordAt4(p, o1),
				wordAt4(p, o2),
				wordAt4(p, o3),
				wordAt4(p, o4),
				wordAt4(p, o5),
				wordAt4(p, o6),
				wordAt4(p, o7),
				seed)
		}
	default:
		frozen := make([]uintptr, len(o))
		copy(frozen, o)
		return func(p unsafe.Pointer, seed uint64) uint64 {
			var buf [maxPlanWords]uint64
			w := buf[:len(frozen)]
			for i, off := range frozen {
				w[i] = wordAt4(p, off) //nolint:gosec
			}
			return mixWordSlice(w, seed)
		}
	}
}

// planWords partitions merged micro-ops into a word plan and the ops that keep
// threading the seed. The plan's words come first in the emitted closure and
// the threaded ops chain from the value it returns, so the two groups are
// reordered relative to the declaration. That is harmless: the reordering is
// fixed for a type, and two values of the same type are still compared word by
// word in the same positions.
//
// ok is false when the plan would not pay for itself, in which case the caller
// keeps the chain for everything. The header of this file lists the cases.
func planWords(ops []microOp) (pl wordPlan, threaded []microOp, ok bool) {
	var wide, narrow []uintptr
	var wideCanon, narrowCanon uint16

	addWide := func(off uintptr, canon bool) {
		if canon && len(wide) < maxPlanWords {
			wideCanon |= 1 << uint(len(wide)) //nolint:gosec
		}
		wide = append(wide, off)
	}
	addNarrow := func(off uintptr, canon bool) {
		if canon && len(narrow) < maxPlanWords {
			narrowCanon |= 1 << uint(len(narrow)) //nolint:gosec
		}
		narrow = append(narrow, off)
	}

	for _, op := range ops {
		switch op.kind {
		case opFloat64:
			addWide(op.offset, true)
		case opFloat32:
			addNarrow(op.offset, true)
		case opComplex128:
			// Two canonicalized float64 words, real part first.
			addWide(op.offset, true)
			addWide(op.offset+8, true)
		case opComplex64:
			addNarrow(op.offset, true)
			addNarrow(op.offset+4, true)
		case opByteBlock:
			// Only whole eight-byte words. A block of another length keeps its
			// own call to the byte hasher, which is the routine for it anyway.
			if op.size > 0 && op.size%8 == 0 {
				for w := 0; w < op.size/8; w++ {
					addWide(op.offset+uintptr(w)*8, false) //nolint:gosec
				}
			} else {
				threaded = append(threaded, op)
			}
		case opString:
			threaded = append(threaded, op)
		default:
			// An op kind this function does not know must not be silently
			// dropped from the hash.
			return wordPlan{}, nil, false
		}
	}

	switch {
	case len(wide) > 0 && len(narrow) > 0:
		// Mixed load widths would need a second branch per word, which costs
		// more than the mixing saves. See the file header.
		return wordPlan{}, nil, false
	case len(wide) >= 2 && len(wide) <= maxPlanWords:
		return wordPlan{offs: wide, canon: wideCanon}, threaded, true
	case len(narrow) >= 2 && len(narrow) <= maxPlanWords:
		// narrowWordHasher does not read canon — every four-byte word is a
		// float32, so there is nothing to select. It is recorded anyway so
		// that TestNarrowPlansAreAlwaysCanonicalized can check that invariant
		// still holds, which is what licenses [wordAt4] to have no branch.
		return wordPlan{offs: narrow, canon: narrowCanon, narrow: true}, threaded, true
	default:
		return wordPlan{}, nil, false
	}
}

// buildWordClosure returns the word-mixing hasher for a flattened type, or nil
// when the type does not qualify and the caller should keep the chain.
func buildWordClosure(ops []microOp) HashFunction {
	pl, threaded, ok := planWords(ops)
	if !ok {
		return nil
	}

	cheap := cheapWordHasher(pl)
	if len(threaded) == 0 {
		return cheap
	}

	// The words are hashed first, and whatever could not become a word threads
	// the seed on from there. The cheap hasher already applies the plan's
	// offsets itself, so it sits at offset zero.
	fops := make([]fieldOp, 0, 1+len(threaded))
	fops = append(fops, fieldOp{fn: cheap, offset: 0})
	for _, op := range threaded {
		fops = append(fops, microOpToFieldOp(op))
	}
	return buildClosureFromFieldOps(fops)
}
