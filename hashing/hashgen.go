package hashing

import (
	"math"
	"reflect"
	"sync"
	"unsafe"
)

// generatedHashCache caches generated HashFunction values keyed by reflect.Type.
// Each entry is computed once by GenerateHashFunction and reused for all
// subsequent RuntimeHasher instances of the same concrete type K.
var generatedHashCache sync.Map // map[reflect.Type]HashFunction

// GenerateHashFunction attempts to build a fast, reflection-free HashFunction
// for the given type t. It analyses the type's structure at call time using
// reflect and returns a closure that hashes values of that type using only
// unsafe.Pointer arithmetic and the project's primitive hash functions —
// no reflection happens at hash time.
//
// The generated closures are designed for maximum throughput:
//   - Contiguous runs of raw-byte-eligible fields are merged into single
//     HashBytesBlock calls, maximizing cache locality and minimizing call
//     overhead.
//   - Fixed byte-block sizes (12, 16, 20, 24, 28, 32 bytes) use dedicated
//     straight-line hashers instead of the generic block loop.
//   - Special fields (float32, float64, complex64, complex128, string) are
//     handled with canonicalization logic.
//   - Small op counts (1–8) produce dedicated closures with captured
//     function pointers that Go can inline.
//   - The general N-op path uses an array of (HashFunction, offset) pairs
//     with byte-block merging to minimize the number of calls.
//
// If the type cannot be handled (e.g. interface fields), the function
// returns nil and the caller should fall back to HashFallbackMaphash.
//
// Results are cached in generatedHashCache so that repeated calls for the
// same type are essentially free.
func GenerateHashFunction(t reflect.Type) HashFunction {
	if v, ok := generatedHashCache.Load(t); ok {
		// A type the generator cannot handle is cached as a nil HashFunction,
		// so unsupported types are only analysed once. The assertion yields
		// that nil back; v itself is never a nil interface, because Store is
		// only ever handed a value of type HashFunction.
		return v.(HashFunction)
	}

	fn := buildHashFunction(t)
	generatedHashCache.Store(t, fn)
	return fn
}

// ── Field op: a (function, offset) pair ─────────────────────────────────────

// fieldOp represents one hashing step: call fn on the data at base+offset.
type fieldOp struct {
	fn     HashFunction
	offset uintptr
}

// ── Micro-op types (for flattening and merging) ─────────────────────────────

type opKind uint8

const (
	opByteBlock  opKind = iota // hash [offset, offset+size) as raw bytes
	opFloat32                  // read float32, canonicalize, mix
	opFloat64                  // read float64, canonicalize, mix
	opComplex64                // read complex64, canonicalize both parts
	opComplex128               // read complex128, canonicalize both parts
	opString                   // read string header, hash content bytes
)

type microOp struct {
	kind   opKind
	offset uintptr
	size   int // only meaningful for opByteBlock
}

// ── Top-level builder ───────────────────────────────────────────────────────

// buildHashFunction is the non-cached core of GenerateHashFunction.
func buildHashFunction(t reflect.Type) HashFunction {
	if t == nil {
		return nil
	}

	switch t.Kind() {
	case reflect.Struct:
		return buildStructHasher(t)
	case reflect.Array:
		return buildArrayHasher(t)
	case reflect.Float32:
		return HashF32SM
	case reflect.Float64:
		return HashF64WHdet
	case reflect.Complex64:
		return hashComplex64
	case reflect.Complex128:
		return hashComplex128
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr, reflect.Pointer, reflect.UnsafePointer, reflect.Chan:
		return buildScalarBytesHasher(t.Size())
	case reflect.String:
		return HashString
	default:
		return nil
	}
}

// buildScalarBytesHasher returns a HashFunction for a scalar type by size.
func buildScalarBytesHasher(size uintptr) HashFunction {
	switch size {
	case 1:
		return SwirlByte
	case 2:
		return HashI16SM
	case 4:
		return HashI32WHdet
	case 8:
		return HashI64WHdet
	default:
		s := int(size)
		return func(p unsafe.Pointer, seed uint64) uint64 {
			b := unsafe.Slice((*byte)(p), s) //nolint:gosec
			return HashBytesBlock(seed, b)
		}
	}
}

// ── Struct flattening ───────────────────────────────────────────────────────

// flattenStructOps walks a struct type recursively and produces a flat slice
// of leaf-level micro-ops. Returns nil if any field is unsupported.
// Returns a non-nil empty slice if all fields are blank.
func flattenStructOps(t reflect.Type, baseOffset uintptr) []microOp {
	ops := make([]microOp, 0, t.NumField())

	for i := range t.NumField() {
		f := t.Field(i)
		if f.Name == "_" {
			continue
		}

		off := baseOffset + f.Offset
		inner := flattenTypeOps(f.Type, off)
		if inner == nil {
			return nil
		}
		ops = append(ops, inner...)
	}
	return ops
}

// flattenTypeOps produces micro-ops for a single type at the given offset.
func flattenTypeOps(ft reflect.Type, off uintptr) []microOp {
	switch ft.Kind() {
	case reflect.Float32:
		return []microOp{{kind: opFloat32, offset: off}}
	case reflect.Float64:
		return []microOp{{kind: opFloat64, offset: off}}
	case reflect.Complex64:
		return []microOp{{kind: opComplex64, offset: off}}
	case reflect.Complex128:
		return []microOp{{kind: opComplex128, offset: off}}
	case reflect.String:
		return []microOp{{kind: opString, offset: off}}
	case reflect.Struct:
		return flattenStructOps(ft, off)
	case reflect.Array:
		return flattenArrayOps(ft, off)
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr, reflect.Pointer, reflect.UnsafePointer, reflect.Chan:
		return []microOp{{kind: opByteBlock, offset: off, size: int(ft.Size())}}
	default:
		return nil
	}
}

// flattenArrayOps produces micro-ops for an array type.
func flattenArrayOps(t reflect.Type, baseOffset uintptr) []microOp {
	arrLen := t.Len()
	if arrLen == 0 {
		return []microOp{}
	}

	if CanUseUnsafeRawByteBlockHasherType(t).Eligible {
		return []microOp{{kind: opByteBlock, offset: baseOffset, size: int(t.Size())}}
	}

	elemType := t.Elem()
	elemSize := elemType.Size()
	// Deliberately non-nil, matching flattenStructOps: an array whose elements
	// contribute no ops -- an array of blank-only structs, say -- is hashable as
	// a constant, and a nil result here would instead be read as a refusal and
	// send the type to the maphash fallback.
	ops := make([]microOp, 0, arrLen)
	for i := range arrLen {
		off := baseOffset + uintptr(i)*elemSize
		inner := flattenTypeOps(elemType, off)
		if inner == nil {
			return nil
		}
		ops = append(ops, inner...)
	}
	return ops
}

// ── Byte-block merging ──────────────────────────────────────────────────────

// mergeByteBlocks coalesces adjacent opByteBlock ops into larger blocks.
func mergeByteBlocks(ops []microOp) []microOp {
	if len(ops) == 0 {
		return ops
	}
	merged := make([]microOp, 0, len(ops))
	cur := ops[0]
	for _, op := range ops[1:] {
		if cur.kind == opByteBlock && op.kind == opByteBlock &&
			op.offset == cur.offset+uintptr(cur.size) {
			cur.size += op.size
		} else {
			merged = append(merged, cur)
			cur = op
		}
	}
	return append(merged, cur)
}

// ── Convert micro-ops to fieldOps ───────────────────────────────────────────

// microOpToFieldOp converts a micro-op into a fieldOp with a concrete
// HashFunction. Hot byte-block sizes are routed to specialized helpers to
// avoid the generic HashBytesBlock loop and tail dispatch.
func microOpToFieldOp(op microOp) fieldOp {
	switch op.kind {
	case opByteBlock:
		size := op.size
		var fn HashFunction
		switch size {
		case 1:
			fn = SwirlByte
		case 2:
			fn = HashI16SM
		case 4:
			fn = HashI32WHdet
		case 8:
			fn = HashI64WHdet
		default:
			if specialized := fixedSizeByteBlockHasher(size); specialized != nil {
				fn = specialized
			} else {
				// wyBlock rather than HashBytesBlock: the slice would only be
				// built so that HashBytesBlock could recover the pointer from
				// it again. One step fewer, not a measurable one.
				fn = func(p unsafe.Pointer, seed uint64) uint64 {
					return wyBlock(p, size, seed)
				}
			}
		}
		return fieldOp{fn: fn, offset: op.offset}
	case opFloat32:
		return fieldOp{fn: hashFloat32Inline, offset: op.offset}
	case opFloat64:
		return fieldOp{fn: hashFloat64Inline, offset: op.offset}
	case opComplex64:
		return fieldOp{fn: hashComplex64, offset: op.offset}
	case opComplex128:
		return fieldOp{fn: hashComplex128, offset: op.offset}
	case opString:
		return fieldOp{fn: HashString, offset: op.offset}
	default:
		return fieldOp{fn: func(_ unsafe.Pointer, seed uint64) uint64 { return seed }, offset: op.offset}
	}
}

// ── Struct & array hashers ──────────────────────────────────────────────────

// buildStructHasher flattens a struct into micro-ops, merges byte blocks,
// and emits an optimal closure.
func buildStructHasher(t reflect.Type) HashFunction {
	if t.NumField() == 0 {
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	ops := flattenStructOps(t, 0)
	if ops == nil {
		return nil
	}
	if len(ops) == 0 {
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	return buildClosureFromOps(mergeByteBlocks(ops))
}

// buildArrayHasher generates a HashFunction for array types.
func buildArrayHasher(t reflect.Type) HashFunction {
	if t.Len() == 0 {
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	if CanUseUnsafeRawByteBlockHasherType(t).Eligible {
		totalSize := int(t.Size())
		if specialized := fixedSizeByteBlockHasher(totalSize); specialized != nil {
			return specialized
		}
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return wyBlock(p, totalSize, seed)
		}
	}

	ops := flattenArrayOps(t, 0)
	if ops == nil {
		return nil
	}
	if len(ops) == 0 {
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	return buildClosureFromOps(mergeByteBlocks(ops))
}

// ── Closure construction ────────────────────────────────────────────────────

// buildClosureFromOps creates a single HashFunction closure from micro-ops.
// For 1–8 ops, fully unrolled closures with captured function pointers are
// emitted so that Go can inline the inner calls. For larger counts, a tight
// loop over a frozen fieldOp slice is used.
func buildClosureFromOps(ops []microOp) HashFunction {
	switch len(ops) {
	case 0:
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	case 1:
		fop0 := microOpToFieldOp(ops[0])
		fn0, off0 := fop0.fn, fop0.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			return fn0(unsafe.Add(p, off0), seed) //nolint:gosec
		}
	case 2:
		fop0, fop1 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			return fn1(unsafe.Add(p, off1), h)  //nolint:gosec
		}
	case 3:
		fop0, fop1, fop2 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			return fn2(unsafe.Add(p, off2), h)  //nolint:gosec
		}
	case 4:
		fop0, fop1, fop2, fop3 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2]), microOpToFieldOp(ops[3])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		fn3, off3 := fop3.fn, fop3.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			h = fn2(unsafe.Add(p, off2), h)     //nolint:gosec
			return fn3(unsafe.Add(p, off3), h)  //nolint:gosec
		}
	case 5:
		fop0, fop1, fop2, fop3, fop4 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2]), microOpToFieldOp(ops[3]), microOpToFieldOp(ops[4])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		fn3, off3 := fop3.fn, fop3.offset
		fn4, off4 := fop4.fn, fop4.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			h = fn2(unsafe.Add(p, off2), h)     //nolint:gosec
			h = fn3(unsafe.Add(p, off3), h)     //nolint:gosec
			return fn4(unsafe.Add(p, off4), h)  //nolint:gosec
		}
	case 6:
		fop0, fop1, fop2, fop3, fop4, fop5 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2]), microOpToFieldOp(ops[3]), microOpToFieldOp(ops[4]), microOpToFieldOp(ops[5])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		fn3, off3 := fop3.fn, fop3.offset
		fn4, off4 := fop4.fn, fop4.offset
		fn5, off5 := fop5.fn, fop5.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			h = fn2(unsafe.Add(p, off2), h)     //nolint:gosec
			h = fn3(unsafe.Add(p, off3), h)     //nolint:gosec
			h = fn4(unsafe.Add(p, off4), h)     //nolint:gosec
			return fn5(unsafe.Add(p, off5), h)  //nolint:gosec
		}
	case 7:
		fop0, fop1, fop2, fop3, fop4, fop5, fop6 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2]), microOpToFieldOp(ops[3]), microOpToFieldOp(ops[4]), microOpToFieldOp(ops[5]), microOpToFieldOp(ops[6])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		fn3, off3 := fop3.fn, fop3.offset
		fn4, off4 := fop4.fn, fop4.offset
		fn5, off5 := fop5.fn, fop5.offset
		fn6, off6 := fop6.fn, fop6.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			h = fn2(unsafe.Add(p, off2), h)     //nolint:gosec
			h = fn3(unsafe.Add(p, off3), h)     //nolint:gosec
			h = fn4(unsafe.Add(p, off4), h)     //nolint:gosec
			h = fn5(unsafe.Add(p, off5), h)     //nolint:gosec
			return fn6(unsafe.Add(p, off6), h)  //nolint:gosec
		}
	case 8:
		fop0, fop1, fop2, fop3, fop4, fop5, fop6, fop7 := microOpToFieldOp(ops[0]), microOpToFieldOp(ops[1]), microOpToFieldOp(ops[2]), microOpToFieldOp(ops[3]), microOpToFieldOp(ops[4]), microOpToFieldOp(ops[5]), microOpToFieldOp(ops[6]), microOpToFieldOp(ops[7])
		fn0, off0 := fop0.fn, fop0.offset
		fn1, off1 := fop1.fn, fop1.offset
		fn2, off2 := fop2.fn, fop2.offset
		fn3, off3 := fop3.fn, fop3.offset
		fn4, off4 := fop4.fn, fop4.offset
		fn5, off5 := fop5.fn, fop5.offset
		fn6, off6 := fop6.fn, fop6.offset
		fn7, off7 := fop7.fn, fop7.offset
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := fn0(unsafe.Add(p, off0), seed) //nolint:gosec
			h = fn1(unsafe.Add(p, off1), h)     //nolint:gosec
			h = fn2(unsafe.Add(p, off2), h)     //nolint:gosec
			h = fn3(unsafe.Add(p, off3), h)     //nolint:gosec
			h = fn4(unsafe.Add(p, off4), h)     //nolint:gosec
			h = fn5(unsafe.Add(p, off5), h)     //nolint:gosec
			h = fn6(unsafe.Add(p, off6), h)     //nolint:gosec
			return fn7(unsafe.Add(p, off7), h)  //nolint:gosec
		}
	default:
		frozen := make([]fieldOp, len(ops))
		for i, op := range ops {
			frozen[i] = microOpToFieldOp(op)
		}
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := seed
			for _, fop := range frozen {
				h = fop.fn(unsafe.Add(p, fop.offset), h) //nolint:gosec
			}
			return h
		}
	}
}

// ── Fixed byte-block hash helpers ──────────────────────────────────────────

// fixedSizeByteBlockHasher returns a specialized straight-line hasher for hot
// fixed byte-block sizes, or nil when the generic block hasher should be used.
func fixedSizeByteBlockHasher(size int) HashFunction {
	switch size {
	case 12:
		return hashByteBlock12
	case 16:
		return hashByteBlock16
	case 20:
		return hashByteBlock20
	case 24:
		return hashByteBlock24
	case 28:
		return hashByteBlock28
	case 32:
		return hashByteBlock32
	default:
		return nil
	}
}

// The fixed-size helpers below are [wyBlock] with the length pinned to a
// constant: the same reads, the same mixes, the same order, with the length
// switch and the loop bounds resolved at compile time.
//
// They are transcriptions rather than calls because the generic body is past
// the inliner's budget, and these are the shapes a struct key actually takes.
// A transcription can drift, so TestFixedBlockHelpersAreTheGenericPath holds
// each one equal to what the generic path computes for that length, over
// random inputs and several seeds. That test is the reason the duplication is
// safe; do not remove it.
//
// Sizes 20 to 32 take the reference's default branch, which runs its 16-byte
// round once and then reads the last sixteen bytes — overlapping back into what
// the round already consumed. That is where the a and b offsets below come
// from: after one round the cursor stands at 16 and l is size-16, so a is at
// size-16 and b at size-8.

// hashByteBlock12 hashes exactly 12 bytes.
func hashByteBlock12(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	return wyFinal(read8(p, 0), read8(p, 4), 12, k1, seed^P0)
}

// hashByteBlock16 hashes exactly 16 bytes.
func hashByteBlock16(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	return wyFinal(read8(p, 0), read8(p, 8), 16, k1, seed^P0)
}

// hashByteBlock20 hashes exactly 20 bytes.
func hashByteBlock20(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	h := Mix(read8(p, 0)^k1, read8(p, 8)^(seed^P0))
	return wyFinal(read8(p, 4), read8(p, 12), 20, k1, h)
}

// hashByteBlock24 hashes exactly 24 bytes, the size of a three-field uint64
// struct key and the shape the raw-block hasher sees most often.
func hashByteBlock24(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	h := Mix(read8(p, 0)^k1, read8(p, 8)^(seed^P0))
	return wyFinal(read8(p, 8), read8(p, 16), 24, k1, h)
}

// hashByteBlock28 hashes exactly 28 bytes.
func hashByteBlock28(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	h := Mix(read8(p, 0)^k1, read8(p, 8)^(seed^P0))
	return wyFinal(read8(p, 12), read8(p, 20), 28, k1, h)
}

// hashByteBlock32 hashes exactly 32 bytes.
func hashByteBlock32(p unsafe.Pointer, seed uint64) uint64 {
	k1 := seed ^ P1
	h := Mix(read8(p, 0)^k1, read8(p, 8)^(seed^P0))
	return wyFinal(read8(p, 16), read8(p, 24), 32, k1, h)
}

// ── Inline hash helpers ─────────────────────────────────────────────────────

// hashFloat32Inline hashes a float32 value with ±0 and NaN canonicalization.
func hashFloat32Inline(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float32)(p)
	var bits uint32
	switch {
	case f == 0:
		bits = 0
	case math.IsNaN(float64(f)):
		bits = 0x7fc00000
	default:
		bits = math.Float32bits(f)
	}
	v := 0x0000000100000001 * uint64(bits)
	return Splitmix64(seed ^ v)
}

// hashFloat64Inline hashes a float64 value with ±0 and NaN canonicalization.
func hashFloat64Inline(p unsafe.Pointer, seed uint64) uint64 {
	f := *(*float64)(p)
	var bits uint64
	switch {
	case f == 0:
		bits = 0
	case math.IsNaN(f):
		bits = 0x7ff8000000000000
	default:
		bits = math.Float64bits(f)
	}
	return WH64Det(bits, seed)
}

// ── Complex hashers ─────────────────────────────────────────────────────────

// hashComplex64 hashes a complex64 value by canonicalizing and hashing
// both float32 components and combining the results.
func hashComplex64(p unsafe.Pointer, seed uint64) uint64 {
	c := *(*complex64)(p)
	r, i := real(c), imag(c)
	h := WH32DetGR(canonicalF32Bits(r), seed)
	return WH32DetGR(canonicalF32Bits(i), h)
}

// hashComplex128 hashes a complex128 value by canonicalizing and hashing
// both float64 components and combining the results.
func hashComplex128(p unsafe.Pointer, seed uint64) uint64 {
	c := *(*complex128)(p)
	r, i := real(c), imag(c)
	h := WH64Det(canonicalF64Bits(r), seed)
	return WH64Det(canonicalF64Bits(i), h)
}

// canonicalF32Bits returns the IEEE-754 bit representation of f after
// canonicalizing ±0 → +0 and all NaN → a single canonical NaN.
func canonicalF32Bits(f float32) uint32 {
	switch {
	case f == 0:
		return 0
	case math.IsNaN(float64(f)):
		return 0x7fc00000
	default:
		return math.Float32bits(f)
	}
}

// canonicalF64Bits returns the IEEE-754 bit representation of f after
// canonicalizing ±0 → +0 and all NaN → a single canonical NaN.
func canonicalF64Bits(f float64) uint64 {
	switch {
	case f == 0:
		return 0
	case math.IsNaN(f):
		return 0x7ff8000000000000
	default:
		return math.Float64bits(f)
	}
}
