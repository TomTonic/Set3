// nolint:gosec // All unsafe operations below are audited: field offsets are computed via reflect at init time.
package hashing

import (
	"math"
	"reflect"
	"sync"
	"unsafe"
)

// generatedHashCache caches generated HashFunction values keyed by reflect.Type.
// Each entry is computed once by generateHashFunction and reused for all
// subsequent RuntimeHasher instances of the same concrete type K.
var generatedHashCache sync.Map // map[reflect.Type]HashFunction

// GenerateHashFunction attempts to build a fast, reflection-free HashFunction
// for the given type t. It analyses the type's structure at call time using
// reflect and returns a closure that hashes values of that type using only
// unsafe.Pointer arithmetic and the project's primitive hash functions —
// no reflection happens at hash time.
//
// The function handles:
//   - Structs with any combination of scalar, float, complex, string,
//     pointer, array, and nested struct fields (including padding).
//   - Arrays whose element type can be handled.
//   - Scalar types that need canonicalization (float32, float64,
//     complex64, complex128).
//
// If the type cannot be handled (e.g. interface fields), the function
// returns nil and the caller should fall back to HashFallbackMaphash.
//
// Results are cached in generatedHashCache so that repeated calls for the
// same type are essentially free.
func GenerateHashFunction(t reflect.Type) HashFunction {
	if v, ok := generatedHashCache.Load(t); ok {
		if v == nil {
			return nil
		}
		return v.(HashFunction)
	}

	fn := buildHashFunction(t)
	generatedHashCache.Store(t, fn)
	return fn
}

// fieldHashOp describes how to hash a single fixed-size region of a value.
// offset is the byte offset from the value's base pointer.
// hasher is a HashFunction that hashes the bytes at that offset.
type fieldHashOp struct {
	offset uintptr
	hasher HashFunction
}

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

	// Scalars that are safe for raw-byte hashing — we can just read the bytes.
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr,
		reflect.Pointer,
		reflect.UnsafePointer,
		reflect.Chan:
		return buildScalarBytesHasher(t.Size())

	case reflect.String:
		return HashString

	default:
		// Interface, Func, Slice, Map — cannot generate a fast hasher.
		return nil
	}
}

// buildScalarBytesHasher returns a HashFunction that reads size bytes at the
// pointer and hashes them through HashBytesBlock. Used for scalar types that
// are safe for raw-byte hashing but were reached via a named/defined type
// (so they didn't match the type switch in MakeRuntimeHasher).
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
		// Unusual size — fall back to byte block.
		s := int(size)
		return func(p unsafe.Pointer, seed uint64) uint64 {
			b := unsafe.Slice((*byte)(p), s) //nolint:gosec
			return HashBytesBlock(seed, b)
		}
	}
}

// buildStructHasher analyses a struct type and generates a HashFunction that
// hashes each field individually using the appropriate primitive hasher,
// then chains the results. Fields are accessed via precomputed byte offsets —
// no reflection at hash time.
func buildStructHasher(t reflect.Type) HashFunction {
	n := t.NumField()
	if n == 0 {
		// Empty struct — hash the size (0).
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	ops := make([]fieldHashOp, 0, n)
	for i := range n {
		f := t.Field(i)

		if f.Name == "_" {
			// Blank fields are ignored by Go equality — skip them.
			continue
		}

		h := buildHashFunction(f.Type)
		if h == nil {
			// Cannot hash this field type — give up.
			return nil
		}
		ops = append(ops, fieldHashOp{offset: f.Offset, hasher: h})
	}

	if len(ops) == 0 {
		// Struct with only blank fields.
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	// Optimised paths for small field counts to avoid slice overhead.
	switch len(ops) {
	case 1:
		op := ops[0]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			fp := unsafe.Add(p, op.offset) //nolint:gosec
			return op.hasher(fp, seed)
		}
	case 2:
		op0, op1 := ops[0], ops[1]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := op0.hasher(unsafe.Add(p, op0.offset), seed) //nolint:gosec
			return op1.hasher(unsafe.Add(p, op1.offset), h)  //nolint:gosec
		}
	case 3:
		op0, op1, op2 := ops[0], ops[1], ops[2]
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := op0.hasher(unsafe.Add(p, op0.offset), seed) //nolint:gosec
			h = op1.hasher(unsafe.Add(p, op1.offset), h)     //nolint:gosec
			return op2.hasher(unsafe.Add(p, op2.offset), h)  //nolint:gosec
		}
	default:
		// General N-field path.
		frozen := make([]fieldHashOp, len(ops))
		copy(frozen, ops)
		return func(p unsafe.Pointer, seed uint64) uint64 {
			h := seed
			for _, op := range frozen {
				h = op.hasher(unsafe.Add(p, op.offset), h) //nolint:gosec
			}
			return h
		}
	}
}

// buildArrayHasher generates a HashFunction for array types. If the element
// type is eligible for raw-byte-block hashing, the entire array is hashed as
// a byte block. Otherwise, each element is hashed individually.
func buildArrayHasher(t reflect.Type) HashFunction {
	elemType := t.Elem()
	arrLen := t.Len()

	if arrLen == 0 {
		return func(_ unsafe.Pointer, seed uint64) uint64 {
			return WH64Det(0, seed)
		}
	}

	// If the element type is eligible for raw-byte-block hashing,
	// hash the whole array as bytes.
	if CanUseUnsafeRawByteBlockHasherType(elemType).Eligible {
		totalSize := int(t.Size())
		return func(p unsafe.Pointer, seed uint64) uint64 {
			b := unsafe.Slice((*byte)(p), totalSize) //nolint:gosec
			return HashBytesBlock(seed, b)
		}
	}

	elemHasher := buildHashFunction(elemType)
	if elemHasher == nil {
		return nil
	}

	elemSize := elemType.Size()

	// Optimised for single-element arrays.
	if arrLen == 1 {
		return elemHasher
	}

	return func(p unsafe.Pointer, seed uint64) uint64 {
		h := seed
		for i := range arrLen {
			ep := unsafe.Add(p, uintptr(i)*elemSize) //nolint:gosec
			h = elemHasher(ep, h)
		}
		return h
	}
}

// hashComplex64 hashes a complex64 value by canonicalizing and hashing
// both float32 components and combining the results.
func hashComplex64(p unsafe.Pointer, seed uint64) uint64 {
	c := *(*complex64)(p)
	r := real(c)
	i := imag(c)
	rBits := canonicalF32Bits(r)
	iBits := canonicalF32Bits(i)
	h := WH32DetGR(rBits, seed)
	return WH32DetGR(iBits, h)
}

// hashComplex128 hashes a complex128 value by canonicalizing and hashing
// both float64 components and combining the results.
func hashComplex128(p unsafe.Pointer, seed uint64) uint64 {
	c := *(*complex128)(p)
	r := real(c)
	i := imag(c)
	rBits := canonicalF64Bits(r)
	iBits := canonicalF64Bits(i)
	h := WH64Det(rBits, seed)
	return WH64Det(iBits, h)
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
