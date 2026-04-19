package hashing

import "reflect"

// RawByteBlockEligibility describes whether a type can be hashed by reading its
// in-memory bytes as one contiguous block.
//
// "Eligible" means all bytes that would be read by an unsafe byte-block hash are
// semantically relevant for == equality of values of that type.
//
// "Reason" is a short explanation that is useful for debugging and tests.
//
// The analysis is intentionally conservative. Returning false does not imply a
// type cannot be hashed efficiently; it only means a blind raw-memory byte hash
// is not guaranteed to follow Go equality semantics for that type.
type RawByteBlockEligibility struct {
	Eligible bool
	Reason   string
}

// CanUseUnsafeRawByteBlockHasher reports whether K can be hashed by reading the
// complete in-memory representation as a raw byte block.
//
// This helper is intended for MakeRuntimeHasher-like dispatch decisions where we
// want a very fast unsafe memory hasher for types that allow it.
//
// Semantics and safety model:
//  1. Go equality must be preserved: if a == b then byte-block hash input must be
//     identical for a and b.
//  2. Types with value-level normalization requirements (notably float32/float64)
//     are rejected, because +0 and -0 compare equal and NaN bit patterns require
//     canonicalization in this project.
//  3. Structs are accepted only if there is no padding and no blank identifier
//     fields. Padding bytes may contain non-semantic data and blank fields are
//     ignored by ==, so both would break correctness for raw byte hashing.
//  4. Arrays are accepted iff their element type is accepted.
//  5. Slices, maps, strings, interfaces, and function values are rejected because
//     their in-memory representation does not match value equality semantics.
//
// The result is conservative by design to avoid subtle semantic regressions.
func CanUseUnsafeRawByteBlockHasher[K comparable]() RawByteBlockEligibility {
	var zero K
	return CanUseUnsafeRawByteBlockHasherType(reflect.TypeOf(zero))
}

// CanUseUnsafeRawByteBlockHasherType recursively inspects t and decides whether a
// raw-memory byte-block hash is semantics-preserving for values of t.
func CanUseUnsafeRawByteBlockHasherType(t reflect.Type) RawByteBlockEligibility {
	if t == nil {
		return RawByteBlockEligibility{Eligible: false, Reason: "nil type"}
	}

	switch t.Kind() {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Uintptr,
		reflect.Pointer,
		reflect.UnsafePointer,
		reflect.Chan:
		return RawByteBlockEligibility{Eligible: true, Reason: "scalar representation is fully semantic"}

	case reflect.Float32, reflect.Float64:
		return RawByteBlockEligibility{Eligible: false, Reason: "float requires canonicalization (+0/-0 and NaN handling)"}

	case reflect.Complex64, reflect.Complex128:
		return RawByteBlockEligibility{Eligible: false, Reason: "complex contains float components requiring canonicalization"}

	case reflect.Array:
		elem := CanUseUnsafeRawByteBlockHasherType(t.Elem())
		if !elem.Eligible {
			return RawByteBlockEligibility{Eligible: false, Reason: "array element not eligible: " + elem.Reason}
		}
		return RawByteBlockEligibility{Eligible: true, Reason: "array element type is eligible"}

	case reflect.Struct:
		// Ensure there is no inter-field or trailing padding and no blank fields.
		var expectedOffset uintptr
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)

			if f.Name == "_" {
				return RawByteBlockEligibility{Eligible: false, Reason: "struct has blank identifier field ignored by equality"}
			}

			if f.Offset != expectedOffset {
				return RawByteBlockEligibility{Eligible: false, Reason: "struct has padding bytes"}
			}

			child := CanUseUnsafeRawByteBlockHasherType(f.Type)
			if !child.Eligible {
				return RawByteBlockEligibility{Eligible: false, Reason: "struct field " + f.Name + " not eligible: " + child.Reason}
			}

			expectedOffset = f.Offset + f.Type.Size()
		}

		if expectedOffset != t.Size() {
			return RawByteBlockEligibility{Eligible: false, Reason: "struct has trailing padding bytes"}
		}

		return RawByteBlockEligibility{Eligible: true, Reason: "struct fields are eligible and layout has no padding"}

	case reflect.String:
		return RawByteBlockEligibility{Eligible: false, Reason: "string equality is content-based, not header-bytes based"}

	case reflect.Slice:
		return RawByteBlockEligibility{Eligible: false, Reason: "slice is not comparable and header bytes are not value semantics"}

	case reflect.Map:
		return RawByteBlockEligibility{Eligible: false, Reason: "map is not comparable"}

	case reflect.Interface:
		return RawByteBlockEligibility{Eligible: false, Reason: "interface equality is dynamic value semantics, not header bytes"}

	case reflect.Func:
		return RawByteBlockEligibility{Eligible: false, Reason: "func values are not comparable except nil"}

	default:
		return RawByteBlockEligibility{Eligible: false, Reason: "unsupported kind for raw byte-block hashing"}
	}
}
