package hashing

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// namedIface is a named interface type. Since Go 1.20 ordinary interface
// types satisfy the comparable constraint, so they can reach
// MakeRuntimeHasher through the public API.
type namedIface interface{ M() }

type ifaceImpl struct{ V int }

func (ifaceImpl) M() {}

// TestMakeRuntimeHasher_AnyKey verifies that an interface type parameter does
// not panic during hasher construction (reflect.TypeOf of a nil interface
// returns nil) and that hashing is deterministic per value.
func TestMakeRuntimeHasher_AnyKey(t *testing.T) {
	h := MakeRuntimeHasher[any](0x1234)

	require.Equal(t, h.Hash(42), h.Hash(42), "hashing must be deterministic")
	require.Equal(t, h.Hash("abc"), h.Hash("abc"), "hashing must be deterministic")
	require.NotEqual(t, h.Hash(42), h.Hash(43))

	// Known characteristic of the interface path: hash/maphash hashes numeric
	// values by numeric content, so int32(1) and int64(1) collide even though
	// they are not equal under ==. That is a collision, not a correctness
	// problem - Set3 still compares elements with == before reporting a match.
	require.Equal(t, h.Hash(int32(1)), h.Hash(int64(1)),
		"documents stdlib maphash behaviour; update if the stdlib changes")
}

// TestMakeRuntimeHasher_NamedInterfaceKey covers named interface types.
func TestMakeRuntimeHasher_NamedInterfaceKey(t *testing.T) {
	h := MakeRuntimeHasher[namedIface](0x1234)

	a := ifaceImpl{V: 1}
	b := ifaceImpl{V: 2}
	require.Equal(t, h.Hash(a), h.Hash(a))
	require.NotEqual(t, h.Hash(a), h.Hash(b))
}

// TestMakeRuntimeHasher_AnyKeySeeded verifies the seed is honoured on the
// interface path.
func TestMakeRuntimeHasher_AnyKeySeeded(t *testing.T) {
	h1 := MakeRuntimeHasher[any](0x1111)
	h2 := MakeRuntimeHasher[any](0x2222)
	require.NotEqual(t, h1.Hash(42), h2.Hash(42), "seed must influence the hash")
}

// TestMakeRuntimeHasher_AnyKeyNonComparableDynamicType documents that hashing
// a non-comparable dynamic type panics, exactly as == would.
func TestMakeRuntimeHasher_AnyKeyNonComparableDynamicType(t *testing.T) {
	h := MakeRuntimeHasher[any](0x1234)
	require.Panics(t, func() { _ = h.Hash([]int{1, 2, 3}) })
}
