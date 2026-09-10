package hashing

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// Named types over the primitive kinds. Dispatch must not depend on whether
// the type is the predeclared one or a named type with the same underlying
// type.
type (
	namedU8     uint8
	namedI16    int16
	namedU32    uint32
	namedU64    uint64
	namedInt    int
	namedF64    float64
	namedString string
	namedBool   bool
)

// TestNamedTypesGetSameHasherAsUnderlyingType verifies that a named type is
// hashed by the same specialized function as its underlying type. Before the
// switch to kind-based dispatch these fell through to the generic byte-block
// hasher, which was both slower and produced different values.
func TestNamedTypesGetSameHasherAsUnderlyingType(t *testing.T) {
	const seed = 0x1234

	require.Equal(t, MakeRuntimeHasher[uint8](seed).Hash(7), MakeRuntimeHasher[namedU8](seed).Hash(7))
	require.Equal(t, MakeRuntimeHasher[int16](seed).Hash(-7), MakeRuntimeHasher[namedI16](seed).Hash(-7))
	require.Equal(t, MakeRuntimeHasher[uint32](seed).Hash(7), MakeRuntimeHasher[namedU32](seed).Hash(7))
	require.Equal(t, MakeRuntimeHasher[uint64](seed).Hash(7), MakeRuntimeHasher[namedU64](seed).Hash(7))
	require.Equal(t, MakeRuntimeHasher[int](seed).Hash(7), MakeRuntimeHasher[namedInt](seed).Hash(7))
	require.Equal(t, MakeRuntimeHasher[float64](seed).Hash(7), MakeRuntimeHasher[namedF64](seed).Hash(7))
	require.Equal(t, MakeRuntimeHasher[string](seed).Hash("hi"), MakeRuntimeHasher[namedString](seed).Hash("hi"))
	require.Equal(t, MakeRuntimeHasher[bool](seed).Hash(true), MakeRuntimeHasher[namedBool](seed).Hash(true))
}

// TestNamedTypeHashingStaysCorrect sanity-checks distinctness and determinism
// on the named-type path.
func TestNamedTypeHashingStaysCorrect(t *testing.T) {
	h := MakeRuntimeHasher[namedU64](0x1234)
	require.Equal(t, h.Hash(1), h.Hash(1))
	require.NotEqual(t, h.Hash(1), h.Hash(2))
}

// TestPointerKindsAreHashed covers the pointer-shaped scalar kinds now routed
// to HashPtr.
func TestPointerKindsAreHashed(t *testing.T) {
	a, b := 1, 2
	hp := MakeRuntimeHasher[*int](0x1234)
	require.Equal(t, hp.Hash(&a), hp.Hash(&a))
	require.NotEqual(t, hp.Hash(&a), hp.Hash(&b))

	ch1, ch2 := make(chan int), make(chan int)
	hc := MakeRuntimeHasher[chan int](0x1234)
	require.Equal(t, hc.Hash(ch1), hc.Hash(ch1))
	require.NotEqual(t, hc.Hash(ch1), hc.Hash(ch2))
}

func BenchmarkNamedU64(b *testing.B) {
	h := MakeRuntimeHasher[namedU64](0x1234)
	var s uint64
	for i := 0; b.Loop(); i++ {
		s += h.Hash(namedU64(i))
	}
	benchSink = s
}

func BenchmarkPlainU64(b *testing.B) {
	h := MakeRuntimeHasher[uint64](0x1234)
	var s uint64
	for i := 0; b.Loop(); i++ {
		s += h.Hash(uint64(i))
	}
	benchSink = s
}
