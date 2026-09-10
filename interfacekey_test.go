package set3

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestSet3_InterfaceElementType verifies that Set3 can be instantiated with an
// interface element type. Since Go 1.20 ordinary interface types satisfy the
// comparable constraint, so this is reachable through the public API.
func TestSet3_InterfaceElementType(t *testing.T) {
	s := Empty[any]()
	s.Add(42)
	s.Add("hello")
	s.Add(42)

	require.Equal(t, 2, int(s.Size()))
	require.True(t, s.Contains(42))
	require.True(t, s.Contains("hello"))
	require.False(t, s.Contains(43))
}

// TestSet3_InterfaceElementTypeWithCapacity covers the other constructor.
func TestSet3_InterfaceElementTypeWithCapacity(t *testing.T) {
	s := EmptyWithCapacity[any](64)
	for i := range 100 {
		s.Add(i)
	}
	require.Equal(t, 100, int(s.Size()))
	require.True(t, s.Contains(99))
	require.False(t, s.Contains(100))
}
