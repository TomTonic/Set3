//go:build set3lab

package setbench

// this type is for benchmark comparison only
type nativeMap[T comparable] map[T]struct{}

//go:noinline
func emptyNativeWithCapacity[T comparable](size uint32) *nativeMap[T] {
	result := make(nativeMap[T], size)
	return &result
}

//go:noinline
func (thisSet *nativeMap[T]) add(val T) {
	(*thisSet)[val] = struct{}{}
}

//go:noinline
func (thisSet *nativeMap[T]) contains(val T) bool {
	_, b := (*thisSet)[val]
	return b
}

//go:noinline
func (thisSet *nativeMap[T]) count() uint32 {
	return uint32(len(*thisSet)) //nolint:gosec
}
