//go:build set3lab

package setbench

// nativeMap wraps map[T]struct{} so that the benchmarks in this package can
// drive it through the same shape of call as Set3.
//
// The wrappers used to carry //go:noinline. That made the comparison unfair in
// the map's disfavour: Set3's own Add and Contains are far past the inliner's
// budget (454 and 232 against 80) and could never be inlined either, so
// forbidding it on one side only added a call frame the other side pays for
// anyway. What a caller actually gets is one inlinable wrapper against one
// method that is not, and that is what these now measure.
//
// The directives were not there to stop the map work being optimized away; the
// benchmarks accumulate every result into a sink, which is what prevents that.
// If a future change makes a benchmark's map operations vanish, the fix is a
// sink, not a noinline.
type nativeMap[T comparable] map[T]struct{}

func emptyNativeWithCapacity[T comparable](size uint32) *nativeMap[T] {
	result := make(nativeMap[T], size)
	return &result
}

func (thisSet *nativeMap[T]) add(val T) {
	(*thisSet)[val] = struct{}{}
}

func (thisSet *nativeMap[T]) contains(val T) bool {
	_, b := (*thisSet)[val]
	return b
}

func (thisSet *nativeMap[T]) count() uint32 {
	return uint32(len(*thisSet)) //nolint:gosec
}
