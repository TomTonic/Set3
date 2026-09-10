package hashing

import (
	"math"
	"reflect"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// The tests in this file work on the builder pipeline inside hashgen.go rather
// than on the hash functions it produces: flattening a type into micro-ops,
// coalescing adjacent ops, choosing a hash routine per op, and emitting the
// final closure. hashgen_test.go covers the same machinery from the outside, by
// the hashes it produces for real Go types. Both views are needed. The outside
// view cannot reach the branches that only defend against inputs no type can
// currently produce, and the inside view cannot tell whether the pipeline as a
// whole still hashes a struct correctly.

// --- Types that steer the builder down a specific path -----------------------

type bldMapField struct {
	A uint64
	M map[string]int
}

type bldSliceField struct {
	A uint64
	S []byte
}

type bldFuncField struct {
	A uint64
	F func()
}

type bldZeroArrayField struct {
	A uint64
	Z [0]uint32
	B uint64
}

type bldEligibleArrayField struct {
	A uint64
	B [4]uint32
}

type bldInterfaceArrayField struct {
	A uint64
	B [2]any
}

type bldAdjacentBlocks struct {
	A uint32
	B uint32
	C uint32
	D uint32
}

type bldSingleOp struct {
	F float64
}

// bldNineOps has nine fields that cannot be merged into byte blocks, which is
// one more than the largest unrolled closure buildClosureFromOps emits.
type bldNineOps struct {
	A, B, C, D, E, F, G, H, I float64
}

// bldBlankOnly has a field, so it is not an empty struct, but that field is
// blank and therefore invisible to ==.
type bldBlankOnly struct {
	_ uint64
}

type bldFloat32Field struct {
	F float32
}

type bldComplex64Field struct {
	C complex64
}

type bldComplex128Field struct {
	C complex128
}

// --- Helpers -----------------------------------------------------------------

// hashOf runs a generated hash function over the value v.
func hashOf[T any](fn HashFunction, v T, seed uint64) uint64 {
	return fn(unsafe.Pointer(&v), seed)
}

// mustGenerate fails the test if the generator cannot handle the sample's type.
func mustGenerate(t *testing.T, sample any) HashFunction {
	t.Helper()
	fn := GenerateHashFunction(reflect.TypeOf(sample))
	require.NotNil(t, fn, "GenerateHashFunction returned nil for %T", sample)
	return fn
}

// --- Rejection: types the generator hands back to the maphash fallback -------

// TestGenerateHashFunctionRejectsTypesItCannotHash verifies that a set built on
// an element type the generator does not understand still works, by returning
// nil so that MakeRuntimeHasher falls back to Go's own maphash.
//
// The generator reads a value's memory directly. For a field whose in-memory
// form is a pointer to somewhere else -- a map, a slice, a func, an interface --
// that memory says nothing about equality, so producing a hash from it would
// put equal values in different buckets. Refusing is the correct answer.
//
// It offers a type per rejection route through the flattener -- a bare
// unsupported kind, one nested in a struct, one nested in an array, and a nil
// type -- and requires nil back for each.
func TestGenerateHashFunctionRejectsTypesItCannotHash(t *testing.T) {
	var iface any

	cases := []struct {
		name string
		typ  reflect.Type
	}{
		{"nil type", nil},
		{"interface", reflect.TypeOf(&iface).Elem()},
		{"map", reflect.TypeOf(map[string]int(nil))},
		{"slice", reflect.TypeOf([]byte(nil))},
		{"func", reflect.TypeOf(func() {})},
		{"struct with a map field", reflect.TypeOf(bldMapField{})},
		{"struct with a slice field", reflect.TypeOf(bldSliceField{})},
		{"struct with a func field", reflect.TypeOf(bldFuncField{})},
		{"struct with an array of interfaces", reflect.TypeOf(bldInterfaceArrayField{})},
		{"array of interfaces", reflect.TypeOf([2]any{})},
		{"array of maps", reflect.TypeOf([2]map[string]int{})},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Nil(t, GenerateHashFunction(c.typ),
				"generator must decline this type so the caller falls back to maphash")
		})
	}
}

// TestGenerateHashFunctionCachesItsRefusals verifies that asking about the same
// unsupported type twice is as cheap as asking once.
//
// Every Set3 created for a given element type calls into the generator. Type
// analysis walks the whole layout with reflection, so repeating it for a type
// already known to be unsupported would make set construction needlessly slow.
//
// It asks twice for a type the generator refuses and requires nil both times,
// which is only correct if the cached refusal is recognised as a refusal.
func TestGenerateHashFunctionCachesItsRefusals(t *testing.T) {
	typ := reflect.TypeOf(bldMapField{})

	require.Nil(t, GenerateHashFunction(typ), "first call")
	require.Nil(t, GenerateHashFunction(typ), "second call, served from the cache")

	cached, ok := generatedHashCache.Load(typ)
	require.True(t, ok, "the refusal must be remembered, not recomputed")

	// The refusal is stored as a nil value of type HashFunction, never as a bare
	// nil, which is what lets GenerateHashFunction assert the type unconditionally
	// on the cache-hit path.
	fn, isHashFunction := cached.(HashFunction)
	require.True(t, isHashFunction, "cache entry is not a HashFunction: %T", cached)
	require.Nil(t, fn)
}

// --- buildScalarBytesHasher --------------------------------------------------

// TestBuildScalarBytesHasherReadsExactlyTheRequestedWidth verifies that hashing
// a scalar reads that scalar and nothing around it.
//
// Scalars are dispatched by width, and each width gets a routine that reads a
// fixed number of bytes through an unsafe pointer. Reading one byte too many
// would make a key's hash depend on whatever happens to sit next to it in
// memory, which for a struct field means the hash changes when an unrelated
// field does.
//
// It lays each width out in a buffer followed by a guard byte, hashes it,
// changes only the guard, and requires the hash to be unchanged -- then changes
// a byte inside the value and requires the hash to move.
func TestBuildScalarBytesHasherReadsExactlyTheRequestedWidth(t *testing.T) {
	for _, size := range []uintptr{1, 2, 4, 8, 3, 5, 16} {
		t.Run(strconvUintptr(size)+" bytes", func(t *testing.T) {
			fn := buildScalarBytesHasher(size)
			require.NotNil(t, fn)

			buf := make([]byte, size+8)
			for i := range buf {
				buf[i] = byte(0xA0 + i)
			}
			p := unsafe.Pointer(&buf[0])
			const seed = uint64(0x5EED)

			base := fn(p, seed)
			require.Equal(t, base, fn(p, seed), "hash must be deterministic")

			buf[size] ^= 0xFF // first byte past the value
			require.Equal(t, base, fn(p, seed),
				"hash changed after touching a byte outside the %d-byte value", size)

			buf[size-1] ^= 0xFF // last byte inside the value
			require.NotEqual(t, base, fn(p, seed),
				"hash did not change after touching the last byte of the value")
		})
	}
}

// TestBuildScalarBytesHasherRespondsToTheSeed verifies that re-seeding a set
// actually changes where its elements land.
//
// Set3 draws a fresh seed whenever it rehashes, which is how it breaks up a
// collision pattern that forced the rehash in the first place. A width whose
// hash ignored the seed would keep colliding forever.
//
// It hashes the same bytes under two seeds for every scalar width and requires
// the results to differ.
func TestBuildScalarBytesHasherRespondsToTheSeed(t *testing.T) {
	for _, size := range []uintptr{1, 2, 4, 8, 3, 16} {
		fn := buildScalarBytesHasher(size)
		buf := make([]byte, size)
		for i := range buf {
			buf[i] = byte(i + 1)
		}
		p := unsafe.Pointer(&buf[0])
		require.NotEqual(t, fn(p, 1), fn(p, 2), "width %d ignores the seed", size)
	}
}

// strconvUintptr renders a uintptr for subtest names without pulling strconv
// into the file for a single call.
func strconvUintptr(v uintptr) string {
	if v == 0 {
		return "0"
	}
	var digits []byte
	for v > 0 {
		digits = append([]byte{byte('0' + v%10)}, digits...)
		v /= 10
	}
	return string(digits)
}

// --- flattenTypeOps / flattenArrayOps ----------------------------------------

// TestFlattenTypeOpsProducesOneOpPerLeafValue verifies that the flattener
// describes a type as the list of primitive reads needed to hash it.
//
// Everything downstream -- block merging, routine selection, closure emission --
// works from that list, so an op with a wrong kind or offset produces a hash
// function that reads the wrong memory.
//
// It flattens one type per supported kind at a known offset and requires the
// expected op kind, offset and, for byte blocks, width.
func TestFlattenTypeOpsProducesOneOpPerLeafValue(t *testing.T) {
	const off = uintptr(24)

	cases := []struct {
		name string
		typ  reflect.Type
		want []microOp
	}{
		{"float32", reflect.TypeOf(float32(0)), []microOp{{kind: opFloat32, offset: off}}},
		{"float64", reflect.TypeOf(float64(0)), []microOp{{kind: opFloat64, offset: off}}},
		{"complex64", reflect.TypeOf(complex64(0)), []microOp{{kind: opComplex64, offset: off}}},
		{"complex128", reflect.TypeOf(complex128(0)), []microOp{{kind: opComplex128, offset: off}}},
		{"string", reflect.TypeOf(""), []microOp{{kind: opString, offset: off}}},
		{"bool", reflect.TypeOf(false), []microOp{{kind: opByteBlock, offset: off, size: 1}}},
		{"uint16", reflect.TypeOf(uint16(0)), []microOp{{kind: opByteBlock, offset: off, size: 2}}},
		{"uint64", reflect.TypeOf(uint64(0)), []microOp{{kind: opByteBlock, offset: off, size: 8}}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, flattenTypeOps(c.typ, off))
		})
	}
}

// TestFlattenTypeOpsRejectsKindsWithoutAValueRepresentation verifies that the
// flattener refuses a leaf whose bytes do not describe its value, rather than
// silently hashing a pointer.
//
// This is the single point where an unsupported field anywhere in a nested
// struct or array turns the whole type into a maphash fallback, so it has to
// return nil rather than an empty op list -- an empty list would be read as
// "nothing to hash" and produce a constant hash for every value.
//
// It flattens each unsupported kind and requires a nil op list, explicitly
// distinguishing nil from empty.
func TestFlattenTypeOpsRejectsKindsWithoutAValueRepresentation(t *testing.T) {
	var iface any

	for name, typ := range map[string]reflect.Type{
		"map":       reflect.TypeOf(map[int]int(nil)),
		"slice":     reflect.TypeOf([]int(nil)),
		"func":      reflect.TypeOf(func() {}),
		"interface": reflect.TypeOf(&iface).Elem(),
	} {
		t.Run(name, func(t *testing.T) {
			ops := flattenTypeOps(typ, 0)
			require.Nil(t, ops, "must be nil, not an empty list: an empty list means "+
				"'hash nothing' and would give every value the same hash")
		})
	}
}

// TestFlattenArrayOpsCollapsesEligibleArraysIntoOneBlock verifies that an array
// of plain scalars costs one hash call rather than one per element.
//
// An array whose element type has no padding and needs no canonicalization is
// contiguous, semantically relevant memory, so it can be read in a single pass.
// For a [4]uint32 field that is one call instead of four.
//
// It flattens arrays of several shapes and requires a single byte-block op
// spanning the whole array for the eligible ones, an empty list for a
// zero-length array, per-element ops where the elements need canonicalization,
// and nil where an element is unsupported.
func TestFlattenArrayOpsCollapsesEligibleArraysIntoOneBlock(t *testing.T) {
	t.Run("eligible elements collapse to one block", func(t *testing.T) {
		ops := flattenArrayOps(reflect.TypeOf([4]uint32{}), 8)
		require.Equal(t, []microOp{{kind: opByteBlock, offset: 8, size: 16}}, ops)
	})

	t.Run("zero length yields no work but is not a refusal", func(t *testing.T) {
		ops := flattenArrayOps(reflect.TypeOf([0]uint32{}), 8)
		require.NotNil(t, ops, "an empty array is hashable, it just contributes nothing")
		require.Empty(t, ops)
	})

	t.Run("elements needing canonicalization stay separate", func(t *testing.T) {
		ops := flattenArrayOps(reflect.TypeOf([3]float64{}), 0)
		require.Equal(t, []microOp{
			{kind: opFloat64, offset: 0},
			{kind: opFloat64, offset: 8},
			{kind: opFloat64, offset: 16},
		}, ops)
	})

	t.Run("an unsupported element refuses the whole array", func(t *testing.T) {
		require.Nil(t, flattenArrayOps(reflect.TypeOf([2]any{}), 0))
	})

	t.Run("elements that contribute nothing are not a refusal", func(t *testing.T) {
		ops := flattenArrayOps(reflect.TypeOf([3]bldBlankOnly{}), 0)
		require.NotNil(t, ops, "an array of blank-only structs is hashable as a constant; "+
			"returning nil here would send it to the maphash fallback instead")
		require.Empty(t, ops)
	})
}

// --- mergeByteBlocks ---------------------------------------------------------

// TestMergeByteBlocksCoalescesOnlyTrulyAdjacentBlocks verifies the optimization
// that turns several small reads of a struct into one large one.
//
// Merging is what makes a struct of eight uint32 fields cost one 32-byte hash
// instead of eight 4-byte hashes. It is only correct for blocks that are
// physically adjacent: merging across a padding gap would pull padding bytes,
// which are not part of a value's identity, into the hash.
//
// It feeds op lists that are adjacent, separated by a gap, and interrupted by an
// op of another kind, and requires merging in the first case only.
func TestMergeByteBlocksCoalescesOnlyTrulyAdjacentBlocks(t *testing.T) {
	cases := []struct {
		name string
		in   []microOp
		want []microOp
	}{
		{
			name: "empty input is returned untouched",
			in:   []microOp{},
			want: []microOp{},
		},
		{
			name: "a single op has nothing to merge with",
			in:   []microOp{{kind: opByteBlock, offset: 0, size: 4}},
			want: []microOp{{kind: opByteBlock, offset: 0, size: 4}},
		},
		{
			name: "adjacent blocks become one",
			in: []microOp{
				{kind: opByteBlock, offset: 0, size: 4},
				{kind: opByteBlock, offset: 4, size: 4},
				{kind: opByteBlock, offset: 8, size: 8},
			},
			want: []microOp{{kind: opByteBlock, offset: 0, size: 16}},
		},
		{
			name: "a padding gap prevents merging",
			in: []microOp{
				{kind: opByteBlock, offset: 0, size: 4},
				{kind: opByteBlock, offset: 8, size: 4},
			},
			want: []microOp{
				{kind: opByteBlock, offset: 0, size: 4},
				{kind: opByteBlock, offset: 8, size: 4},
			},
		},
		{
			name: "a float in between interrupts the run",
			in: []microOp{
				{kind: opByteBlock, offset: 0, size: 4},
				{kind: opFloat32, offset: 4},
				{kind: opByteBlock, offset: 8, size: 4},
				{kind: opByteBlock, offset: 12, size: 4},
			},
			want: []microOp{
				{kind: opByteBlock, offset: 0, size: 4},
				{kind: opFloat32, offset: 4},
				{kind: opByteBlock, offset: 8, size: 8},
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			require.Equal(t, c.want, mergeByteBlocks(c.in))
		})
	}
}

// TestMergeByteBlocksNeverChangesTheBytesCovered verifies the invariant that
// makes merging safe to apply at all.
//
// Merging rewrites the plan for reading a value. Whatever it produces has to
// describe the same bytes as the plan it replaces, or the generated hash
// function reads memory the value does not own.
//
// It merges the op lists of several real struct layouts and requires the set of
// covered byte offsets to be identical before and after.
func TestMergeByteBlocksNeverChangesTheBytesCovered(t *testing.T) {
	samples := []any{
		bldAdjacentBlocks{}, genPaddingStruct{}, genMixed{}, genNestedStruct{},
		bldEligibleArrayField{}, bldSingleOp{}, bldNineOps{},
	}

	covered := func(ops []microOp) map[uintptr]bool {
		m := make(map[uintptr]bool)
		for _, op := range ops {
			if op.kind != opByteBlock {
				m[op.offset] = true
				continue
			}
			for i := range uintptr(op.size) {
				m[op.offset+i] = true
			}
		}
		return m
	}

	for _, s := range samples {
		ops := flattenStructOps(reflect.TypeOf(s), 0)
		require.NotNil(t, ops, "%T should be flattenable", s)
		require.Equal(t, covered(ops), covered(mergeByteBlocks(ops)),
			"merging changed which bytes of %T are hashed", s)
	}
}

// --- microOpToFieldOp --------------------------------------------------------

// TestMicroOpToFieldOpGivesEveryOpKindAWorkingHasher verifies that each step in
// a hash plan is turned into a routine that actually reads that step's data.
//
// This is where a plan becomes executable code. A kind mapped to the wrong
// routine would read the right offset with the wrong interpretation -- a float
// hashed as raw bytes, for instance, which would put +0 and -0 in different
// buckets even though Go considers them equal.
//
// It converts one op of every kind, runs the result over real memory, and
// requires a hash that both depends on the data and responds to the seed.
func TestMicroOpToFieldOpGivesEveryOpKindAWorkingHasher(t *testing.T) {
	kinds := []struct {
		name string
		op   microOp
	}{
		{"byte block, 1 byte", microOp{kind: opByteBlock, size: 1}},
		{"byte block, 2 bytes", microOp{kind: opByteBlock, size: 2}},
		{"byte block, 4 bytes", microOp{kind: opByteBlock, size: 4}},
		{"byte block, 8 bytes", microOp{kind: opByteBlock, size: 8}},
		{"byte block, 16 bytes (specialized)", microOp{kind: opByteBlock, size: 16}},
		{"byte block, 40 bytes (generic)", microOp{kind: opByteBlock, size: 40}},
		{"float32", microOp{kind: opFloat32}},
		{"float64", microOp{kind: opFloat64}},
		{"complex64", microOp{kind: opComplex64}},
		{"complex128", microOp{kind: opComplex128}},
	}

	for _, k := range kinds {
		t.Run(k.name, func(t *testing.T) {
			fop := microOpToFieldOp(k.op)
			require.NotNil(t, fop.fn)

			buf := make([]byte, 64)
			for i := range buf {
				buf[i] = byte(i*7 + 1)
			}
			p := unsafe.Pointer(&buf[0])

			h := fop.fn(p, 0x1234)
			require.Equal(t, h, fop.fn(p, 0x1234), "must be deterministic")
			require.NotEqual(t, h, fop.fn(p, 0x4321), "must depend on the seed")

			buf[0] ^= 0xFF
			require.NotEqual(t, h, fop.fn(p, 0x1234), "must depend on the data")
		})
	}

	t.Run("string", func(t *testing.T) {
		fop := microOpToFieldOp(microOp{kind: opString})
		s := "hashed by content, not by header"
		other := "a different string entirely......"
		require.NotEqual(t,
			fop.fn(unsafe.Pointer(&s), 9),
			fop.fn(unsafe.Pointer(&other), 9))
	})
}

// TestMicroOpToFieldOpPreservesTheOffset verifies that a step's position in the
// value is carried through to the routine that will execute it.
//
// The offset is applied by the emitted closure, not by the routine itself, so a
// lost offset would not crash: every field would simply be read from the start
// of the struct, and structs differing only in a later field would collide.
//
// It converts ops at several offsets and requires the offset to survive.
func TestMicroOpToFieldOpPreservesTheOffset(t *testing.T) {
	for _, off := range []uintptr{0, 1, 8, 4096} {
		require.Equal(t, off, microOpToFieldOp(microOp{kind: opByteBlock, offset: off, size: 4}).offset)
		require.Equal(t, off, microOpToFieldOp(microOp{kind: opFloat64, offset: off}).offset)
	}
}

// TestMicroOpToFieldOpPassesTheSeedThroughForAnUnknownKind verifies the
// behaviour of the defensive branch that catches an op kind nobody has defined.
//
// No current type can produce such an op; the branch exists so that adding an
// opKind without extending this switch degrades into "contributes nothing to the
// hash" rather than a nil function pointer and a crash inside a set operation.
//
// It converts an op with an out-of-range kind and requires a routine that
// returns the incoming seed unchanged, for any data.
func TestMicroOpToFieldOpPassesTheSeedThroughForAnUnknownKind(t *testing.T) {
	fop := microOpToFieldOp(microOp{kind: opKind(200), offset: 16})
	require.NotNil(t, fop.fn, "an unknown kind must still yield a callable routine")
	require.Equal(t, uintptr(16), fop.offset)

	buf := make([]byte, 32)
	for _, seed := range []uint64{0, 1, 0xFFFF_FFFF_FFFF_FFFF} {
		require.Equal(t, seed, fop.fn(unsafe.Pointer(&buf[0]), seed),
			"an unknown op must leave the running hash untouched")
	}
}

// --- buildClosureFromOps -----------------------------------------------------

// TestBuildClosureFromOpsHandlesAnEmptyPlan verifies that a plan with no steps
// still yields a usable hash function.
//
// Both callers check for an empty plan before reaching this branch, so it is
// unreachable through any Go type today. It is kept, and tested, because the
// alternative to a constant hasher here is returning nil, and a nil
// HashFunction is not distinguishable from "unsupported type" any more.
//
// It builds a closure from no ops and requires a deterministic, seed-dependent
// hash that does not read the value at all.
func TestBuildClosureFromOpsHandlesAnEmptyPlan(t *testing.T) {
	fn := buildClosureFromOps(nil)
	require.NotNil(t, fn)

	buf := make([]byte, 16)
	p := unsafe.Pointer(&buf[0])

	h := fn(p, 0x99)
	require.Equal(t, h, fn(p, 0x99))
	require.NotEqual(t, h, fn(p, 0x9A), "an empty plan still has to mix the seed")

	buf[0] = 0xFF
	require.Equal(t, h, fn(p, 0x99), "an empty plan must not read the value")
}

// TestBuildClosureFromOpsAgreesWithTheLoopForEveryPlanSize verifies that the
// unrolled closures emitted for small plans compute exactly what the general
// loop computes.
//
// buildClosureFromOps emits nine different shapes: a dedicated closure for each
// plan size from one to eight ops, so the inner calls can be inlined, and a loop
// for anything larger. Nine implementations of one definition is nine chances
// for them to disagree, and a disagreement would mean the hash of a type depends
// on how many fields it happens to have.
//
// It builds plans of one through twelve identical ops and requires the emitted
// closure to match a straightforward reference fold over the same ops.
func TestBuildClosureFromOpsAgreesWithTheLoopForEveryPlanSize(t *testing.T) {
	buf := make([]byte, 256)
	for i := range buf {
		buf[i] = byte(i*13 + 5)
	}
	p := unsafe.Pointer(&buf[0])
	const seed = uint64(0xC0FFEE)

	for n := 1; n <= 12; n++ {
		ops := make([]microOp, n)
		for i := range ops {
			ops[i] = microOp{kind: opByteBlock, offset: uintptr(i * 8), size: 8}
		}

		want := seed
		for _, op := range ops {
			fop := microOpToFieldOp(op)
			want = fop.fn(unsafe.Add(p, fop.offset), want)
		}

		got := buildClosureFromOps(ops)(p, seed)
		require.Equal(t, want, got, "the closure for %d ops disagrees with the reference fold", n)
	}
}

// --- Whole-type behaviour that reaches the remaining builder branches --------

// TestGenerateHashFunctionForStructsWithMoreThanEightFields verifies that a
// struct wide enough to leave the unrolled closures behind is still hashed
// correctly.
//
// Plans of up to eight steps get a dedicated closure; anything larger falls back
// to a loop over a frozen step list. A struct with nine independent float fields
// is the smallest thing that gets there.
//
// It hashes such a struct and requires every field to matter: changing any one
// of the nine must change the hash.
func TestGenerateHashFunctionForStructsWithMoreThanEightFields(t *testing.T) {
	fn := mustGenerate(t, bldNineOps{})

	base := bldNineOps{A: 1, B: 2, C: 3, D: 4, E: 5, F: 6, G: 7, H: 8, I: 9}
	h := hashOf(fn, base, 0x1234)

	fields := []func(*bldNineOps){
		func(v *bldNineOps) { v.A = 100 }, func(v *bldNineOps) { v.B = 100 },
		func(v *bldNineOps) { v.C = 100 }, func(v *bldNineOps) { v.D = 100 },
		func(v *bldNineOps) { v.E = 100 }, func(v *bldNineOps) { v.F = 100 },
		func(v *bldNineOps) { v.G = 100 }, func(v *bldNineOps) { v.H = 100 },
		func(v *bldNineOps) { v.I = 100 },
	}
	for i, mutate := range fields {
		v := base
		mutate(&v)
		require.NotEqual(t, h, hashOf(fn, v, 0x1234), "field %d does not affect the hash", i)
	}
}

// TestGenerateHashFunctionForSingleFieldStructs verifies that a struct holding
// one value hashes as that value does.
//
// The one-op plan gets its own closure, the shortest of the nine shapes. It is
// also the shape a great many real key types reduce to after block merging.
//
// It hashes a single-float struct and requires agreement with hashing the bare
// float at the same seed.
func TestGenerateHashFunctionForSingleFieldStructs(t *testing.T) {
	fn := mustGenerate(t, bldSingleOp{})

	for _, f := range []float64{0, 1, -1, math.Pi, math.Inf(1)} {
		v := bldSingleOp{F: f}
		want := HashF64WHdet(unsafe.Pointer(&f), 0x77)
		require.Equal(t, want, hashOf(fn, v, 0x77), "single-field struct disagrees with the bare field for %v", f)
	}
}

// TestGenerateHashFunctionIgnoresBlankAndZeroLengthFields verifies that fields
// which cannot hold a value do not influence the hash.
//
// A blank field and a zero-length array occupy layout but are invisible to ==,
// so two values Go considers equal can differ in exactly those bytes. Letting
// them reach the hash would put equal values in different buckets.
//
// It hashes structs whose only content is blank or zero-length, with the
// underlying memory deliberately dirtied, and requires the hash to be unmoved.
func TestGenerateHashFunctionIgnoresBlankAndZeroLengthFields(t *testing.T) {
	t.Run("a struct of only blank fields hashes as a constant", func(t *testing.T) {
		fn := mustGenerate(t, genAllBlank{})

		var buf [64]byte
		p := unsafe.Pointer(&buf[0])
		h := fn(p, 0x31)
		for i := range buf {
			buf[i] = 0xFF
		}
		require.Equal(t, h, fn(p, 0x31), "blank field bytes leaked into the hash")
		require.NotEqual(t, h, fn(p, 0x32), "the seed must still be mixed in")
	})

	t.Run("an array of blank-only structs hashes as a constant", func(t *testing.T) {
		fn := mustGenerate(t, [3]bldBlankOnly{})

		var buf [64]byte
		p := unsafe.Pointer(&buf[0])
		h := fn(p, 0x41)
		for i := range buf {
			buf[i] = 0xFF
		}
		require.Equal(t, h, fn(p, 0x41), "blank field bytes leaked into the hash")
	})

	t.Run("a zero-length array field contributes nothing", func(t *testing.T) {
		fn := mustGenerate(t, bldZeroArrayField{})
		a := bldZeroArrayField{A: 7, B: 9}
		b := bldZeroArrayField{A: 7, B: 9}
		require.Equal(t, hashOf(fn, a, 5), hashOf(fn, b, 5))
		require.NotEqual(t, hashOf(fn, a, 5), hashOf(fn, bldZeroArrayField{A: 7, B: 10}, 5))
	})
}

// TestGenerateHashFunctionForArrays verifies that arrays are hashed as whole
// values through each of the routes the builder can take for them.
//
// An array of plain scalars is read in one pass, and there are two ways to do
// that: a straight-line routine for the handful of sizes that have one, and a
// general block hasher for everything else. An array needing canonicalization is
// read element by element instead. All three must agree that two equal arrays
// hash equally and two different ones usually do not.
//
// It hashes arrays sized to hit each route and requires element sensitivity at
// every position.
func TestGenerateHashFunctionForArrays(t *testing.T) {
	t.Run("empty array", func(t *testing.T) {
		fn := mustGenerate(t, [0]int{})
		v := [0]int{}
		require.Equal(t, hashOf(fn, v, 3), hashOf(fn, v, 3))
		require.NotEqual(t, hashOf(fn, v, 3), hashOf(fn, v, 4))
	})

	t.Run("16 bytes, straight-line routine", func(t *testing.T) {
		fn := mustGenerate(t, [4]uint32{})
		requireEveryElementMatters(t, fn, [4]uint32{1, 2, 3, 4})
	})

	t.Run("40 bytes, general block hasher", func(t *testing.T) {
		fn := mustGenerate(t, [5]uint64{})
		requireEveryElementMatters(t, fn, [5]uint64{1, 2, 3, 4, 5})
	})

	t.Run("array field inside a struct collapses to one block", func(t *testing.T) {
		fn := mustGenerate(t, bldEligibleArrayField{})
		base := bldEligibleArrayField{A: 1, B: [4]uint32{2, 3, 4, 5}}
		h := hashOf(fn, base, 0x11)
		for i := range base.B {
			v := base
			v.B[i] = 99
			require.NotEqual(t, h, hashOf(fn, v, 0x11), "array element %d does not affect the hash", i)
		}
	})
}

// requireEveryElementMatters fails the test unless changing any single element
// of the array changes its hash.
func requireEveryElementMatters[T ~[4]uint32 | ~[5]uint64](t *testing.T, fn HashFunction, v T) {
	t.Helper()
	const seed = uint64(0x2468)
	h := fn(unsafe.Pointer(&v), seed)

	bytes := unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v))
	for i := range bytes {
		bytes[i] ^= 0xFF
		require.NotEqual(t, h, fn(unsafe.Pointer(&v), seed), "byte %d does not affect the hash", i)
		bytes[i] ^= 0xFF
	}
	require.Equal(t, h, fn(unsafe.Pointer(&v), seed), "restoring the bytes must restore the hash")
}

// --- Float and complex canonicalization through the generated closures -------

// TestGeneratedHashCanonicalizesFloat32 verifies that a float32 field obeys Go's
// equality rules rather than its bit pattern.
//
// Go says +0 == -0, so a set must not hold both. Go says NaN != NaN, so a NaN
// can never be found again -- but every NaN bit pattern still has to hash the
// same, or a set that collects NaNs fills one bucket per pattern.
//
// It hashes a float32 field holding +0 against -0, and several distinct NaN
// encodings against each other, and requires equal hashes; then requires an
// ordinary value to hash differently from zero.
func TestGeneratedHashCanonicalizesFloat32(t *testing.T) {
	fn := mustGenerate(t, bldFloat32Field{})
	const seed = uint64(0x5150)

	posZero := hashOf(fn, bldFloat32Field{F: 0}, seed)
	negZero := hashOf(fn, bldFloat32Field{F: float32(math.Copysign(0, -1))}, seed)
	require.Equal(t, posZero, negZero, "+0 and -0 are equal in Go and must hash equally")

	nan1 := math.Float32frombits(0x7FC00000)
	nan2 := math.Float32frombits(0x7FC00001)
	nan3 := math.Float32frombits(0xFFC00000) // negative NaN
	h1 := hashOf(fn, bldFloat32Field{F: nan1}, seed)
	require.Equal(t, h1, hashOf(fn, bldFloat32Field{F: nan2}, seed))
	require.Equal(t, h1, hashOf(fn, bldFloat32Field{F: nan3}, seed))

	require.NotEqual(t, posZero, hashOf(fn, bldFloat32Field{F: 1.5}, seed))
	require.NotEqual(t, posZero, hashOf(fn, bldFloat32Field{F: float32(math.Inf(1))}, seed),
		"infinity is an ordinary value, not a NaN")
}

// TestGeneratedHashCanonicalizesComplex verifies that the same equality rules
// hold for each half of a complex value.
//
// A complex is two floats, and Go compares it componentwise, so ±0 and NaN need
// canonicalizing in the real and imaginary part independently. Canonicalizing
// only one half is an easy mistake that ordinary values would never reveal.
//
// It varies each component in turn between +0 and -0, and between NaN patterns,
// and requires the hash to stay put, for complex64 and complex128.
func TestGeneratedHashCanonicalizesComplex(t *testing.T) {
	const seed = uint64(0x1DEA)
	negZero := math.Copysign(0, -1)

	t.Run("complex64", func(t *testing.T) {
		fn := mustGenerate(t, bldComplex64Field{})
		h := hashOf(fn, bldComplex64Field{C: complex(float32(0), float32(0))}, seed)
		require.Equal(t, h, hashOf(fn, bldComplex64Field{C: complex(float32(negZero), float32(0))}, seed),
			"a negative zero in the real part changed the hash")
		require.Equal(t, h, hashOf(fn, bldComplex64Field{C: complex(float32(0), float32(negZero))}, seed),
			"a negative zero in the imaginary part changed the hash")

		nanA := math.Float32frombits(0x7FC00000)
		nanB := math.Float32frombits(0x7FC00009)
		require.Equal(t,
			hashOf(fn, bldComplex64Field{C: complex(nanA, nanA)}, seed),
			hashOf(fn, bldComplex64Field{C: complex(nanB, nanB)}, seed),
			"different NaN encodings must hash the same")

		require.NotEqual(t, h, hashOf(fn, bldComplex64Field{C: complex(float32(1), float32(0))}, seed))
		require.NotEqual(t, h, hashOf(fn, bldComplex64Field{C: complex(float32(0), float32(1))}, seed))
	})

	t.Run("complex128", func(t *testing.T) {
		fn := mustGenerate(t, bldComplex128Field{})
		h := hashOf(fn, bldComplex128Field{C: complex(0.0, 0.0)}, seed)
		require.Equal(t, h, hashOf(fn, bldComplex128Field{C: complex(negZero, 0.0)}, seed),
			"a negative zero in the real part changed the hash")
		require.Equal(t, h, hashOf(fn, bldComplex128Field{C: complex(0.0, negZero)}, seed),
			"a negative zero in the imaginary part changed the hash")

		nanA := math.Float64frombits(0x7FF8000000000000)
		nanB := math.Float64frombits(0x7FF8000000000009)
		require.Equal(t,
			hashOf(fn, bldComplex128Field{C: complex(nanA, nanA)}, seed),
			hashOf(fn, bldComplex128Field{C: complex(nanB, nanB)}, seed),
			"different NaN encodings must hash the same")
		require.Equal(t,
			hashOf(fn, bldComplex128Field{C: complex(1.0, nanA)}, seed),
			hashOf(fn, bldComplex128Field{C: complex(1.0, nanB)}, seed),
			"the imaginary part is not canonicalized independently")

		require.NotEqual(t, h, hashOf(fn, bldComplex128Field{C: complex(1.0, 0.0)}, seed))
		require.NotEqual(t, h, hashOf(fn, bldComplex128Field{C: complex(0.0, 1.0)}, seed))
	})
}
