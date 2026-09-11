# Set3

[![Go Reference](https://pkg.go.dev/badge/github.com/TomTonic/Set3.svg)](https://pkg.go.dev/github.com/TomTonic/Set3)
[![Linter](https://github.com/TomTonic/Set3/actions/workflows/lint.yml/badge.svg)](https://github.com/TomTonic/Set3/actions/workflows/lint.yml)
[![Tests](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml)
![coverage](https://raw.githubusercontent.com/TomTonic/Set3/badges/.badges/main/coverage.svg)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9470/badge)](https://www.bestpractices.dev/projects/9470)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/TomTonic/Set3/badge)](https://scorecard.dev/viewer/?uri=github.com/TomTonic/Set3)

Set3 is a high-performance, native Golang set implementation. Against `map[type]struct{}`, the standard foundation for most
Go set implementations, it holds about a third of the bytes and is faster on most workloads — dramatically so on iteration, set
algebra and bulk building. The [Performance](#performance) section has the measurements, one workload at a time, each with the
confidence interval and the machine's own noise floor next to it. Set3 additionally lets you trade space against speed through
`RehashToCapacity(newCapacity)`, which an implementation built on `map[type]struct{}` cannot offer at all — and which the suite
uses to compare the two at the *same* occupancy as well as as-shipped.

The code is derived from [SwissMap](https://github.com/dolthub/swiss) and it implements the "Fast, Efficient, Cache-friendly Hash Table" found in [Abseil](https://abseil.io/blog/20180927-swisstables).
For details on the algorithm see the [CppCon 2017 talk by Matt Kulukundis](https://www.youtube.com/watch?v=ncHmEUmJZf4).
The dependency on x86 assembler for [SSE2/SSE3](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) instructions has been removed for portability and speed; the code runs faster without SSE and the necessary additional stack frame.
Hashing is done by the [`hashing`](hashing) package, which picks a hash function per element type when a set is created and falls back to Go's own `hash/maphash` for types it has no specialised routine for.

The name "Set3" comes from the fact that this was the 3rd attempt for an optimized datastructure/code-layout to get the best runtime performance.

## Installation

To use the `Set3` package in your Go project, follow these steps:

1. **Initialize a Go module** (if you haven't already):

   ```sh
   go mod init your-module-name
   ```

2. **Add the package**: Simply import the package in your Go code, and Go modules will handle the rest:

   ```go
   import "github.com/TomTonic/Set3"
   ```

3. **Download dependencies**: Run the following command to download the dependencies:

   ```sh
   go mod tidy
   ```

   This will automatically download and install the Set3 package along with any other dependencies.

## Using Set3

The following test case creates two sets and demonstrates some operations. For a full list of operations on the `Set3` type, see [API doc](https://pkg.go.dev/github.com/TomTonic/Set3#Set3).

```go
func TestExample(t *testing.T) {
    // create a new Set3
    set1 := Empty[int]()
    // add some elements
    set1.Add(1)
    set1.Add(2)
    set1.Add(3)
    // add some more elements
    set1.AddAllOf(4, 5, 6)
    // create a second set directly from an array
    set2 := FromArray([]int{2, 3, 4, 5})
    // check if set2 is a subset of set1. must be true in this case
    isSubset := set1.ContainsAll(set2)
    assert.True(t, isSubset, "%v is not a subset of %v", set2, set1)
    // mathematical operations like Unite, Subtract and Intersect
    // do not manipulate a Set3 but return a new set
    intersect := set1.Intersect(set2)
    // compare sets. as set2 is a subset of set1, intersect must be equal to set2
    equal := intersect.Equals(set2)
    assert.True(t, equal, "%v is not equal to %v", intersect, set2)
}
```

## Repository layout

```text
set3.go            the library — the Set3 type and every operation on it
hashing/           hash function selection and the per-type hash routines
internal/prime/    the primality search that sizes the control table
lab/               experiments and long-running measurements (see lab/README.md)
```

Everything under `lab/` is compiled out by default. It carries the `set3lab`
build tag, so `go build ./...` and `go test ./...` do not see it at all:

```sh
go test ./...                          # the library — seconds
go test -tags set3lab -short ./lab/... # the experiments too — minutes
```

The linter is configured with the tag on, so the experiment code is still
checked on every run and cannot rot unnoticed. `lab/README.md` explains what
lives there and how to run each suite for real.

## Performance

Set3 is measured against `map[T]struct{}` by the suite in
[`lab/setcompare`](lab/setcompare): twelve workloads, four key types, and set
sizes from a thousand elements to two million by default, or to eight million
with `SET3_CMP_HUGE=1`. Six of the workloads are shaped
after something a program actually does with a set — a membership filter on a
request path, a deduplication window with an expiry, a live index under churn,
the visited set of a graph traversal, an inverted-index intersection, a flush
that walks every element. The other four isolate one cost each, because a
realistic workload that mixes four costs cannot tell you which of them moved,
and two repeat the membership workloads with Set3 rehashed to the native map's
occupancy — see [Memory](#memory) for why that is a separate question.

Every number comes with the evidence for it. The suite runs the full
[rtcompare](https://github.com/TomTonic/rtcompare/blob/main/HOWTO.md) protocol:
batches sized so the clock contributes a bounded error, each candidate run
against *itself* first to find out what this machine reports as a difference
when there provably is none, the two candidates then measured interleaved,
each series tested for a trend across the run, and correlated samples resampled
in blocks. A result that does not clear both zero and that noise floor is
reported as **unresolved** rather than as a small win, and such cells are drawn
in the charts rather than dropped.

```sh
go test -tags set3lab -run TestCompareSuite -v -count=1 -timeout 180m ./lab/setcompare
go run  -tags set3lab ./lab/cmd/setchart -in lab/results/setcompare
```

The measurement takes upwards of an hour; drawing takes an instant. It writes
`lab/results/setcompare/` as CSV, with a `run.txt` naming the machine and the
configuration, and the chart tool turns those into the SVGs below. See
[lab/README.md](lab/README.md) for the knobs.

Read the machine's own verdict on itself before reading any result: `run.txt`
and the `note` column record the noise floor, the cells that drifted, and the
rate at which the setup reported a difference between two runs of *identical*
code. A run whose warnings mention a false-signal rate well above 10% was taken
on a machine that was not holding still, and its magnitudes should not be
quoted.

### How much faster

> **Being re-measured.** The figures that stood here came from a run whose
> batch lengths were too short, which made the answers depend on how long the
> harness looked rather than on the code — see `targetBatchDuration` and
> `collectOptionsFor` in [`lab/setcompare/runtime.go`](lab/setcompare/runtime.go)
> for what went wrong and
> `TestQuantizationTargetDoesNotChangeTheAnswer` for the guard that now stands
> over it. The suite is being re-run at the corrected settings, and at two
> operating points rather than one: as shipped, and with Set3 rehashed to the
> native map's occupancy, so that the space/time trade is held fixed and the
> layouts are compared on their own. The charts and the table return here when
> that run lands.
>
> The memory results below are unaffected: they come from a separate,
> deterministic pass that has reproduced to the digit across every run.

### Memory

Set3 holds between 31% and 64% of the native map's bytes, depending on the key
type and on how the container was filled. This is measured as heap still live
after a full collection with the container reachable — not as anything either
container reports about itself — and the measurement is calibrated against a
`[]uint64` of known size on every run. (The chart returns with the rest of them;
the table is the substance.)

| Fill history | `Set3[uint64]` | `map[uint64]struct{}` | ratio |
| --- | --- | --- | --- |
| created at the right capacity | 11.1 B/elem | 36.1 B/elem | **0.31×** |
| grown from empty | 13.1 B/elem | 36.1 B/elem | **0.36×** |
| filled, then half removed | 22.2 B/elem | 72.2 B/elem | **0.31×** |
| a sliding window after 4× its size in churn | 16.6 B/elem | 36.1 B/elem | **0.46×** |

Two mechanisms account for all of the difference, and both are properties of
the Go runtime rather than of anyone's benchmark:

1. **A `struct{}` value is not free.** The native map's slot is a struct of key
   and element, and Go pads a struct whose last field is zero-sized so that a
   pointer one past the end cannot escape the object. `struct{uint64; struct{}}`
   therefore occupies 16 bytes — exactly as much as `struct{uint64; uint64}`,
   which is why a `map[uint64]struct{}` and a `map[uint64]uint64` measure
   identically here. With eight control bytes per group of eight slots, the
   native map pays 17 bytes per slot. Set3 stores the keys themselves in a
   `[8]T` array with one 64-bit control word beside it, and pays 9.
2. **The occupancy differs.** At 17 bytes per slot, 36 bytes per element works
   out to 2.12 slots per element, so the map runs about 47% full. Set3 runs to
   its limit of 6.67 slots in 8, which is 83%.

Nine bytes per slot at 83% against seventeen at 47% is the whole of it. Note
what that is *not*: it is not waste. A lower load factor buys shorter probe
sequences, and the native map spends memory on exactly that — which is why the
suite also measures Set3 rehashed to the map's occupancy. At equal load factor
Set3 still holds about 53% of the bytes, because the padded slot is a separate
effect from the occupancy; what changes is the probe length, and that is the
part a speed comparison at unequal occupancy silently folds in.

Note also that these figures moved with Go itself. On the bucket map that this
README's original 25% figure was taken on, `map[uint64]struct{}` cost about 12
bytes per element; the Swiss-table rewrite in Go 1.24 tripled it. The suite
carries that as an executable note — `TestNativeMapFootprintIsWhatWeMeasure`
fails if the runtime's representation changes again.

### Profile-Guided Optimization

`Set3` picks its hash function once, when the set is created, and stores it as a
function value. Calling it is therefore an indirect call that the compiler
cannot resolve on its own, and for small keys that call is a noticeable part of
the work: hashing a `uint64` costs about 1 ns, the indirect call about 0.8 ns on
top.

[Profile-guided optimization](https://go.dev/doc/pgo) removes it. With a profile
that covers the hot path, the compiler replaces the indirect call with a guarded
direct call, which is then inlinable:

```text
./hashing/hasher.go:41:13: PGO devirtualizing function call hashing.h.fn to hashing.HashI64WHdet
```

Measured on `Set3[uint64]` (AMD Ryzen 9 7900, Go 1.26.8):

| Benchmark              | no PGO    | with PGO  |       |
| ---------------------- | --------- | --------- | ----- |
| `Contains`             | 8.07 ns   | 6.77 ns   | -16%  |
| `Add` (1000 elements)  | 9406 ns   | 8327 ns   | -11%  |

PGO removes the indirect call itself. The call *frame* around it is a separate
matter and is already gone: `RuntimeHasher.Hash` is written as a single
expression so that its inline cost stays inside the compiler's budget and it
gets inlined into `Contains`, `Add` and `Remove`. See the note on `Hash` in
[`hashing/hasher.go`](hashing/hasher.go) before reformatting it.

PGO is applied by the binary being built, not by this library, so the win is
yours to collect: record a CPU profile of your application under a realistic
load, drop it in your main package as `default.pgo`, and rebuild. Nothing in
`Set3` needs to change.

### Earlier measurements (v0.4.0, Go 1.23)

The charts below predate the suite above. They were produced with
[v0.4.0](https://github.com/TomTonic/Set3/releases/tag/v0.4.0) on Go 1.23.1
without PGO, by `lab/setbench`, and they cover a range the current suite does
not: every size from 1 to 300 elements, one at a time, where the constant costs
of allocating a container still dominate. Raw results are
[in plain text](lab/results/benchresult.txt); see
[lab/README.md](lab/README.md) for how to reproduce them.

#### Inserting Nodes into an Empty Set

The following chart illustrates the time required to insert random uint64 values into newly allocated sets.
The displayed times encompass the set allocation process.
All sets were allocated with an initial capacity of 21 elements, which is the current default in Set3.
As a result, rehashing occurs—sometimes multiple times—for larger sets.
This effect is clearly visible in the two charts below.

n = 1 ... 300 (step size +1, linear scale)
![Time for Inserting n Random Elements into an Empty Set, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/b2496fb4-2ff8-4539-9e95-748d108df830)

Please note that the memory chart displays the total memory consumption divided by the number of elements in the set.
This effectively represents the memory usage for storing a single element, i.e., 8 bytes.
Additionally, be aware of the lower bound of 10 bytes and the logarithmic scale of the y-axis.

n = 1 ... 300 (step size +1, linear scale)
![Memory required to store an Element in a Set of Size n, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/ba04f5cf-bca1-453b-9f90-e55d9ede58e5)

#### Searching Nodes in a Populated Set

The following chart illustrates the time required to determine whether a random value is present in the set.
The test driver maintains a 30% hit ratio, ensuring that 30% of the queried values are contained within the set, while the remaining 70% are not.
The x-axis represents sets of varying sizes, and the y-axis indicates the average time taken to look up a random value in a set of the corresponding size.

n = 1 ... 300 (step size +1, linear scale)
![Time for Searching Random Values in a Set of Size n, 30% Hit Rate, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/bf77efc4-fb60-4de4-a65e-087318e3958c)
