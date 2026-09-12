# Set3

[![Go Reference](https://pkg.go.dev/badge/github.com/TomTonic/Set3.svg)](https://pkg.go.dev/github.com/TomTonic/Set3)
[![Linter](https://github.com/TomTonic/Set3/actions/workflows/lint.yml/badge.svg)](https://github.com/TomTonic/Set3/actions/workflows/lint.yml)
[![Tests](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml)
![coverage](https://raw.githubusercontent.com/TomTonic/Set3/badges/.badges/main/coverage.svg)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9470/badge)](https://www.bestpractices.dev/projects/9470)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/TomTonic/Set3/badge)](https://scorecard.dev/viewer/?uri=github.com/TomTonic/Set3)

Set3 is a high-performance, native Golang set implementation built on the Abseil "Swiss table" layout rather than on
`map[type]struct{}`, the standard foundation for most Go set implementations.

Out of the box it holds 42% of the native map's bytes for `uint64` keys and is faster in 105 of the 110 measured cases
that cleared this machine's noise floor — by a median of 35%, and by 58% to 74% on iteration. It is smaller and faster
at the same time, which is possible because the two containers sit at different operating points: Set3 fills its table
to 60% where the native map stops near 47%, and its slot costs 9 bytes where the map's costs 17. Unlike a map, Set3 lets
you move that operating point with `RehashToCapacity(newCapacity)` — a lookup-bound set can trade memory back for a few
more percent. The [Performance](#performance) section shows the whole curve, one workload at a time, each number with
its confidence interval and the machine's own noise floor beside it, including the five cases it loses — all of them
with `string` keys.

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
compact.go         in-place tombstone reclamation, called from Remove
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
[`lab/setcompare`](lab/setcompare): thirteen workloads, four key types — three
of them in the charts below — and set sizes from a thousand elements to two
million by default, or to eight million with `SET3_CMP_HUGE=1`. Eight of the
workloads are shaped after something a program actually does with a set: a
membership filter on a request path, a deduplicator over a skewed event stream,
a sliding dedup window with an expiry, the same window over keys it has never
seen before, a live index under churn, the visited set of a graph traversal, an
inverted-index intersection, and a flush that walks every element. Three isolate
one cost each, because a realistic workload that mixes four costs cannot tell
you which of them moved. The last two repeat the membership workloads with Set3
rehashed to the native map's occupancy, so that the space/time trade is held
fixed and the layouts are compared on their own.

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

The measurement takes about ninety minutes; drawing takes an instant. It writes
`lab/results/setcompare/` as CSV, with a `run.txt` naming the machine and the
configuration, and the chart tool turns those into the SVGs below. See
[lab/README.md](lab/README.md) for the knobs.

Read the machine's own verdict on itself before reading any result: `run.txt`
and the `note` column record the noise floor, the cells that drifted, and the
rate at which the setup reported a difference between two runs of *identical*
code. A run whose warnings mention a false-signal rate well above 10% was taken
on a machine that was not holding still, and its magnitudes should not be
quoted.

### Set3 is a curve; a map is a point

This is the measurement that frames all the others. Both axes are costs, so
down and left is better, and the shaded region is everything cheaper than the
native map *on both axes at once*. Each Set3 point is labelled with the
occupancy that produced it.

![Set3's space/time curve, uint64 keys](lab/results/setcompare/loadcurve-uint64.svg)

Read at 16 384 `uint64` elements, 30% hit rate:

| operating point | bytes/element | ns/lookup | vs. the map |
| --- | --- | --- | --- |
| `map[uint64]struct{}` | 36.1 | 10.38 | — |
| Set3 at 0.59 load — **as shipped** | 15.3 | 7.90 | **+24.2%** |
| Set3 at 0.50 load | 18.0 | 7.18 | **+30.7%** |
| Set3 at 0.41 load | 22.0 | 6.80 | **+34.6%** |
| Set3 at 0.34 load | 26.5 | 6.69 | **+35.5%** |
| Set3 at 0.29 load | 31.5 | 6.71 | +35.3% |
| Set3 at 0.24 load | 38.0 | 6.69 | +35.5% |

The curve has no right-hand half, and that is deliberate: `set3maxAvgGroupLoad`
is 4.8 elements per group of eight, so 0.60 is the highest average occupancy any
table reaches. The default *is* the right-hand end.

Three things follow, and none of them is visible in a single-number comparison.

**The default sits inside the shaded region.** As shipped, Set3 is 24% faster
on this workload while holding 42% of the map's bytes. Both axes at once, with
no tuning and no call to `RehashToCapacity`.

**One step left is nearly free, and the rest is not.** Going from 0.59 to 0.50
costs 2.7 bytes per element and buys 6.4 percentage points. Going on to 0.24
costs another twenty bytes — putting Set3 *above* the map's 36.1 — and buys 4.8
more. The knee is around 0.41: +34.6% at 61% of the map's memory.

**Past the knee it can reverse.** At two million elements the curve turns back
up, because by then the table stops fitting where it did. For `string` keys it
peaks at +15.9% around 0.42 occupancy and falls to +5.5% at 0.24; for `uint64`
it peaks at +33.4% around 0.51. The optimum occupancy is a function of your
working set, which is exactly why the knob exists and why the chart is a curve
rather than a recommendation.

So the actionable advice is short: **leave it alone.** The default is a good
operating point on both axes. If a profile says a particular set is
lookup-bound and you have the memory, one `RehashToCapacity` to about 0.50 is
worth a few percent; below that you are paying bytes for nothing.

#### Why the default is 4.8

`set3maxAvgGroupLoad` was 6.667 — 0.83 occupancy — and that was chosen for
memory alone. It was retuned against measurement: five values, each with its own
build and its own full run of the suite, 61 cells apiece.

| `set3maxAvgGroupLoad` | occupancy | `uint64` B/elem | vs. map | `lookup-hit30`, 6 cells | worst cell in the suite |
| --- | --- | --- | --- | --- | --- |
| 6.5 | 0.81 | 11.1 | 0.31× | −25.2% … +10.0% | −25.2% |
| 5.5 | 0.69 | 13.1 | 0.36× | +5.6% … +26.0% | −11.6% |
| **4.8** | **0.60** | **15.0** | **0.42×** | **+13.5% … +29.9%** | **−8.1%** |
| 4.2 | 0.53 | 17.1 | 0.48× | +16.3% … +31.5% | −7.4% |
| 3.6 | 0.45 | 20.0 | 0.55× | +14.5% … +33.9% | −37.4% |

The median across all 61 cells barely moves — +30.1%, +30.0%, +30.7%, +31.5%,
+29.9% — so the choice is not about the typical case at all. What moves is the
worst case. At 6.5 the table is tight enough that `lookup-hit30` *loses* to the
native map in four of six cells: −25.2% for `string` keys at 16 384 elements,
−17.0% for `uint64`, and Set3 is then on the wrong side of the shaded region in
the chart above. At 3.6 it fails from the other direction — `build-growing` with
`string` keys at two million falls to −37.4%, because a roomier table means more
memory to touch while filling it.

4.8 is the largest value at which nothing in the suite loses by more than 8%,
and it keeps the memory ratio under a half. 4.2 is marginally better on speed
for 14% more memory; that is the trade the default declines and
`RehashToCapacity` exposes.

### Workload by workload

Positive means Set3 is faster. The whisker is the 95% confidence interval, the
pale band is the noise floor, and a grey bar established nothing. Set3 is at
its default occupancy here except in the `-eqload` rows, which repeat the
membership workloads with it rehashed to the map's.

![Set3 vs map[uint64]struct{}](lab/results/setcompare/speedup-uint64.svg)

Of 119 cells, 110 cleared both zero and this machine's noise floor. The ranges
below are over those 110; the nine that did not are named at the end of this
section rather than folded in.

| Workload | `uint64` | `string` | `struct3x64` |
| --- | --- | --- | --- |
| `iterate` | +58 … +74% | +56 … +66% | +61 … +71% |
| `build-presized` | +44 … +65% | +38 … +67% | +55 … +70% |
| `churn-fresh` | +47 … +56% | — | +50 … +64% |
| `sliding-window` | +47 … +55% | +29 … +44% | — |
| `intersect` | +31 … +62% | −5 … +45% | — |
| `graph-visited` | +33 … +42% | — | — |
| `lookup-hit30` | +25 … +33% | +13 … +18% | +21 … +46% |
| `lookup-hit30-eqload` | +26 … +37% | +15 … +20% | +23 … +37% |
| `lookup-hit95` | +5 … +28% | — | +20 … +34% |
| `lookup-hit95-eqload` | +14 … +29% | — | +16 … +40% |
| `dedup-stream` | +22 … +28% | −19 … +3% | +30 … +46% |
| `mixed-index` | +15 … +42% | +8 … +12% | — |
| `build-growing` | +16 … +41% | −8 … +17% | — |

Iteration is the largest and most durable difference: Set3 scans its groups
linearly, the native map walks buckets from a randomised start, and at two
million `uint64` elements that is 2.77 ns against 7.40 ns per element. Building
a presized set of two million is 19.44 ns against 56.31 ns per element — most of
which is not hashing but touching under half as much memory.

The two `-eqload` rows hold the space/time trade fixed and ask what is left when
the operating point is removed from the comparison: the answer is most of it.
`lookup-hit30` at the default gives +25 … +33% for `uint64` and the same
workload at the map's own occupancy gives +26 … +37%. At 6.667 this pair was the
sharpest result in the suite, because the default was the thing costing the
lookups; at 4.8 the two rows agree, which is the measurement saying the default
is no longer the limiting factor.

String keys remain Set3's weakest case, and all five losses are there:
`dedup-stream` at 16k, 262k and 2.1M (−5.8%, −19.4%, −8.4%),
`build-growing` at 262k (−7.5%) and `intersect` at 1k (−5.3%). `dedup-stream` is
the clearest of them and the easiest to explain: it is `Contains`-then-`Add` over
a skewed stream, so it pays the hash twice per event against a map that pays it
once and reuses the lookup's position. A string hash is expensive enough for that
to decide the workload; for `uint64` and `struct3x64` keys the same scenario wins
by +22 … +28% and +30 … +46%.

![Set3 vs map[string]struct{}](lab/results/setcompare/speedup-string.svg)

About half the cost of a small-set string lookup is the hash itself, and
`BenchmarkStringHashRoutes` in the suite puts the routine these charts were
measured with at 3.93 ns against the runtime's 3.55 ns for a 20-byte key. That
gap is small; what dominated it before was a serial dependency chain of widening
multiplies — an N-word key cost 2N multiplies that the processor could not
overlap, because each one needed the previous one's result.

That chain is gone. The byte routines are now a specialization of
`memHashFallback` from the Go runtime's own map hasher — the Go authors'
adaptation of [wyhash](https://github.com/wangyi-fudan/wyhash) — which consumes
words in independent lanes and closes with two mixing steps instead of one per
word. Set3 specializes it to native endianness, derives its four secret words
from the per-set seed instead of a process-global random array, and pins the hot
fixed key sizes to straight-line entry points.

Measured as latency, where nothing can overlap: a 16-byte key falls from 6.34 ns
to 2.54, a 24-byte struct key from 8.44 to 3.51, a 32-byte one from 10.51 to
3.40, a 24-byte string from 10.59 to 5.49 and a 64-byte string from 20.69 to
6.43. `lab/hashperf` keeps the superseded routine runnable as the baseline, so
the comparison stays checkable.

Quality is not argued from the speedup. `lab/hashquality/smhasher_test.go` is the
Go authors' port of [SMHasher](https://github.com/aappleby/smhasher) retargeted
onto this package — sanity, appended zeros, small keys, all-zero lengths, two
nonzero bytes, cyclic repeats, sparse bit patterns, block permutations, windowed
rotations, text, avalanche and seed sensitivity, all measured against a collision
bound derived from the birthday paradox. The `hashing` package carries its own
structural tests on top of that, including one that pins the read window to
exactly the key and one that holds each fixed-size entry point equal to the
generic path.

#### Struct keys that are not one block

A struct of integers or pointers merges into a single block of memory and goes
straight through the routine above — that is the `struct3x64` column here, and
it never had a problem. A struct holding floats cannot: -0.0 and +0.0 compare
equal and must not hash apart, every NaN has to hash alike, and the padding
between fields holds whatever was there before. The generator therefore used to
hash such a struct one field at a time, threading each field's hash into the
next as its seed. An N-field struct cost 2N widening multiplies that the
processor could not overlap, because each one needed the previous one's result.

It no longer does. Fields that reduce to a canonical 64-bit word — a float, a
complex part, an eight-byte run of byte-stable fields — are turned into a word
sequence, and that sequence is hashed by `wyBlock`'s own arithmetic, which
consumes two words per multiply and splits into three independent lanes past
six words. Fields that cannot become a word keep threading the seed, and
strings deliberately do: `HashString` is a real call, so the calls serialize
whatever the data dependencies say, and lane-mixing three string fields
measured 6% *slower* than chaining them.

Measured with rtcompare, ABBA-interleaved, 401 paired rounds of a million
hashes each, comparing the two closures the generator builds for the same type:

| struct shape | chain | words | words are faster by |
| --- | --- | --- | --- |
| 2 × `float64` | 3.74 ns | 1.96 ns | **47.6%** |
| 3 × `float64` | 5.27 ns | 3.13 ns | **40.5%** |
| 4 × `float64` | 6.85 ns | 3.89 ns | **43.1%** |
| 6 × `float64` | 10.07 ns | 5.23 ns | **48.0%** |
| 8 × `float64` | 13.97 ns | 6.86 ns | **50.9%** |
| 3 × `float32` | 6.33 ns | 2.93 ns | **53.6%** |
| 6 × `float32` | 11.81 ns | 4.46 ns | **62.2%** |
| 8 × `float32` | 16.89 ns | 5.62 ns | **66.8%** |
| 2 × `complex128` | 5.69 ns | 3.89 ns | **31.6%** |
| 2 × `int64` + `float64` | 3.40 ns | 2.75 ns | **19.1%** |
| 2 × `float64` + `string` | 6.09 ns | 4.84 ns | **20.5%** |

Each row is established at 100% bootstrap confidence against the ten-percent
threshold below its median — the two `float32` rows above 60% only to 50%,
because that is the highest threshold the test asks for. The `float32` rows
gain most because the chain hashed each float32 with splitmix64: two dependent
multiplies for four bytes of input.

One case is still declined: a struct mixing four-byte and eight-byte words, an
`int32` beside a `float64` for instance. Supporting it needs a second branch
per word to choose the load width, and that branch was measured to cost more
than the mixing saves — 7.22 ns against 3.65 on six words. Such a type keeps
the chain and is correct, just not accelerated.

The mixing is not new arithmetic. Each of the seven word mixers is a
transcription of `wyBlock` at one length, and
`TestWordMixersAreTheGenericPath` requires each to return exactly what the
generic routine returns for the same bytes — an exact oracle rather than a
statistical argument. On top of that, `TestWordPathMatchesAnIndependentWordSequence`
spells out by hand which words each type should produce, and
`TestWhichTypesTakeTheWordPath` pins the classification including every
declined case.

The suite charts predate this change, and the chart below is unaffected by it:
`struct3x64` is three `uint64` fields, which merge into one 24-byte block and
never took the per-field path at all.

![Set3 vs map[struct]struct{}](lab/results/setcompare/speedup-struct3x64.svg)

**The nine cells not quoted above.** Seven were resolved but came from a setup
that reported a difference between two runs of *identical* code in 30% to 68% of
its A/A trials — `mixed-index`/`string`/16k at 65%,
`lookup-hit30`/`string`/1k at 60% and its `-eqload` twin at 68%,
`sliding-window`/`uint64`/1k at 60% and `string`/16k at 30%,
`mixed-index`/`string`/1k at 38%, and `dedup-stream`/`uint64`/262k at 31%. That
rate is a property of the layout on this machine, not of the number of samples:
all seven were re-measured at a four times longer budget and did not improve.
The other two are genuinely unresolved, with intervals that include zero:
`build-growing`/`string`/2.1M at −1.9% [−3.4%, +2.6%] and
`lookup-hit95-eqload`/`uint64`/2.1M at −4.2% [−26.3%, +16.6%]. All nine are in
the CSV with their notes, and drawn grey in the charts.

### What it costs

The percentages say who won. These say what it cost, which is what decides
whether a percentage is worth anything. Both axes are logarithmic.

![Cost per operation, uint64 keys](lab/results/setcompare/cost-uint64.svg)

### Memory

Set3 holds between 42% and 80% of the native map's bytes, depending on the key
type and on how the container was filled. This is measured as heap still live
after a full collection with the container reachable — not as anything either
container reports about itself — and the measurement is calibrated against a
`[]uint64` of known size on every run.

![Retained memory per element, uint64 keys](lab/results/setcompare/memory-uint64.svg)

At two million elements:

| Fill history | `uint64` | `string` | `struct3x64` |
| --- | --- | --- | --- |
| created at the right capacity | **0.42×** | **0.53×** | **0.52×** |
| grown from empty | **0.54×** | **0.70×** | **0.68×** |
| filled, then half removed | **0.42×** | **0.53×** | **0.52×** |
| a window after 4× its size in churn | **0.62×** | **0.80×** | **0.78×** |

In absolute terms the first row is 15.0 B/element against the map's 36.1 for
`uint64`, 28.3 against 53.3 for `string`, and 41.7 against 80.1 for
`struct3x64`.

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
2. **The occupancy differs.** At 17 bytes per slot, 36.1 bytes per element works
   out to 2.12 slots per element, so the map runs about 47% full. Set3 runs to
   its limit of 4.8 slots in 8, which is 60%.

Nine bytes per slot at 60% against seventeen at 47% is the whole of it: 15.0
against 36.1. Note what that is *not*: it is not waste on either side. A lower
load factor buys shorter probe sequences, and the native map spends memory on
exactly that — which is what
[the curve above](#set3-is-a-curve-a-map-is-a-point) is measuring. The two
mechanisms are separable: the padded slot is worth a factor of about 1.9
whatever the occupancy, so at an identical load factor Set3 would hold roughly
53% of the bytes. Only the remaining 11 points are the tuning.

The `grown` row is one growth's worth of headroom: a table that arrives at two
million elements by doubling its way up lands wherever the last growth put it,
which averages out to 0.54 rather than 0.42. `EmptyWithCapacity`,
`RehashToCapacity` and `FromArray` exist to skip that, and the first two size
the table so that the requested count lands exactly at the default load factor.

#### Tombstones, and the churn that finds them

`Remove` can clear a slot outright only when its group has an empty slot to
terminate probes with; in a crowded table it usually does not, and leaves a
tombstone instead. Tombstones count as occupied for the purpose of the load
limit, so a workload that removes as often as it inserts can push a table to
grow while holding a constant number of elements. `Add` reuses a tombstone only
when one happens to lie on the probe path of the element being inserted.

Whether that drifts depends entirely on where the new keys come from, and the
difference is stark enough that a suite can measure one and miss the other.
Measured at a constant 262 144 `uint64` elements, with and without the in-place
compaction described below:

| key source | window turns | without compaction | with it |
| --- | --- | --- | --- |
| a ring of `2n` keys cycled through | 400 | 22.50 B/element, 40% full | 22.50 B/element, 40% full |
| fresh keys, never repeating | 20 | 33.76 B/element, 27% full | 22.50 B/element, 40% full |
| fresh keys, never repeating | 400 | 50.65 B/element, 18% full | 22.50 B/element, 40% full |

A re-inserted key hashes to the home group it had before and very often lands on
its own tombstone, so a cyclic window reuses tombstones almost perfectly and
never drifts at all. A cache, a work queue, or a deduplicator over a live stream
sees keys it has never seen, each probing from a fresh home group, and those
reuse a tombstone only by luck. Without compaction that table grew from 81 929
groups to 122 921 to 184 409 over the run and had not stopped; with it, it
settles after one growth and stays there for as long as the workload runs.

This is why the suite now has two churn scenarios rather than one.
`sliding-window` cycles a bounded ring and measures the first row above — it
could not have shown this defect at any size or duration. `churn-fresh` is the
same fixed-size window over keys the container has never seen, and it is the row
that decides whether tombstones are reused or merely accumulate. Both now report
0.000 B/op in steady state, which is the point: the growth is gone, not merely
amortised.

The reclamation is `compactInPlace` in [compact.go](compact.go), Abseil's
`drop_deletes_without_resize`: it rewrites the control bytes so that every
tombstone becomes empty and every live element becomes a tombstone, then walks
the table moving each element to where it would now probe to, swapping when the
target is still occupied. It allocates nothing — no second table — and a test
asserts that at several sizes. `Remove` calls it when tombstones reach a quarter
of the element limit — equivalently, when the live elements are at or below
three quarters of it. Abseil draws its line within a hair of the same place, at
25/32.

The check lives in `Remove` rather than in `Add` for two reasons: tombstones are
created there and nowhere else, and removal is the rarer operation in practice,
so the branch is on the colder path. Not every `Remove` creates a tombstone
either, so the condition is only evaluated when one was just made.

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
./hashing/hasher.go:54:13: PGO devirtualizing function call hashing.h.fn to hashing.HashI64WHdet
./hashing/hasher.go:54:13: PGO devirtualizing function call hashing.h.fn to hashing.HashString
./hashing/bytehash.go:36:21: PGO devirtualizing function call hashing.specialized to hashing.hashByteBlock24
```

**As a rule of thumb, PGO makes Set3 about 2% to 5% faster.** The point estimate
is +3.9%, with a 95% bootstrap interval of [+1.7%, +5.1%] over 59 cells and
20 000 resamples; Set3 was faster under PGO in 81% of them, with an interquartile
range of +0.9% to +8.2%.

What that does *not* buy is a larger lead over the native map, because the map
gets faster too: it gained +1.3% on the same cells, so the relative advantage
moved by +0.7pp with an interval of [−0.4pp, +1.3pp] — which includes zero.
Collect PGO because it makes your program faster, not because it changes this
comparison.

PGO is a property of the build, so this is the one comparison here that cannot
be interleaved: it was measured by running the whole suite twice on the same
quiet machine, once without a profile and once with one. Everything else in this
README is an ABBA-interleaved measurement, which is why its intervals are
tighter and why this one is quoted as a rule of thumb rather than per cell.

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
