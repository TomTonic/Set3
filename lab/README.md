# lab — experiments and long-running suites

Everything in this directory is **switched off by default**. It is real,
maintained code, but it is not part of the library and it must never slow down
or clutter the everyday `go build ./...` / `go test ./...` cycle.

The switch is a single Go build tag: **`set3lab`**. Every `.go` file below
`lab/` starts with

```go
//go:build set3lab
```

Without the tag the Go tool does not see these packages at all — `go list ./...`
does not name them, `go test ./...` does not run them, and coverage does not
count them. With the tag everything is compiled, vetted, linted, and runnable.

## Which Go the lab runs on

Whatever `go.mod` requires, which is Go 1.27 or newer. That is a deliberate
floor for the whole module, not just for the lab, and it matters here for one
reason worth knowing when comparing old numbers with new: Go 1.27 generates
size-specialized allocation routines for objects under 80 bytes, cutting the
cost of small allocations by up to 30%. A comparison of two containers is
partly a comparison of their allocation behaviour, so a measurement taken on an
older toolchain describes a different runtime.

`run.txt` records the toolchain every measurement was taken on, so a recorded
result always says which runtime it describes. To check whether that change is
what moved a number, `GOEXPERIMENT=nosizespecializedmalloc` turns it back off.

## Running it

```sh
go build   -tags set3lab ./...      # compile everything, including lab
go vet     -tags set3lab ./...
golangci-lint run --build-tags set3lab ./...

go test    -tags set3lab -short ./lab/...   # every suite, cheap paths only
go test    -tags set3lab ./lab/...          # the real thing — hours, not minutes
```

`-short` is the useful middle setting: every suite is compiled and executed, but
the parts that run for tens of minutes skip themselves. That is what CI uses to
prove nothing here has rotted (see `.github/workflows/lab.yml`, weekly and on
demand).

## What lives here

| Package | What it is |
| --- | --- |
| [`hashalt/`](hashalt) | Hash functions that Set3 does *not* use — candidates that lost the comparison, were never finished, or were superseded. The `Serial*` routines are the byte-block hashes the lane-parallel ones replaced, kept runnable so the speed claim stays checkable. |
| [`hashbench/`](hashbench) | Shared plumbing for the hash experiments: data generators, statistics, small PRNGs. |
| [`hashquality/`](hashquality) | Distribution, uniformity, avalanche and ranking suites over `hashalt` and the production functions, plus the Go authors' SMHasher port retargeted onto `hashing` (`smhasher_test.go`). Some suites run for tens of minutes. |
| [`hashperf/`](hashperf) | Runtime A/B comparisons of hash functions via `rtcompare`. |
| [`setbench/`](setbench) | The older `Set3` vs. `map[T]struct{}` measurements: fill/lookup series, memory footprint, a hand-rolled A/B/B/A comparison, and a randomised operation mix for CPU profiling. |
| [`setcompare/`](setcompare) | The current `Set3` vs. `map[T]struct{}` suite: ten workloads across four key types and five sizes, measured through the full `rtcompare` protocol, plus a calibrated memory pass. Writes CSV. |
| [`cmd/powerchi2/`](cmd/powerchi2) | Chi-squared power and sample-size calculator, used to size the tests in `hashquality`. |
| [`cmd/primetest/`](cmd/primetest) | Standalone driver for the primality search that `internal/prime` does inside the library. |
| [`cmd/setchart/`](cmd/setchart) | Turns `setcompare`'s CSV into the SVG charts the top-level README embeds. |
| [`results/`](results) | Recorded measurement output, kept for reference. |

## Three levels of "off"

Some things are too expensive even for a tagged full run, so they are gated a
second time:

1. **Build tag `set3lab`** — off unless asked for. Applies to everything here.
2. **`testing.Short()`** — the expensive body of a suite is skipped under
   `-short`. Applies to most suites.
3. **Environment variables** — the exhaustive runs stay off until named
   explicitly:

   | Variable | Effect |
   | --- | --- |
   | `SET3_HASH32_EXHAUSTIVE=1` | full 2^32 ranking sweep in `hashquality` |
   | `SET3_HASH32_EXHAUSTIVE_WORKERS=n` | worker count for that sweep |
   | `SET3_HASH_EXHAUSTIVE_MULTI=1` | exhaustive multi-criteria ranking |
   | `SET3_HASH_RANK_SEEDS=n` | number of seeds in the ranking runs |
   | `SET3_RTCOMPARE_*` | size, repeats, precision, mode of the older `setbench` comparison |
   | `SET3_CMP_HUGE=1` | adds the 8.4-million-element tier to the `setcompare` suite |
   | `SET3_CMP_*` | everything else about `setcompare`: sizes, key types, scenarios, budget |

A handful of suites are hard-skipped with a bare `t.Skip("unskip for …")`
because they run for hours and only make sense when you are looking at their
output. Those say so in the skip message.

## Reproducing the README benchmarks

The charts in the top-level [README](../README.md) come from `setcompare`, in
two steps — measure, then draw:

```sh
go test -tags set3lab -run TestCompareSuite -v -count=1 -timeout 180m ./lab/setcompare
go run  -tags set3lab ./lab/cmd/setchart -in lab/results/setcompare
```

The first command takes 20 to 40 minutes on a quiet machine and writes
`runtime.csv`, `memory.csv` and `run.txt` into `lab/results/setcompare`. The
second reads them and writes the SVGs plus `README-snippet.md`, which holds the
markdown to paste. Drawing is instant, so a chart can be reworked without
measuring again.

### One setting to know about

`setcompare` does not use rtcompare's default batch length, and the reason
generalises to any suite in this directory. rtcompare sizes batches through
`MaxQuantizationError`, a relative bound on what the *clock* contributes;
its default of 0.001 gives roughly a thousand clock ticks per batch, about
30 microseconds on Linux. That is the right size for the thing it bounds and
much too short for these workloads, for reasons that have nothing to do with
the clock:

- A short batch amortises its own cold start over very few operations. One
  `setcompare` cell calibrated to **five operations per batch** at the default.
- A collection between batches — rtcompare's `GCBetween` — evicts the caches,
  and the two candidates do not re-warm equally: `Set3`'s table is a third the
  size of the native map's. On a workload that allocates nothing, that alone
  moved the answer by five percentage points, in `Set3`'s favour.
- A workload whose working set is tens of megabytes measures cache and TLB
  state as much as it measures code. A short batch samples the state it started
  in instead of averaging over it.

Together those are not a small correction. On the largest cell the suite
measures, a 30-microsecond batch reported **-44.5%** where a converged
measurement reports **+8.3%** — a sign flip across 53 percentage points.

`setcompare` therefore targets a batch **duration** (`targetBatchDuration`,
3 ms) and derives rtcompare's relative knob from the clock precision measured
on the machine, so the batch stays the same length on a 30 ns Linux clock and a
100 ns Windows one. `TestQuantizationTargetDoesNotChangeTheAnswer` is the
standing guard: it re-measures the most sensitive cells at four batch lengths
and fails if doubling the suite's length still moves the answer.

If you write a new comparison in `lab/`, start from that test rather than from
the default. `lab/hashperf` arrived at the same place independently — its
`MaxQuantizationError: 0.00001` is a 3 ms batch on this machine.

### Running it on a quiet machine

Run it on a machine that is doing nothing else, plugged in, with no CPU
frequency scaling if you can turn it off. The suite will tell you if you did
not: every row carries the noise floor this machine produced between two runs
of identical code, and rows that did not clear it are reported as unresolved
rather than as small wins. A run whose warnings mention a false signal rate
well above 10% was measured on a machine that was not holding still, and its
magnitudes should not be quoted.

Useful settings:

| Variable | Effect |
| --- | --- |
| `SET3_CMP_SIZES=1000,16384` | measure only these set sizes |
| `SET3_CMP_KEYS=uint64` | measure only this key type |
| `SET3_CMP_SCENARIOS=lookup-hit30,iterate` | measure only these workloads |
| `SET3_CMP_BUDGET=60s` | more time per cell, so fewer cells fall back to a reduced schedule |
| `SET3_CMP_HUGE=1` | add the 8.4-million-element tier |
| `SET3_CMP_SKIP_MEMORY=1` | runtime pass only |
| `SET3_CMP_TAG=pgo` | label the run in `run.txt` |

### The older setbench charts

The previous generation of charts came from `setbench` and is kept because the
data behind them is:

```sh
go test -tags set3lab -v -count=1 -timeout=120m \
  -run "^(TestSet3Fill|TestNativeMapFill|TestSet3Find|TestNativeMapFind)$" \
  ./lab/setbench > lab/results/benchresult.txt
```

Those four tests start with `t.Skip`; comment the skip out first. The full run
takes about 45 minutes. The Go benchmark form of the same measurements is:

```sh
go test -tags set3lab -benchmem -benchtime=6s -timeout=480m -run="^$" \
  -bench "^(BenchmarkSet3Fill|BenchmarkNativeMapFill|BenchmarkSet3Find|BenchmarkNativeMapFind)$" \
  ./lab/setbench > lab/results/benchresult.txt
```

## Adding something here

Give the new file the `//go:build set3lab` line, put it in the package it
belongs to, and make sure it survives `go test -tags set3lab -short ./lab/...`.
If it takes longer than a few seconds, guard the expensive part with
`testing.Short()` so the weekly CI run stays useful.
