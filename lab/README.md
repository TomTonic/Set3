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
| [`hashalt/`](hashalt) | Hash functions that Set3 does *not* use — candidates that lost the comparison, or were never finished. Kept so the choice stays reviewable. |
| [`hashbench/`](hashbench) | Shared plumbing for the hash experiments: data generators, statistics, small PRNGs. |
| [`hashquality/`](hashquality) | Distribution, uniformity, avalanche and ranking suites over `hashalt` and the production functions. Some suites run for tens of minutes. |
| [`hashperf/`](hashperf) | Runtime A/B comparisons of hash functions via `rtcompare`. |
| [`setbench/`](setbench) | `Set3` vs. `map[T]struct{}`: fill/lookup series, memory footprint, the `rtcompare` head-to-head, and a randomised operation mix for CPU profiling. |
| [`cmd/powerchi2/`](cmd/powerchi2) | Chi-squared power and sample-size calculator, used to size the tests in `hashquality`. |
| [`cmd/primetest/`](cmd/primetest) | Standalone driver for the primality search that `internal/prime` does inside the library. |
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
   | `SET3_RTCOMPARE_*` | size, repeats, precision, mode of the `setbench` comparison |

A handful of suites are hard-skipped with a bare `t.Skip("unskip for …")`
because they run for hours and only make sense when you are looking at their
output. Those say so in the skip message.

## Reproducing the README benchmarks

The charts in the top-level [README](../README.md) come from `setbench`:

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
