//go:build set3lab

// Package hashalt collects hash functions that Set3 does not use.
//
// Every function here was a candidate for the production dispatch in
// [github.com/TomTonic/Set3/hashing] and lost, has not been evaluated yet, or
// was superseded and is kept as the baseline its replacement is measured
// against. They are kept so that the choice stays reviewable: hashquality
// re-runs the distribution and ranking comparisons against them, hashperf
// re-runs the runtime comparisons. Each name is labelled with its strategy
// (SM = SplitMix, WH = WyHash non-deterministic, MH = maphash); the Serial*
// routines are the superseded byte-block hashes, named for the serial
// dependency chain that motivated replacing them.
//
// This package is behind the set3lab build tag; see lab/README.md.
package hashalt
