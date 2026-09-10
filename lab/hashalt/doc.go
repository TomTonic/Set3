//go:build set3lab

// Package hashalt collects hash functions that Set3 does not use.
//
// Every function here was a candidate for the production dispatch in
// [github.com/TomTonic/Set3/hashing] and lost, or has not been evaluated yet.
// They are kept so that the choice stays reviewable: hashquality re-runs the
// distribution and ranking comparisons against them, hashperf re-runs the
// runtime comparisons. Each name is labelled with its strategy (SM = SplitMix,
// WH = WyHash non-deterministic, MH = maphash).
//
// This package is behind the set3lab build tag; see lab/README.md.
package hashalt
