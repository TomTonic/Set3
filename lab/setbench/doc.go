//go:build set3lab

// Package setbench holds the long-running measurements of Set3 against Go's
// native map[T]struct{}: the fill and lookup series behind the charts in
// README.md, the memory footprint survey, the rtcompare A/B/B/A comparison,
// and a randomised operation mix used for CPU profiling.
//
// Runtimes are measured in minutes, so most entry points skip themselves
// unless explicitly unskipped or run without -short. nativemap.go exists only
// to give the native map the same call shape as Set3, so both sides of a
// comparison pay for one non-inlined call.
//
// This package is behind the set3lab build tag; see lab/README.md.
package setbench
