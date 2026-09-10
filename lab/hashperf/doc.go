//go:build set3lab

// Package hashperf compares hash function implementations by runtime.
//
// The tests use the rtcompare library to measure execution speed and decide
// statistically whether a difference is real. They are expensive by design and
// skip themselves under coverage instrumentation, because the counter updates
// coverage adds to hot paths can invert the tiny deltas being measured.
//
// This package is behind the set3lab build tag; see lab/README.md.
package hashperf
