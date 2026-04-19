// Package performance contains runtime performance comparison tests for hash
// functions. These tests use the rtcompare library to measure and statistically
// compare the execution speed of different hash function implementations.
//
// All tests are intentionally expensive and are skipped under coverage
// instrumentation because coverage adds counter updates in hot paths that can
// invert tiny runtime deltas.
package performance
