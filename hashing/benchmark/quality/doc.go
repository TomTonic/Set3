// Package quality contains distribution quality tests, uniformity checks,
// and hash function ranking suites for the Set3 hashing infrastructure.
//
// These tests evaluate how well different hash function implementations
// distribute values across bucket counts, how their avalanche behaviour
// compares, and how they rank under a multi-criteria scoring model. They
// are intentionally expensive and are meant to be run selectively rather
// than as part of routine CI.
//
// See also:
//   - [hashing] for the production hash primitives.
//   - [hashing/alternatives] for non-production hash implementations.
//   - [hashing/benchmarks] for shared benchmark utilities.
//   - [hashing/benchmark/performance] for runtime performance comparisons.
package quality
