// Package alternatives provides non-production hash function implementations
// and experimental constants for the Set3 hash set.
//
// These functions are not used by the production [hashing.MakeRuntimeHasher]
// dispatch but are valuable for benchmarking, quality testing, and
// ranking comparisons. Each alternative is labelled with its strategy
// (SM = SplitMix, WH = WyHash non-deterministic, MH = maphash) so that
// benchmark code can reference them by name.
//
// See also:
//   - [hashing] for the production hash primitives.
//   - [hashing/benchmarks] for shared benchmark utilities.
//   - [hashing/benchmark/quality] for distribution and ranking tests.
//   - [hashing/benchmark/performance] for runtime performance comparisons.
package alternatives
