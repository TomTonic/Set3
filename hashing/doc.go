// Package hashing provides the hash function infrastructure for the Set3 hash
// set implementation.
//
// It contains:
//   - [HashFunction]: the raw hash function signature used by all hash routines.
//   - [RuntimeHasher]: a per-type, per-seed hasher that dispatches to an optimized
//     hash function chosen at construction time by [MakeRuntimeHasher].
//   - Production-grade hash implementations for all Go primitive types (bool,
//     integer types, floats, strings, byte slices, and fixed-size arrays).
//   - [RawByteBlockEligibility] analysis to determine whether a type's in-memory
//     layout is safe for raw byte-block hashing.
//
// All mathematical constants used across the hashing strategy (golden ratio,
// sqrt(2)-1, (pi+e)/7 series at various bit widths) are defined here as a
// coherent reference set. Production code uses a subset; the remaining
// constants exist for the candidates and comparison suites under lab/.
//
// The hash functions that were considered and rejected, and the suites that
// measure distribution quality and runtime against them, live under lab/ and
// are compiled only with the set3lab build tag. See lab/README.md.
package hashing
