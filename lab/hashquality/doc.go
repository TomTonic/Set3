//go:build set3lab

// Package hashquality measures how well hash functions spread values.
//
// The suites here check bucket distribution over many table sizes, compare
// avalanche behaviour, and rank the candidates in
// [github.com/TomTonic/Set3/lab/hashalt] against the production functions
// under a multi-criteria scoring model. Several of them run for tens of
// minutes; the most extreme ones only run when an environment variable asks
// for them explicitly.
//
// This package is behind the set3lab build tag; see lab/README.md.
package hashquality
