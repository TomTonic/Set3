//go:build set3lab && !race

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the Go distribution's LICENSE file.
//
// This file is a port of hash/maphash/smhasher_test.go from the Go standard
// library, retargeted from maphash onto Set3's hashing package. The test bodies,
// the key families and the acceptance bounds are the Go authors' work and are
// deliberately left as they found them; what changed is which function is under
// test, the removal of dependencies on internal packages, and the build tag that
// keeps it out of the everyday test cycle.
//
// Smhasher itself is Austin Appleby's torture test for hash functions
// (https://github.com/aappleby/smhasher); the Go file is a port of part of it.

package hashquality

import (
	"fmt"
	"math"
	"math/rand"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

// Why this suite is here rather than a home-grown one
//
// The hashing package carries its own structural tests: the read window, the
// seed response, degenerate operands, block order, avalanche, chi-squared over
// the H2 tag and the group index. Those were written against a specific
// implementation and by the same hand that wrote it, which is the weakest
// possible position from which to look for a flaw.
//
// This suite was not. It is the set of key families that broke real hash
// functions in the wild — text, cyclic repeats, sparse bit patterns, block
// permutations, windowed rotations — measured against a collision bound derived
// from the birthday paradox rather than from a threshold someone picked. A hash
// that passes it is not proven good, but every cheap way of being bad has been
// tried on it by people who were not trying to confirm it works.

// fixedSeed stands in for maphash's process seed. It is fixed so that a failure
// is reproducible; the seed-sensitivity of the hash is what TestSmhasherSeed
// covers.
const fixedSeed = uint64(0x243f6a8885a308d3)

func bytesHash(b []byte) uint64 {
	return hashing.HashBytesBlock(fixedSeed, b)
}

func stringHash(s string) uint64 {
	return hashing.HashString(unsafe.Pointer(&s), fixedSeed) //nolint:gosec
}

func seededStringHash(s string, seed uint64) uint64 {
	return hashing.HashString(unsafe.Pointer(&s), seed) //nolint:gosec
}

const hashSize = 64

func randBytes(r *rand.Rand, b []byte) {
	r.Read(b) //nolint:errcheck,gosec // cannot fail
}

// Sanity checks: the hash must not depend on bytes outside the key, and must
// not depend on how the key is aligned.
func TestSmhasherSanity(t *testing.T) {
	t.Parallel()
	r := rand.New(rand.NewSource(1234)) //nolint:gosec
	const REP = 10
	const KEYMAX = 128
	const PAD = 16
	const OFFMAX = 16
	for range REP {
		for n := range KEYMAX {
			for i := range OFFMAX {
				var b [KEYMAX + OFFMAX + 2*PAD]byte
				var c [KEYMAX + OFFMAX + 2*PAD]byte
				randBytes(r, b[:])
				randBytes(r, c[:])
				copy(c[PAD+i:PAD+i+n], b[PAD:PAD+n])
				if bytesHash(b[PAD:PAD+n]) != bytesHash(c[PAD+i:PAD+i+n]) {
					t.Fatalf("hash depends on bytes outside the key (n=%d, offset=%d)", n, i)
				}
			}
		}
	}
}

// A hashSet measures the frequency of hash collisions.
type hashSet struct {
	list []uint64
}

func newHashSet() *hashSet {
	return &hashSet{list: make([]uint64, 0, 1024)}
}

func (s *hashSet) add(h uint64)  { s.list = append(s.list, h) }
func (s *hashSet) addS(x string) { s.add(stringHash(x)) }
func (s *hashSet) addB(x []byte) { s.add(bytesHash(x)) }
func (s *hashSet) addSeeded(x string, seed uint64) {
	s.add(seededStringHash(x, seed))
}

// check compares the observed number of collisions against what a uniformly
// random 64-bit hash would produce over the same number of keys.
func (s *hashSet) check(t *testing.T) {
	t.Helper()
	list := s.list
	slices.Sort(list)

	collisions := 0
	for i := 1; i < len(list); i++ {
		if list[i] == list[i-1] {
			collisions++
		}
	}
	n := len(list)

	const SLOP = 10.0
	pairs := int64(n) * int64(n-1) / 2
	expected := float64(pairs) / math.Pow(2.0, float64(hashSize))
	stddev := math.Sqrt(expected)
	if float64(collisions) > expected+SLOP*(3*stddev+1) {
		t.Errorf("unexpected number of collisions: got=%d mean=%f stddev=%f", collisions, expected, stddev)
	}
	s.list = s.list[:0]
}

// A string plus appended zeros must make distinct hashes.
func TestSmhasherAppendedZeros(t *testing.T) {
	t.Parallel()
	s := "hello" + strings.Repeat("\x00", 256)
	h := newHashSet()
	for i := 0; i <= len(s); i++ {
		h.addS(s[:i])
	}
	h.check(t)
}

// All 0-3 byte strings have distinct hashes.
func TestSmhasherSmallKeys(t *testing.T) {
	t.Parallel()
	h := newHashSet()
	var b [3]byte
	for i := range 256 {
		b[0] = byte(i) //nolint:gosec
		h.addB(b[:1])
		for j := range 256 {
			b[1] = byte(j) //nolint:gosec
			h.addB(b[:2])
			if !testing.Short() {
				for k := range 256 {
					b[2] = byte(k) //nolint:gosec
					h.addB(b[:3])
				}
			}
		}
	}
	h.check(t)
}

// Different length strings of all zeros have distinct hashes.
func TestSmhasherZeros(t *testing.T) {
	t.Parallel()
	N := 256 * 1024
	if testing.Short() {
		N = 1024
	}
	h := newHashSet()
	b := make([]byte, N)
	for i := 0; i <= N; i++ {
		h.addB(b[:i])
	}
	h.check(t)
}

// Strings with up to two nonzero bytes all have distinct hashes.
func TestSmhasherTwoNonzero(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	h := newHashSet()
	for n := 2; n <= 16; n++ {
		twoNonZero(h, n)
	}
	h.check(t)
}

func twoNonZero(h *hashSet, n int) {
	b := make([]byte, n)
	h.addB(b)
	for i := range n {
		for x := 1; x < 256; x++ {
			b[i] = byte(x) //nolint:gosec
			h.addB(b)
			b[i] = 0
		}
	}
	for i := range n {
		for x := 1; x < 256; x++ {
			b[i] = byte(x) //nolint:gosec
			for j := i + 1; j < n; j++ {
				for y := 1; y < 256; y++ {
					b[j] = byte(y) //nolint:gosec
					h.addB(b)
					b[j] = 0
				}
			}
			b[i] = 0
		}
	}
}

// Strings with repeats, like "abcdabcdabcdabcd...".
func TestSmhasherCyclic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	r := rand.New(rand.NewSource(1234)) //nolint:gosec
	const REPEAT = 8
	const N = 1000000
	h := newHashSet()
	for n := 4; n <= 12; n++ {
		b := make([]byte, REPEAT*n)
		for i := range N {
			b[0] = byte(i * 79 % 97)   //nolint:gosec
			b[1] = byte(i * 43 % 137)  //nolint:gosec
			b[2] = byte(i * 151 % 197) //nolint:gosec
			b[3] = byte(i * 199 % 251) //nolint:gosec
			randBytes(r, b[4:n])
			for j := n; j < n*REPEAT; j++ {
				b[j] = b[j-n]
			}
			h.addB(b)
		}
		h.check(t)
	}
}

// Strings with only a few bits set.
func TestSmhasherSparse(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	h := newHashSet()
	sparse(t, h, 32, 6)
	sparse(t, h, 40, 6)
	sparse(t, h, 48, 5)
	sparse(t, h, 56, 5)
	sparse(t, h, 64, 5)
	sparse(t, h, 96, 4)
	sparse(t, h, 256, 3)
	sparse(t, h, 2048, 2)
}

func sparse(t *testing.T, h *hashSet, n int, k int) {
	t.Helper()
	b := make([]byte, n/8)
	setbits(h, b, 0, k)
	h.check(t)
}

// setbits sets up to k bits at index i and greater.
func setbits(h *hashSet, b []byte, i int, k int) {
	h.addB(b)
	if k == 0 {
		return
	}
	for j := i; j < len(b)*8; j++ {
		b[j/8] |= byte(1 << uint(j&7))
		setbits(h, b, j+1, k-1)
		b[j/8] &= byte(^(1 << uint(j&7)))
	}
}

// All possible combinations of n blocks from the set s.
func TestSmhasherPermutation(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	h := newHashSet()
	permutation(t, h, []uint32{0, 1, 2, 3, 4, 5, 6, 7}, 8)
	permutation(t, h, []uint32{0, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 8)
	permutation(t, h, []uint32{0, 1}, 20)
	permutation(t, h, []uint32{0, 1 << 31}, 20)
	permutation(t, h, []uint32{0, 1, 2, 3, 4, 5, 6, 7, 1 << 29, 2 << 29, 3 << 29, 4 << 29, 5 << 29, 6 << 29, 7 << 29}, 6)
}

func permutation(t *testing.T, h *hashSet, s []uint32, n int) {
	t.Helper()
	b := make([]byte, n*4)
	genPerm(h, b, s, 0)
	h.check(t)
}

func genPerm(h *hashSet, b []byte, s []uint32, n int) {
	h.addB(b[:n])
	if n == len(b) {
		return
	}
	for _, v := range s {
		b[n] = byte(v)         //nolint:gosec
		b[n+1] = byte(v >> 8)  //nolint:gosec
		b[n+2] = byte(v >> 16) //nolint:gosec
		b[n+3] = byte(v >> 24) //nolint:gosec
		genPerm(h, b, s, n+4)
	}
}

type bytesKey struct{ b []byte }

func (k *bytesKey) clear()              { clear(k.b) }
func (k *bytesKey) random(r *rand.Rand) { randBytes(r, k.b) }
func (k *bytesKey) bits() int           { return len(k.b) * 8 }
func (k *bytesKey) flipBit(i int)       { k.b[i>>3] ^= byte(1 << uint(i&7)) }
func (k *bytesKey) hash() uint64        { return bytesHash(k.b) }
func (k *bytesKey) name() string        { return fmt.Sprintf("bytes%d", len(k.b)) }

// Flipping a single bit of a key should flip each output bit with 50%
// probability.
func TestSmhasherAvalanche(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	for _, n := range []int{2, 4, 8, 16, 32, 200} {
		avalancheTest1(t, &bytesKey{make([]byte, n)})
	}
}

func avalancheTest1(t *testing.T, k *bytesKey) {
	t.Helper()
	const REP = 100000
	r := rand.New(rand.NewSource(1234)) //nolint:gosec
	n := k.bits()

	// grid[i][j] counts whether flipping input bit i affects output bit j.
	grid := make([][hashSize]int, n)

	for range REP {
		k.random(r)
		h := k.hash()
		for i := range n {
			k.flipBit(i)
			d := h ^ k.hash()
			k.flipBit(i)
			g := &grid[i]
			for j := range hashSize {
				g[j] += int(d & 1) //nolint:gosec
				d >>= 1
			}
		}
	}

	// Each entry should be about REP/2. Find bounds such that a truly random
	// experiment would fall inside them with probability .9999 over all N
	// cells, then allow the slack the Go suite allows.
	N := n * hashSize
	var c float64
	for c = 0.0; math.Pow(math.Erf(c/math.Sqrt(2)), float64(N)) < .9999; c += .1 { //nolint:revive // empty body is the search
	}
	c *= 11.0
	mean := .5 * REP
	stddev := .5 * math.Sqrt(REP)
	low := int(mean - c*stddev)
	high := int(mean + c*stddev)
	for i := range n {
		for j := range hashSize {
			x := grid[i][j]
			if x < low || x > high {
				t.Errorf("bad bias for %s bit %d -> bit %d: %d/%d", k.name(), i, j, x, REP)
			}
		}
	}
}

// All bit rotations of a set of distinct keys.
func TestSmhasherWindowed(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	windowed(t, &bytesKey{make([]byte, 128)})
}

func windowed(t *testing.T, k *bytesKey) {
	t.Helper()
	const BITS = 16
	h := newHashSet()
	for r := range k.bits() {
		for i := range 1 << BITS {
			k.clear()
			for j := range BITS {
				if i>>uint(j)&1 != 0 {
					k.flipBit((j + r) % k.bits())
				}
			}
			h.add(k.hash())
		}
		h.check(t)
	}
}

// All keys of the form prefix + [A-Za-z0-9]*N + suffix.
func TestSmhasherText(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	h := newHashSet()
	text(t, h, "Foo", "Bar")
	text(t, h, "FooBar", "")
	text(t, h, "", "FooBar")
}

func text(t *testing.T, h *hashSet, prefix, suffix string) {
	t.Helper()
	const N = 4
	const S = "ABCDEFGHIJKLMNOPQRSTabcdefghijklmnopqrst0123456789"
	const L = len(S)
	b := make([]byte, len(prefix)+N+len(suffix))
	copy(b, prefix)
	copy(b[len(prefix)+N:], suffix)
	c := b[len(prefix):]
	for i := range L {
		c[0] = S[i]
		for j := range L {
			c[1] = S[j]
			for k := range L {
				c[2] = S[k]
				for x := range L {
					c[3] = S[x]
					h.addB(b)
				}
			}
		}
	}
	h.check(t)
}

// Different seed values must generate different hashes.
//
// This is the property Set3 leans on hardest: a rehash triggered by a collision
// pattern draws a fresh seed specifically to break that pattern up, which only
// works if the seed reaches the output everywhere.
func TestSmhasherSeed(t *testing.T) {
	t.Parallel()
	h := newHashSet()
	const N = 100000
	s := "hello"
	for i := range N {
		h.addSeeded(s, uint64(i+1))     //nolint:gosec
		h.addSeeded(s, uint64(i+1)<<32) //nolint:gosec // make sure high bits are used
	}
	h.check(t)
}
