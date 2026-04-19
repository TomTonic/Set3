package benchmark

import (
	"fmt"
	"math"
	"math/bits"
	"strconv"
)

// FilteredNumbers returns a channel that emits natural numbers starting at 2
// that are prime, a power of two, or a multiple of 10/50/100. It emits
// exactly count values. This provides a diverse set of bucket counts for
// distribution quality testing.
func FilteredNumbers(count uint64) <-chan uint64 {
	ch := make(chan uint64)
	go func() {
		defer close(ch)
		if count <= 0 {
			return
		}
		emitted := uint64(0)
		for n := uint64(2); emitted < count; n++ {
			if n%100 == 0 || (n <= 500 && n%50 == 0) || (n <= 100 && n%10 == 0) || IsPowerOfTwo(n) || IsPrime(n) {
				ch <- n
				emitted++
			}
		}
	}()
	return ch
}

// IsPrime performs a naive trial-division primality test.
// Suitable for numbers used in benchmark bucket counts (typically < 1M).
func IsPrime(n uint64) bool {
	if n < 2 {
		return false
	}
	if n%2 == 0 {
		return n == 2
	}
	for i := uint64(3); i*i <= n; i += 2 {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// IsPowerOfTwo reports whether n is a positive power of two.
func IsPowerOfTwo(n uint64) bool {
	return n > 0 && (n&(n-1)) == 0
}

// NextPrime returns the smallest prime >= n using trial division.
func NextPrime(n uint64) uint64 {
	if n <= 2 {
		return 2
	}
	if n%2 == 0 {
		n++
	}
	for !IsPrime(n) {
		n += 2
	}
	return n
}

// Pow2String returns "2^k" if n is an exact power of two (n>0), otherwise the
// decimal representation of n. Works on uint64 inputs.
func Pow2String(n uint64) string {
	if n == 0 {
		return "0"
	}
	if n&(n-1) == 0 { // power of two
		k := bits.TrailingZeros64(n)
		return fmt.Sprintf("2^%d", k)
	}
	return strconv.FormatUint(n, 10)
}

// GetGroupIndex computes the group index for a given hash value and group count
// using the "multiply-high" method: hi, _ = bits.Mul64(hash, groupCount).
// This mirrors the approach used by the Set3 hash set for bucket selection.
func GetGroupIndex(hash, groupCount uint64) uint64 {
	hi, _ := bits.Mul64(hash, groupCount)
	return hi
}

// ComputeRelStdDevAndMaxDev calculates the relative standard deviation and
// maximum relative deviation for a slice of bucket counts. Both metrics
// are expressed relative to the mean count.
func ComputeRelStdDevAndMaxDev(counts []uint32) (relStdDev, maxRelDev float64) {
	if len(counts) == 0 {
		return 0, 0
	}
	var sum uint64
	for _, c := range counts {
		sum += uint64(c)
	}
	mean := float64(sum) / float64(len(counts))
	if mean == 0 {
		return 0, 0
	}
	var sqsum float64
	for _, c := range counts {
		d := float64(c) - mean
		sqsum += d * d
		if rd := math.Abs(d) / mean; rd > maxRelDev {
			maxRelDev = rd
		}
	}
	stddev := math.Sqrt(sqsum / float64(len(counts)))
	relStdDev = stddev / mean
	return relStdDev, maxRelDev
}

// UniqueSortedUint32s deduplicates a sorted slice of uint32 in place.
func UniqueSortedUint32s(s []uint32) []uint32 {
	if len(s) <= 1 {
		return s
	}
	w := 1
	for r := 1; r < len(s); r++ {
		if s[r] != s[r-1] {
			s[w] = s[r]
			w++
		}
	}
	return s[:w]
}

// --- XorShift PRNG types for test-data generation ---

// XorShift16Star is a 16-bit xorshift* generator with full 2^16-1 period.
type XorShift16Star struct {
	State uint16
}

// Uint16 advances the generator and returns the next value.
func (x *XorShift16Star) Uint16() uint16 {
	s := x.State
	s ^= s << 7
	s ^= s >> 9
	s ^= s << 8
	x.State = s
	return s * 0x9E37 // xorshift16*
}

// XorShift24Star is a 24-bit xorshift* generator (state held in low 24 bits of uint32).
type XorShift24Star struct {
	State uint32
}

const mask24 = (1 << 24) - 1

// Uint24 advances the generator and returns the next 24-bit value.
func (x *XorShift24Star) Uint24() uint32 {
	s := x.State & mask24
	s ^= (s << 7) & mask24
	s ^= s >> 13
	s ^= (s << 11) & mask24
	x.State = s
	return (s * 0x9E3779) & mask24 // xorshift24*
}

// XorShift32Star is a 32-bit xorshift* generator with full 2^32-1 period.
type XorShift32Star struct {
	State uint32
}

// Uint32 advances the generator and returns the next value.
func (x *XorShift32Star) Uint32() uint32 {
	s := x.State
	s ^= s << 13
	s ^= s >> 17
	s ^= s << 5
	x.State = s
	return s * 0x9E3779B9 // xorshift32*
}
