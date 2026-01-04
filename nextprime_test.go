package set3

import (
	"math"
	"testing"
)

// Test that primesUnder64k contains no zero entries (completely filled).
func TestPrimesUnder64k_Filled(t *testing.T) {
	for i, p := range primesUnder64k {
		if p == 0 {
			t.Fatalf("primesUnder64k has zero at index %d", i)
		}
	}
}

// Test that every value stored in primesUnder64k is actually prime.
func TestPrimesUnder64k_ArePrime(t *testing.T) {
	for _, p := range primesUnder64k {
		if !isPrimeNaive(uint64(p)) {
			t.Fatalf("value %d in primesUnder64k is not prime", p)
		}
	}
}

// Test that no prime < 65536 is missing from primesUnder64k.
func TestPrimesUnder64k_Complete(t *testing.T) {
	present := make(map[uint16]bool, len(primesUnder64k))
	for _, p := range primesUnder64k {
		present[p] = true
	}
	for i := 2; i < 65536; i++ {
		if isPrimeNaive(uint64(i)) {
			if !present[uint16(i)] {
				t.Fatalf("prime %d (<65536) missing from primesUnder64k", i)
			}
		}
	}
}

// Test that the primes in primesUnder64k are sorted ascending.
func TestPrimesUnder64k_Sorted(t *testing.T) {
	for i := 1; i < len(primesUnder64k); i++ {
		if primesUnder64k[i-1] > primesUnder64k[i] {
			t.Fatalf("primesUnder64k not sorted at index %d: %d >= %d", i-1, primesUnder64k[i-1], primesUnder64k[i])
		}
	}
}

// Test primeTestDivisors for small limits 0,1,2 (and basic behavior).
func TestPrimeTestDivisors_SmallLimits(t *testing.T) {
	cases := []struct {
		limit     uint64
		wantCount int
	}{
		{0, 0},
		{1, 0},
		{2, 1},
	}
	for _, c := range cases {
		cnt := 0
		for v := range primeTestDivisors(c.limit) {
			if v > c.limit {
				t.Fatalf("value %d > limit %d", v, c.limit)
			}
			cnt++
		}
		if cnt != c.wantCount {
			t.Fatalf("limit %d: expected %d values, got %d", c.limit, c.wantCount, cnt)
		}
	}
}

// Test that primeTestDivisors never yields values greater than the provided limit
// for a selection of limits (including 13 and 30 as requested).
func TestPrimeTestDivisors_NoValueGreaterThanSqrt(t *testing.T) {
	limits := []uint64{0, 1, 2, 13, 30, 100, 1000, 2413453}
	for _, lim := range limits {
		for v := range primeTestDivisors(lim) {
			if v > uint64(math.Sqrt(float64(lim))) {
				t.Fatalf("primeTestDivisors(%d) produced %d > %d", lim, v, uint64(math.Sqrt(float64(lim))))
			}
		}
	}
}

// Test that primeTestDivisors returns all primesUnder64k then the next 15 odd numbers
// when called with limit = lastPrime + 30.
func TestPrimeTestDivisors_LastPlus50(t *testing.T) {
	last := uint64(primesUnder64k[len(primesUnder64k)-1])
	candidate := (last + 50) * (last + 50)
	ch := primeTestDivisors(candidate)

	// First, consume and verify all primesUnder64k
	for i, p := range primesUnder64k {
		v, ok := <-ch
		if !ok {
			t.Fatalf("channel closed prematurely at prime index %d", i)
		}
		if v != uint64(p) {
			t.Fatalf("expected prime %d at index %d, got %d", p, i, v)
		}
	}

	// Then all remaining values up to limit should be odd and not divisible by 3
	for v := range ch {
		if v%2 == 0 {
			t.Fatalf("expected odd value after primes, got even %d", v)
		}
		if v%3 == 0 {
			t.Fatalf("value %d after primes is divisible by 3", v)
		}
		if v > candidate {
			t.Fatalf("value %d > limit %d", v, candidate)
		}
	}
}

func TestIsPrime_KnownValues(t *testing.T) {
	cases := []struct {
		n    uint64
		want bool
	}{
		{0, false}, {1, false}, {2, true}, {3, true}, {4, false},
		{5, true}, {9, false}, {11, true}, {15, false}, {17, true},
		{65521, true}, // last prime < 65536
	}
	for _, c := range cases {
		if got := isPrime(c.n); got != c.want {
			t.Fatalf("isPrime(%d) = %v, want %v", c.n, got, c.want)
		}
	}
}

func TestIsPrime_AgainstNaiveUpTo20000(t *testing.T) {
	for i := uint64(0); i <= 20000; i++ {
		got := isPrime(i)
		want := isPrimeNaive(i)
		if got != want {
			t.Fatalf("mismatch at %d: isPrime=%v isPrimeNaive=%v", i, got, want)
		}
	}
}

func TestIsPrime_Large32BitCases(t *testing.T) {
	// 4294967291 is a known prime near 2^32; 4294967290 is composite (even).
	if !isPrime(4294967291) {
		t.Fatalf("expected 4294967291 to be prime")
	}
	if isPrime(4294967290) {
		t.Fatalf("expected 4294967290 to be composite")
	}
}
func TestNextPrime_KnownValues(t *testing.T) {
	cases := []struct {
		in, want uint64
	}{
		{0, 2},
		{1, 2},
		{2, 2},
		{3, 3},
		{4, 5},
		{14, 17},
		{15, 17},
		{16, 17},
		{17, 17},
		{65520, 65521},
		{65521, 65521},
		{4294967290, 4294967291}, // known prime after 4294967290
	}
	for _, c := range cases {
		got := nextPrime(c.in)
		if got != c.want {
			t.Fatalf("nextPrime(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestNextPrime_MinimalitySmallRange(t *testing.T) {
	for n := uint64(0); n <= 1000; n++ {
		got := nextPrime(n)
		if !isPrimeNaive(got) {
			t.Fatalf("nextPrime(%d) returned non-prime %d", n, got)
		}
		for k := n; k < got; k++ {
			if isPrimeNaive(k) {
				t.Fatalf("nextPrime(%d) returned %d but smaller prime %d exists", n, got, k)
			}
		}
	}
}
