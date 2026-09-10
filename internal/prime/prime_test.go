package prime

import (
	"math"
	"testing"
)

func isPrimeNaive(n uint64) bool {
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
		{2, 0},
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
		got := Next(c.in)
		if got != c.want {
			t.Fatalf("Next(%d) = %d, want %d", c.in, got, c.want)
		}
	}
}

func TestNextPrime_MinimalitySmallRange(t *testing.T) {
	for n := uint64(0); n <= 1000; n++ {
		got := Next(n)
		if !isPrimeNaive(got) {
			t.Fatalf("Next(%d) returned non-prime %d", n, got)
		}
		for k := n; k < got; k++ {
			if isPrimeNaive(k) {
				t.Fatalf("Next(%d) returned %d but smaller prime %d exists", n, got, k)
			}
		}
	}
}

// --- The boundary between the table and open trial division ------------------
//
// Everything below 65521^2 is decided by the precomputed table alone. Above it,
// both isPrime and primeTestDivisors have to walk odd numbers themselves, in a
// +2/+4 stride that skips multiples of 3. That walk is what sizes every Set3
// larger than about 660 million groups, and until these tests it was the least
// exercised code in the library.

// tableLimit is the largest candidate the precomputed table can decide on its
// own: above it, some factor of the candidate may be larger than any entry.
const tableLimit = lastPrimeUnder64k * lastPrimeUnder64k // 4293001441

// TestSieveFindsExactlyTheExpectedNumberOfPrimes verifies the assumption the
// whole prime table rests on: that there are exactly 6542 primes below 65536,
// so the fixed-size array is neither short nor padded with zeros.
//
// init panics when this does not hold, which protects the program but cannot be
// observed from a test. sievePrimesUnder64k returns the count instead, so the
// invariant behind that panic can be checked here.
//
// It runs the sieve and requires the reported count to equal the array length.
func TestSieveFindsExactlyTheExpectedNumberOfPrimes(t *testing.T) {
	primes, found := sievePrimesUnder64k()
	if found != len(primes) {
		t.Fatalf("sieve found %d primes below 65536, array holds %d", found, len(primes))
	}
	if primes != primesUnder64k {
		t.Fatal("a fresh sieve disagrees with the table built at init")
	}
}

// TestSieveEndsAtTheHardcodedLastPrime verifies that lastPrimeUnder64k, which is
// written out as a literal, still matches the table the sieve produces.
//
// Both trial-division walks use the constant to decide where the table stops and
// their own stride begins. If the two ever drifted apart, the walks would skip
// or repeat divisors and primality answers would be silently wrong.
//
// It compares the constant against the last table entry, and additionally
// asserts the property that forces the walk to start at +4 rather than +2:
// lastPrimeUnder64k+2 is divisible by 3.
func TestSieveEndsAtTheHardcodedLastPrime(t *testing.T) {
	if got := uint64(primesUnder64k[len(primesUnder64k)-1]); got != lastPrimeUnder64k {
		t.Fatalf("table ends at %d, constant says %d", got, lastPrimeUnder64k)
	}
	if (lastPrimeUnder64k+2)%3 != 0 {
		t.Fatalf("%d is no longer divisible by 3; the +4 start of the divisor walk "+
			"was chosen to step over it and may now skip a divisor",
			lastPrimeUnder64k+2)
	}
	if firstTrialDivisorAbove64k%2 == 0 || firstTrialDivisorAbove64k%3 == 0 {
		t.Fatalf("divisor walk starts at %d, which is even or divisible by 3",
			firstTrialDivisorAbove64k)
	}
}

// TestPrimeTestDivisorsStopsAtTheTableWhenItSuffices verifies that no divisor
// beyond the precomputed table is produced for candidates the table can already
// decide, so the common case costs nothing extra.
//
// primeTestDivisors is the divisor source for trial division in this package.
// For any candidate up to 65521^2 every possible factor is at most 65521 and is
// therefore in the table.
//
// It asks for divisors of exactly 65521^2 and requires the stream to be the
// table and nothing else.
func TestPrimeTestDivisorsStopsAtTheTableWhenItSuffices(t *testing.T) {
	var got []uint64
	for v := range primeTestDivisors(tableLimit) {
		got = append(got, v)
	}
	if len(got) != len(primesUnder64k) {
		t.Fatalf("got %d divisors for the largest table-decidable candidate, want %d",
			len(got), len(primesUnder64k))
	}
	for i, p := range primesUnder64k {
		if got[i] != uint64(p) {
			t.Fatalf("divisor %d is %d, want %d", i, got[i], p)
		}
	}
}

// TestPrimeTestDivisorsWalksPastTheTableByExactlyOneStep verifies the narrowest
// case of the hand-rolled divisor walk: a candidate just large enough to need
// one divisor beyond the table, and no more.
//
// This pins the loop's exit condition in primeTestDivisors, which checks the
// next value's square before emitting it. Emitting one divisor too many is
// harmless; stopping one too early would let a composite pass as prime.
//
// It picks a candidate in the window [65525^2, 65527^2) and requires exactly one
// post-table divisor, namely 65525.
func TestPrimeTestDivisorsWalksPastTheTableByExactlyOneStep(t *testing.T) {
	const candidate = firstTrialDivisorAbove64k * firstTrialDivisorAbove64k // 65525^2

	var extra []uint64
	seen := 0
	for v := range primeTestDivisors(candidate) {
		seen++
		if seen > len(primesUnder64k) {
			extra = append(extra, v)
		}
	}
	if len(extra) != 1 || extra[0] != firstTrialDivisorAbove64k {
		t.Fatalf("divisors past the table = %v, want exactly [%d]", extra, firstTrialDivisorAbove64k)
	}
}

// TestPrimeTestDivisorsCoversEveryPossibleFactor verifies the property trial
// division actually depends on: that no prime which could divide the candidate
// is left out of the stream.
//
// The walk deliberately skips even numbers and multiples of 3 to save two
// thirds of the divisions. That is only safe as long as nothing prime is skipped
// with them, which is easy to get subtly wrong at the seam where the table ends.
//
// It takes candidates on both sides of that seam and requires every prime up to
// sqrt(candidate) to appear in the stream.
func TestPrimeTestDivisorsCoversEveryPossibleFactor(t *testing.T) {
	candidates := []uint64{
		1_000_000,
		tableLimit - 1,
		tableLimit,
		tableLimit + 1,
		firstTrialDivisorAbove64k * firstTrialDivisorAbove64k,
		4_300_000_000,
		4_294_967_291, // largest prime below 2^32
	}
	for _, c := range candidates {
		produced := make(map[uint64]bool)
		for v := range primeTestDivisors(c) {
			produced[v] = true
		}
		for p := uint64(2); p*p <= c; p++ {
			if !isPrimeNaive(p) {
				continue
			}
			if !produced[p] {
				t.Fatalf("primeTestDivisors(%d) never produced the prime %d, "+
					"which could divide the candidate", c, p)
			}
		}
	}
}

// TestIsPrimeDecidesNumbersWithFactorsAboveTheTable verifies that primality is
// still answered correctly once the precomputed table cannot decide it alone.
//
// A set grown past roughly 660 million groups asks for prime group counts in
// this range. Here isPrime falls back to its own +2/+4 divisor walk, whose two
// divisibility checks and early exit had no test coverage at all.
//
// It uses composites whose smallest factor sits above the table -- squares of
// the first primes past 65521 -- and primes chosen so that the walk terminates
// through each of its exits, and requires the expected answer for each.
func TestIsPrimeDecidesNumbersWithFactorsAboveTheTable(t *testing.T) {
	cases := []struct {
		n    uint64
		want bool
		why  string
	}{
		{65537 * 65537, false, "square of the first prime above the table"},
		{65539 * 65539, false, "square of the second prime above the table"},
		{65543 * 65543, false, "square of the third prime above the table"},
		{65551 * 65551, false, "square of the fourth prime above the table"},
		{65537 * 65539, false, "product of two distinct primes above the table"},
		{4293001443, false, "composite above the table with a small factor (3)"},
		{4293001469, true, "prime just above the table limit"},
		{4293525631, true, "prime whose walk exits through the mid-loop break"},
		{4294967291, true, "largest prime below 2^32"},
		{4294967295, false, "2^32-1 = 3 * 5 * 17 * 257 * 65537"},
	}
	for _, c := range cases {
		if got := isPrime(c.n); got != c.want {
			t.Fatalf("isPrime(%d) = %v, want %v (%s)", c.n, got, c.want, c.why)
		}
	}
}

// TestIsPrimeMatchesNaiveAboveTheTable verifies isPrime against a plain,
// obviously-correct trial division for a stretch of numbers just past the point
// where the optimized divisor walk takes over.
//
// The known-value test above can only cover the cases someone thought of. This
// one covers every number in a window straddling the seam, which is where an
// off-by-one in the walk's stride or start would show up.
//
// It compares isPrime against isPrimeNaive for 2000 consecutive candidates
// around 65521^2 and requires them to agree on every one.
func TestIsPrimeMatchesNaiveAboveTheTable(t *testing.T) {
	for n := tableLimit - 1000; n <= tableLimit+1000; n++ {
		if got, want := isPrime(n), isPrimeNaive(n); got != want {
			t.Fatalf("isPrime(%d) = %v, isPrimeNaive = %v", n, got, want)
		}
	}
}

// TestNextPrimeAboveTheTable verifies that Next still returns the smallest prime
// at or above its argument once the search leaves the precomputed table.
//
// Next is what turns a requested Set3 capacity into an allocated group count.
// Returning a composite would break the index reduction that assumes a prime
// group count; returning a needlessly large prime would waste memory.
//
// It asks for the next prime above several points past 65521^2 and requires the
// answer to be prime and to have no prime strictly between it and the argument.
func TestNextPrimeAboveTheTable(t *testing.T) {
	for _, n := range []uint64{tableLimit, tableLimit + 1, 4293525000, 4294967280} {
		got := Next(n)
		if !isPrimeNaive(got) {
			t.Fatalf("Next(%d) = %d, which is not prime", n, got)
		}
		for k := n; k < got; k++ {
			if isPrimeNaive(k) {
				t.Fatalf("Next(%d) = %d, but %d is a smaller prime at or above %d", n, got, k, n)
			}
		}
	}
}
