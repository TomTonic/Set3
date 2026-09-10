// Package prime provides the primality helpers Set3 uses to size its control
// table. Set3 always allocates a prime number of groups, because reducing a
// 64-bit hash modulo a prime spreads the high bits far better than a
// power-of-two mask does, which keeps the probe sequences short.
//
// The implementation is trial division with a compile-time-generated table of
// all primes below 65536. That is fast enough for the sizes a set is grown to
// and needs no external dependency.
package prime

// Next returns the smallest prime number greater than or equal to n.
//
// n may be any uint64; values below 2 return 2. The result is the group count
// Set3 allocates for a requested capacity, so callers pass the minimum number
// of groups they need and take the next prime at or above it.
func Next(n uint64) uint64 {
	if n <= 2 {
		return 2
	}

	if isPrime(n) {
		return n
	}

	start := n + 1
	if start%2 == 0 {
		start++
	}
	for candidate := start; ; candidate += 2 {
		if isPrime(candidate) {
			return candidate
		}
	}
}

// The first 6542 primes are less than 65536 and can be stored in uint16 values.
// These values are cached in primesUnder64k for fast trial division:
// These values are the most probable divisors, as small divisors occur more
// frequently in natural numbers. As we only need to test 6542 divisors instead
// of 32768 divisors (all odd numbers) up to 65536, we only need about 20% of the
// number of trial divisions for numbers up to 2^32.
var primesUnder64k = [6542]uint16{}

// lastPrimeUnder64k is the final entry of primesUnder64k.
//
// It is written out as a constant rather than read from the table because both
// trial divisions below need to know where the table ends before they can walk
// past it. TestSieveEndsAtTheHardcodedLastPrime keeps the two in agreement.
const lastPrimeUnder64k = uint64(65521)

// firstTrialDivisorAbove64k is where trial division continues once the table is
// exhausted: the first odd number above the table that 3 does not divide.
//
// The obvious choice, lastPrimeUnder64k+2, is 65523 = 3 * 21841, so the walk
// would start on a divisor it is trying to skip. Starting two further along
// puts it on 65525 and keeps the +2/+4 stride free of multiples of 3.
const firstTrialDivisorAbove64k = lastPrimeUnder64k + 4

func init() {
	primes, found := sievePrimesUnder64k()
	if found != len(primesUnder64k) {
		// Unreachable: there are exactly 6542 primes below 65536 and the sieve
		// finds all of them, which TestSieveFindsExactlyTheExpectedNumberOfPrimes
		// checks directly. Kept because a table that is short or padded with
		// zeros would not fail here, it would quietly answer primality wrong for
		// the rest of the program's life.
		panic("unexpected number of primes under 65536")
	}
	primesUnder64k = primes
}

// sievePrimesUnder64k runs a Sieve of Eratosthenes over [2, 65536) and returns
// the primes it found in ascending order, together with how many there were.
//
// found is returned instead of being asserted here so that a test can check the
// count that init relies on; init turns a wrong count into a panic, which is the
// only sensible response at program start but not something a test can observe.
func sievePrimesUnder64k() (primes [6542]uint16, found int) {
	const limit = 65536
	isComposite := make([]bool, limit)
	for i := 2; i*i < limit; i++ {
		if !isComposite[i] {
			for j := i * i; j < limit; j += i {
				isComposite[j] = true
			}
		}
	}
	for i := 2; i < limit; i++ {
		if !isComposite[i] {
			if found < len(primes) {
				primes[found] = uint16(i)
			}
			found++
		}
	}
	return primes, found
}

// primeTestDivisors returns a channel producing all candidate divisors in ascending order.
// The channel yields primes < 65536, then odd numbers that are not divisible by 3
// up to sqrt(candidate).
func primeTestDivisors(candidate uint64) <-chan uint64 {
	ch := make(chan uint64, 256)
	go func() {
		defer close(ch)
		for _, p := range &primesUnder64k {
			sq := uint64(p) * uint64(p)
			if sq > candidate {
				return
			}
			ch <- uint64(p)
		}
		if candidate <= lastPrimeUnder64k*lastPrimeUnder64k {
			return
		}
		for v := firstTrialDivisorAbove64k; v*v <= candidate; {
			ch <- v
			v += 2
			if v*v > candidate {
				break
			}
			ch <- v
			// skip v += 2 is divisible by 3
			v += 4 // skip multiples of 3 to spare another third of trial divisions
		}
	}()
	return ch
}

func isPrime(x uint64) bool {
	if x < 2 {
		return false
	}
	if x == 2 {
		return true
	}
	if x&1 == 0 {
		return false
	}

	for _, p := range &primesUnder64k {
		pp := uint64(p)
		sq := pp * pp
		if sq > x {
			return true
		}
		if x%pp == 0 {
			return x == pp
		}
	}

	for v := firstTrialDivisorAbove64k; v*v <= x; {
		if x%v == 0 {
			return false
		}
		v += 2
		if v*v > x {
			break
		}
		if x%v == 0 {
			return false
		}
		// Skip the next odd value which is divisible by 3.
		v += 4
	}

	return true
}
