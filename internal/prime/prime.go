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

func init() {
	// Generate all primes under 65536 using Sieve of Eratosthenes
	const limit = 65536
	isComposite := make([]bool, limit)
	for i := 2; i*i < limit; i++ {
		if !isComposite[i] {
			for j := i * i; j < limit; j += i {
				isComposite[j] = true
			}
		}
	}
	index := 0
	for i := 2; i < limit; i++ {
		if !isComposite[i] {
			primesUnder64k[index] = uint16(i)
			index++
		}
	}
	if index != len(primesUnder64k) {
		panic("unexpected number of primes under 65536")
	}
}

// primeTestDivisors returns a channel producing all candidate divisors in ascending order.
// The channel yields primes < 65536, then odd numbers that are not divisible by 3
// up to sqrt(candidate).
func primeTestDivisors(candidate uint64) <-chan uint64 {
	ch := make(chan uint64, 256)
	go func() {
		defer close(ch)
		last := uint64(primesUnder64k[len(primesUnder64k)-1])
		for _, p := range &primesUnder64k {
			sq := uint64(p) * uint64(p)
			if sq > candidate {
				return
			}
			ch <- uint64(p)
		}
		if candidate <= last*last {
			return
		}
		if last != 65521 {
			panic("last prime under 65536 should be 65521")
		}
		start := last + 2 // caution: 65521 is divisible by 3
		start += 2        // now start is odd and not divisible by 3
		for v := start; v*v <= candidate; {
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

	const lastPrimeUnder64k = uint64(65521)
	for v := lastPrimeUnder64k + 4; v*v <= x; {
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
