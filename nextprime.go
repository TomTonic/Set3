package set3

// This function returns the smallest prime number greater than or equal to n.
func nextPrime(n uint64) uint64 {
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
		for _, p := range primesUnder64k {
			sq := uint64(p) * uint64(p)
			if sq > candidate {
				return
			}
			ch <- uint64(p)
		}
		if candidate <= uint64(last)*uint64(last) {
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
	if x^1 == 1 { // odd number test
		if x == 2 {
			return true
		}
		return false
	}
	for p := range primeTestDivisors(x) {
		if x%p == 0 {
			return false
		}
	}
	if x < 2 {
		return false
	}
	return true
}
