package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
)

type providerRequest struct {
	limit uint64
	resp  chan uint64
}

// - first 200 primes are embedded at compile-time in `first200Primes`.
// - `testCandidates(limit)` returns a channel that yields all embedded primes
//   up to `limit`, and if `limit` is larger than the largest embedded prime
//   it then yields odd numbers (not necessarily prime) up to `limit`.

var first200Primes = []uint64{
	2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
}

// testCandidates returns a channel producing candidate divisors up to limit.
// The channel yields embedded primes <= limit, then odd numbers > lastEmbedded up to limit.
func testCandidates(limit uint64) <-chan uint64 {
	ch := make(chan uint64, 32)
	go func() {
		defer close(ch)
		last := first200Primes[len(first200Primes)-1]
		for _, p := range first200Primes {
			if p > limit {
				return
			}
			ch <- p
		}
		if limit <= last {
			return
		}
		start := last + 1
		if start%2 == 0 {
			start++
		}
		for v := start; v <= limit; v += 2 {
			ch <- v
		}
	}()
	return ch
}

// isPrimeUsingCandidates tests x by consuming testCandidates up to sqrt(x).
func isPrimeUsingCandidates(x uint64) bool {
	if x < 2 {
		return false
	}
	if x == 2 {
		return true
	}
	if x%2 == 0 {
		return false
	}
	limit := uint64(math.Sqrt(float64(x)))
	for p := range testCandidates(limit) {
		if p > limit {
			break
		}
		if x%p == 0 {
			return false
		}
	}
	return true
}

func usageAndExit() {
	fmt.Fprint(os.Stderr, `Usage: primetest -m M
Find largest prime <= M.
Options:
	-m M     number to search (required)
`)
	os.Exit(2)
}

func largestPrimeLE(m uint64) (uint64, bool) {
	if m < 2 {
		return 0, false
	}
	for cand := m; cand >= 2; cand-- {
		if isPrimeUsingCandidates(cand) {
			return cand, true
		}
		if cand == 2 {
			break
		}
	}
	return 0, false
}

func main() {
	var mFlag string
	flag.StringVar(&mFlag, "m", "", "upper bound M (required)")
	flag.Parse()

	if mFlag == "" {
		usageAndExit()
	}
	mVal, err := strconv.ParseUint(mFlag, 10, 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid M: %v\n", err)
		os.Exit(2)
	}
	if res, ok := largestPrimeLE(mVal); ok {
		fmt.Println(res)
	} else {
		fmt.Printf("no prime <= %d\n", mVal)
	}
}
