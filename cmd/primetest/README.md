# primetest

Purpose
- Finds the largest prime less than or equal to a given number `M`.
- Uses a background `PrimeProvider` that keeps a small cache of the first N primes and expands it on demand up to a configured maximum.

Build

```sh
go build ./cmd/primetest
```

Usage

```sh
./primetest -m M [-n N] [-max MAX]
```

Examples

- Find the largest prime <= 1_000_000 using the default cache:

```sh
./primetest -m 1000000
```

- Precompute 200 primes in the cache and allow the cache to expand up to 100000:

```sh
./primetest -m 1000000 -n 200 -max 100000
```

Flags

- `-m M` (required): upper bound M (search for largest prime <= M).
- `-n N` (optional): number of primes to precompute in the provider cache (default: 100).
- `-max MAX` (optional): maximum prime value to which the cache may be expanded (default: 100000). If a request requires divisors beyond `MAX`, the provider will return odd numbers (not necessarily prime) as a fallback.

How it works (brief)

- `PrimeProvider` runs in the background and handles request/response pairs: for each request (`limit`) it opens a response channel and first sends all cached primes ≤ min(limit, max).
- If the cache is insufficient, it extends the cache up to min(limit, max) and sends newly found primes on the response channel as they are discovered.
- If `limit` is greater than `max`, the provider will supply odd numbers (not necessarily primes) above `max` so that trial division checks can continue without unbounded cache growth.

Note

- This program is intended as a practical tool for 64 bit values. For very large `M` values consider more efficient algorithms (segmented sieves, deterministic Miller–Rabin bases, etc.).
