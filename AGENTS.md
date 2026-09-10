# AI Agent Guidelines

## Project Overview

Set3 is a Go set implementation built on the Abseil "Swiss table" layout, not
on `map[T]struct{}`. It is a single-module Go project with no build steps
beyond the Go tool.

```text
set3.go            the library: the Set3 type and every operation on it
set3_test.go       behaviour tests   set3_tombstone_test.go  probe-chain tests
set3_fuzz_test.go  fuzz targets
hashing/           hash function selection and the per-type hash routines
internal/prime/    the primality search that sizes the control table
lab/               experiments and long-running measurements
```

## The lab/ directory

`lab/` holds experimental hash functions, benchmark harnesses, and test suites
that run for tens of minutes to hours. Every file there starts with

```go
//go:build set3lab
```

so the Go tool ignores it unless the tag is passed. This is deliberate: the
everyday build and test cycle must stay fast, but the code must stay compiled,
linted, and runnable. Do not delete experiment code to "clean up" — move it to
`lab/` and tag it. Do not un-tag anything in `lab/`. See `lab/README.md`.

## Build & Test Commands

```bash
go build ./...          # Build (lab/ excluded)
go test ./...           # Run tests (seconds)
go test ./... -race     # Run tests with race detector
go test ./... -cover    # Run tests with coverage
golangci-lint run       # Run linter — covers lab/ too, the tag is in .golangci.yml

go build -tags set3lab ./...            # compile lab/ as well
go test -tags set3lab -short ./lab/...  # run every lab suite, cheap paths only
go test -tags set3lab ./lab/...         # the full experiments: hours
```

`go vet ./...` reports one known finding, `unsafeptr` in
`hashing/hasher.go`. That line is the pointer-laundering idiom `Noescape` exists
for; golangci-lint has it suppressed with a reason. Use `golangci-lint run`
rather than bare `go vet`.

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`).
- Use `golangci-lint` with the project's `.golangci.yml` configuration.
- Keep functions focused and under 60 lines where practical.
- Prefer returning errors over panicking.
- Use Go's standard error wrapping: `fmt.Errorf("context: %w", err)`.
- Do not use `panic()` in library code.

### Function and Method Documentation

Every exported function and method must have a godoc comment. Write it
like a good JavaDoc entry but with more emphasis on **context and usage
guidance** than a pure specification:

1. **First sentence**: A concise summary of what the function does,
   starting with the function name (Go convention).
2. **Parameters**: Document each parameter — its type, valid ranges,
   and what it controls.
3. **Return values**: What is returned on success and on error.
4. **Usage context**: When and why a caller would use this function.
   Mention typical call sites, related functions, or common patterns.
5. **Example** (optional but encouraged): A short inline example or
   reference to a testable example (`Example*` function).

Example:

```go
// EmptyWithCapacity creates a new, empty Set3 with room for at least
// initialCapacity elements.
//
// initialCapacity is a hint, not a limit: the set grows on demand, it
// just rehashes on the way. Pass the number of elements you expect to
// add when you know it, and the set never rehashes while filling.
//
// Returns a set that is ready to use; there is no error case.
//
// Use this over Empty whenever the size is known up front — filling a
// set that starts at the default capacity of 21 rehashes several times.
// To resize a set that already holds elements, see Set3.RehashToCapacity.
func EmptyWithCapacity[T comparable](initialCapacity uint32) *Set3[T] { ... }
```

Unexported helpers do not require full documentation, but a one-line
comment explaining *why* the helper exists is expected.

## Testing Requirements

- All new functionality must include tests.
- Use table-driven tests where appropriate.
- Maintain at least 98% statement coverage; the coverage workflow enforces it.
- Run `go test ./... -race` before submitting changes.
- Fuzz tests are welcome for functions that parse external input.

### Test Documentation

Every test function must have a doc comment that reads **outside-in**.
Structure the comment in this order:

1. **User perspective**: What does the tested code achieve for the end user,
   described in the user's own terminology? Avoid implementation jargon.
2. **Context**: Which module, package, or feature area does the tested code
   belong to? How does it fit into the larger system?
3. **Concrete expectation**: What specific behavior is this test verifying?

Example:

```go
// TestAddReusesTombstoneWithoutBreakingOverflowProbe verifies that a set
// still finds every element it contains after elements have been removed
// and new ones added in their place — the case where a naive Swiss table
// silently loses entries.
//
// This covers the probe-chain handling in Set3.Add, which may reuse the
// slot of a deleted element only when doing so cannot cut a probe chain
// that a later element depends on.
//
// It builds a group whose slots all map to the same start bucket, removes
// one element from the middle of the chain, adds a new element that lands
// in the freed slot, and asserts that every remaining element is still
// found.
func TestAddReusesTombstoneWithoutBreakingOverflowProbe(t *testing.T) { ... }
```

For table-driven tests, document the overall test function with the
outside-in structure and give each sub-test case a descriptive name
that reads as an assertion (e.g. `"returns error for empty input"`).

## Commit Messages

- Use imperative mood ("Add feature", not "Added feature").
- Limit subject line to 72 characters.
- Prefix dependency updates with `deps-upd:`.
- Separate subject from body with a blank line.

## Dependencies

- Minimize external dependencies.
- All dependencies are managed via Renovate (see `.github/renovate.json`).
- Run `go mod tidy` after adding or removing dependencies.
- Do not add dependencies with known vulnerabilities.

## Security

- Never commit secrets, credentials, or API keys.
- The `gosec` linter is enabled — do not disable it.
- Validate all external input at system boundaries.
- Use `crypto/rand` for security-sensitive randomness, not `math/rand`.

## CI/CD

- Every push is checked by: golangci-lint, the test/coverage run, the fuzz
  targets, CodeQL, and dependency review.
- Coverage is tracked via badges pushed to the `badges` branch.
- `lab/` is built, vetted and smoke-tested weekly by `.github/workflows/lab.yml`,
  and on demand via workflow_dispatch. Nothing else compiles it, so that job is
  the only thing standing between the experiments and bit rot.
- Dependency updates are automated via Renovate with automerge for patches
  and minor updates.
- OpenSSF Scorecard runs weekly.

## File Organization

- Keep the repository root minimal. `set3.go` and its test files are the only
  Go files that belong there; everything else goes into a subpackage.
- Test files live next to the code they test (`foo_test.go` next to `foo.go`).
- Experimental code, benchmark drivers, and suites that run longer than a few
  seconds belong in `lab/` behind the `set3lab` build tag, never in the
  default build.
- Community files (`CONTRIBUTING.md`, `SECURITY.md`) and tool configuration
  that GitHub accepts there live in `.github/`.
