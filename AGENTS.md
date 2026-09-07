# AI Agent Guidelines

## Project Overview

Set3 is a high-performance, native Go set implementation — 10-30% faster and using 25% less memory than `map[type]struct{}`. Originally forked from `dolthub/swiss`, it has since been substantially reworked and is actively maintained as its own project.

## Build & Test Commands

```bash
go build ./...          # Build
go test ./...           # Run tests
go test ./... -race     # Run tests with race detector
go test ./... -cover    # Run tests with coverage
golangci-lint run       # Run linter
```

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`).
- Use `golangci-lint` (default ruleset; this repo has no custom `.golangci.yml`).
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

Unexported helpers do not require full documentation, but a one-line
comment explaining *why* the helper exists is expected.

## Testing Requirements

- All new functionality must include tests.
- Use table-driven tests where appropriate.
- Maintain high test coverage; CI tracks it via a gist-based badge.
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
- All dependencies are kept current via Dependabot.
- Run `go mod tidy` after adding or removing dependencies.
- Do not add dependencies with known vulnerabilities.

## Security

- Never commit secrets, credentials, or API keys.
- Static analysis runs via `golangci-lint` in CI (default ruleset — no custom `.golangci.yml` in this repo).
- Validate all external input at system boundaries.
- Use `crypto/rand` for security-sensitive randomness, not `math/rand`.

## CI/CD

- All pushes and PRs are checked by: golangci-lint (`lint.yml`), `go vet`, `go test`, CodeQL, dependency review, and OpenSSF Scorecard.
- Coverage is tracked via a gist-based badge.
- Fuzz testing runs via `fuzz.yml`.
- Dependency updates are automated via Dependabot.

## File Organization

- Keep the top-level package clean; use subdirectories for internal packages.
- Test files live next to the code they test (`foo_test.go` next to `foo.go`).
- `NOTICE` records attribution for code inherited from the upstream `dolthub/swiss` fork.

## Release Notes

Handled by the `release-notes` agent (symlinked from `go-project-defaults`'s
`.claude/agents/release-notes.md` into `~/.claude/agents/` — see
[go-project-defaults's README](https://github.com/TomTonic/go-project-defaults#claude-code-agents-symlink-into-claudeagents)
for setup). Project-specific parameters (core dependency, tag scheme
quirks, pre-vetted non-applicable advisories) live in
`.claude/release-notes.yml` — copy `release-notes.example.yml` from
[go-project-defaults](https://github.com/TomTonic/go-project-defaults) and
customize.
