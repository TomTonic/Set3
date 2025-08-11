# Contributing

(This text is derived from the [OSSF Best Practice Badge CONTRIBUTING rules](https://github.com/coreinfrastructure/best-practices-badge/blob/main/CONTRIBUTING.md).)

Feedback and contributions are very welcome! Here’s how you can contribute, divided into the following sections:

* General Information
* Vulnerability Reporting
* Documentation Changes
* Code Changes
* Reuse (Third-Party Components)

## General Information

For specific proposals, please submit them as
[pull requests](https://github.com/TomTonic/Set3/pulls)
or
[issues](https://github.com/TomTonic/Set3/issues)
via our
[GitHub site](https://github.com/coreinfrastructure/best-practices-badge).
Welcome aboard!

### Pull Requests and Branches

Pull requests are preferred for their specificity.
For more on creating a pull request, see
[GitHub’s guide](https://help.github.com/articles/using-pull-requests/).

We request creating different branches for different logical
changes and submitting a pull request to the main branch when done.
See GitHub's documentation on
[creating branches](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
and
[using pull requests](https://help.github.com/articles/using-pull-requests/).

### Handling Proposals

We use GitHub to track proposed changes via its
[issue tracker](https://github.com/TomTonic/Set3/issues) and
[pull requests](https://github.com/TomTonic/Set3/pulls).
Issues are assigned to an individual who works on them and marks them complete.
If there are questions or objections, the conversation area of that
issue or pull request is used to resolve them.

### Reviews

Our policy is to have as many proposed modifications as possible reviewed by
someone other than the author. This ensures the modification is worthwhile and
free of known issues.

We categorize proposals into two types:

1. **Low-risk modifications.**  Proposed by authorized committers, pass all

   tests, and are unlikely to have problems. These include documentation
   updates and minor function updates. The project lead can designate any
   modification as low-risk.
2. Other modifications.  Require review by someone else or acceptance by the

   project lead. Typically, this involves creating a branch and a pull request
   for review before acceptance.

### Developer Certificate of Origin (DCO)

All contributions must agree to the Linux kernel developers'
[Developer Certificate of Origin (DCO) version 1.1](https://developercertificate.org).
This certifies that the contributor has the right to submit the patch for
inclusion in the project.

Simply submitting a contribution implies this agreement, however,
please include a "Signed-off-by" tag in every patch
(this tag is a conventional way to confirm that you agree to the DCO).
You can do this with `git commit --signoff` (the `-s` flag
is a synonym for `--signoff`).

Alternatively, add the following at the end of the commit message, separated
by a blank line from the body of the commit:

````txt
Signed-off-by: YOUR NAME <YOUR.EMAIL@EXAMPLE.COM>
```

`

You can sign-off by default in this project by creating a file
(say "git-template") that contains
some blank lines and the signed-off-by text above;
then configure git to use that as a commit template.  For example:

````sh
git config commit.template ~/best-practices-badge/git-template
```

`

It's not practical to fix old contributions in git, so if one is forgotten,
do not try to fix them.  We presume that if someone sometimes used a DCO,
a commit without a DCO is an accident and the DCO still applies.

### Proactive Approach

We proactively detect and eliminate
mistakes and vulnerabilities as soon as possible,
reducing their impact.
We use defensive design, coding styles,
various tools,
and an automatic test suite with significant coverage.
We also release the software as open source for community review.

## Vulnerability Reporting (security issues)

Please privately report vulnerabilities you find so we can fix them.

See [SECURITY.md](./SECURITY.md) for information on how to report vulnerabilities privately.

## Documentation Changes

Most documentation is in Markdown format (.md files).

Where reasonable, limit yourself to Markdown
that will be accepted by different processors
(e.g., CommonMark or original Markdown).

## Code Changes

Code should be DRY (Don’t Repeat Yourself),
clear, and obviously correct.
Some technical debt is inevitable, but avoid excessive debt.
Improving refactorings are welcome.

### Automated Tests

When adding or changing functionality, include new tests as
part of your contribution.
Ensure Go code has at least 98% statement coverage.
Additional tests are welcome.

We encourage test-driven development (TDD): create tests first, ensure they fail,
then add code to pass the tests.
Each git commit should include both
the test and the improvement to facilitate `git bisect`.

### Security, Privacy, and Performance

Pay attention to security and work with our
security hardening mechanisms.
Protect private information, especially passwords and email addresses.
Avoid tracking mechanisms where possible
and ensure third parties can’t use interactions for tracking.

We aim for decent performance for typical users.
Use benchmark testing as needed
(e.g., see <https://github.com/TomTonic/Set3/blob/main/run_benchmark.txt>).

### Continuous Integration

We use [Github Actions](https://github.com/TomTonic/Set3/actions)
for continuous integration. If problems are found, please fix them.

## Git Commit Messages

Follow these guidelines for writing git commit messages:

1. Separate subject from body with a blank line.
2. Limit the subject line to 50 characters (flexible up to 72 characters).
3. Capitalize the subject line.
4. Do not end the subject line with a period.
5. Use the imperative mood in the subject line (*command* form).
6. Wrap the body at 72 characters (`fmt -w 72`).
7. Use the body to explain what and why, not how

   (git tracks how it was changed in detail, don't repeat that).

## Reuse (Supply Chain)

### Requirements for Reused Components

We prefer reusing components over writing extensive new code.
However, please evaluate all new components before adding them,
including assessing their necessity.
This helps us minimize the risk of relying on poorly
maintained or vulnerable software.
Set3 provides a concise set of functionalities for a
fundamental [abstract data type](https://en.wikipedia.org/wiki/Abstract_data_type).
Therefore, Set3 should not require a Go project using it to pull in a significant
tree of dependencies.

#### License Requirements for Reused Components

All *required* reused software *must* be open source software (OSS).
It's okay to *optionally* use proprietary software and add
portability fixes.

### Updating Reused Components

Please update only one or a few components per commit, rather than
updating everything at once. This approach simplifies debugging.
If a problem arises later, we can
use `git bisect` to quickly identify the cause.
