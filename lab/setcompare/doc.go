//go:build set3lab

// Copyright 2024 TomTonic
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package setcompare compares Set3[T] against Go's native map[T]struct{} on
// runtime and on memory, and writes the result as CSV for the charts in the
// top-level README.
//
// # What makes this different from setbench
//
// [setbench] measures each implementation on its own and prints the numbers.
// This package asks the harder question — is the difference real on this
// machine, right now — and answers it with the full rtcompare protocol
// (see https://github.com/TomTonic/rtcompare/blob/main/HOWTO.md):
//
//  1. Batches are sized so the system clock contributes at most a fixed share
//     of error, and that size is then held fixed for everything that follows.
//  2. Each candidate is run against *itself*, repeatedly, to find out what this
//     machine reports as a difference when there provably is none. That is the
//     noise floor, and every result is read against it.
//  3. The two candidates are measured interleaved (ABBA), never one after the
//     other.
//  4. Each series is tested for a trend across the run, which the bootstrap
//     cannot see because it discards the order the samples arrived in.
//  5. Correlated samples are resampled in blocks instead of one at a time.
//  6. The reported difference carries a confidence interval, and a result that
//     does not clear both zero and the noise floor is reported as unresolved.
//
// A number from here therefore comes with the evidence for it. "Set3 is 12%
// faster" is written down as "12%, 95% CI [9%, 15%], noise floor 0.4%,
// resolved", and when the machine misbehaved the row says so instead of
// quietly reporting the noise.
//
// # What is measured
//
// Ten workloads, most of them shaped after something a program actually does
// with a set, across four key types and five set sizes up to eight million
// elements. See [Workloads] for the list and [scenarioDoc] for what each one
// stands for. Memory is a separate pass: the retained heap of a populated
// container, measured after a full collection, across four fill histories.
//
// # Running it
//
//	go test -tags set3lab -run TestCompareSuite -timeout 180m ./lab/setcompare
//
// The default run takes roughly 20-40 minutes and writes
// lab/results/setcompare/*.csv. Everything is configurable through SET3_CMP_*
// environment variables, see [Config]. Feed the CSVs to lab/cmd/setchart to
// get the SVGs the README embeds.
//
// This package is behind the set3lab build tag; see lab/README.md.
package setcompare
