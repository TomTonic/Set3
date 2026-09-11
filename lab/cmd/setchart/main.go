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

// Command setchart turns the CSV files written by lab/setcompare into the SVG
// charts the README embeds.
//
// It is a separate program rather than part of the suite because the two have
// very different costs: measuring takes half an hour on a quiet machine, and
// drawing takes a tenth of a second. Keeping them apart means a chart can be
// redrawn — relabelled, recoloured, split differently — without spending the
// half hour again, and it means the CSVs stay the archive of what was actually
// measured.
//
// Usage:
//
//	go run -tags set3lab ./lab/cmd/setchart -in lab/results/setcompare -out lab/results/setcompare
//
// It writes one speedup chart, one cost chart and one memory chart per key
// type, plus README-snippet.md holding the markdown that embeds them.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	in := flag.String("in", "lab/results/setcompare", "directory holding runtime.csv and memory.csv")
	out := flag.String("out", "", "directory to write the SVG files to (default: same as -in)")
	subtitle := flag.String("subtitle", "", "subtitle for every chart (default: taken from run.txt)")
	flag.Parse()

	if *out == "" {
		*out = *in
	}
	if err := run(*in, *out, *subtitle); err != nil {
		fmt.Fprintln(os.Stderr, "setchart:", err)
		os.Exit(1)
	}
}

// run loads whichever of the two CSVs exist and draws everything they support.
//
// A missing file is not an error: a run launched with SET3_CMP_SKIP_MEMORY
// produces only runtime.csv, and redrawing its charts should not require
// inventing the other half.
func run(inDir, outDir, subtitle string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("create %s: %w", outDir, err)
	}
	if subtitle == "" {
		subtitle = subtitleFromManifest(filepath.Join(inDir, "run.txt"))
	}

	var written []string
	runtimeRows, err := loadIfPresent(filepath.Join(inDir, "runtime.csv"), loadRuntime)
	if err != nil {
		return err
	}
	memoryRows, err := loadIfPresent(filepath.Join(inDir, "memory.csv"), loadMemory)
	if err != nil {
		return err
	}
	if len(runtimeRows) == 0 && len(memoryRows) == 0 {
		return fmt.Errorf("neither runtime.csv nor memory.csv found in %s", inDir)
	}

	// Deliberately not sorted: the order here is the order the snippet embeds
	// them in, and it is the order the charts should be read in. The verdict
	// first, then what it cost, then what it cost to hold — per key type, so
	// that a reader interested in one key type sees its three charts together.
	for _, keyType := range distinct(runtimeRows, func(r runtimeRow) string { return r.keyType }) {
		written = append(written,
			save(outDir, "speedup-"+keyType+".svg", speedupChart(runtimeRows, keyType, subtitle)),
			save(outDir, "cost-"+keyType+".svg", costChart(runtimeRows, keyType, subtitle)),
			save(outDir, "memory-"+keyType+".svg", memoryChart(memoryRows, keyType, subtitle)))
	}
	written = compact(written)

	snippet := filepath.Join(outDir, "README-snippet.md")
	if err := os.WriteFile(snippet, []byte(readmeSnippet(written, runtimeRows, memoryRows, subtitle)), 0o600); err != nil {
		return fmt.Errorf("write %s: %w", snippet, err)
	}
	for _, path := range append(written, snippet) {
		fmt.Println("wrote", path)
	}
	return nil
}

// loadIfPresent loads a CSV, treating a missing file as an empty result.
func loadIfPresent[T any](path string, load func(string) ([]T, error)) ([]T, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, nil
	}
	return load(path)
}

// save writes a chart and returns its path, or the empty string when there was
// nothing to draw.
func save(dir, name string, c *canvas) string {
	if c == nil {
		return ""
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(c.String()), 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "setchart: write %s: %v\n", path, err)
		return ""
	}
	return path
}

func compact(paths []string) []string {
	out := paths[:0]
	for _, p := range paths {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// subtitleFromManifest builds a one-line provenance string from run.txt, so
// that every chart carries the Go version and the date it was measured on.
// A chart without that is a chart nobody can reproduce.
func subtitleFromManifest(path string) string {
	raw, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return ""
	}
	var goLine, generated string
	for _, line := range strings.Split(string(raw), "\n") {
		switch {
		case strings.HasPrefix(line, "go:"):
			goLine = strings.TrimSpace(strings.TrimPrefix(line, "go:"))
		case strings.HasPrefix(line, "generated:"):
			generated = strings.TrimSpace(strings.TrimPrefix(line, "generated:"))
			if i := strings.Index(generated, "T"); i > 0 {
				generated = generated[:i]
			}
		}
	}
	if goLine == "" {
		return ""
	}
	return goLine + " · measured " + generated + " · rtcompare, 95% intervals"
}

// readmeSnippet writes the markdown that embeds the charts, with the headline
// numbers spelled out next to them.
//
// The numbers are repeated in text rather than left to the images on purpose:
// a chart is not readable by a screen reader, is not greppable, and does not
// survive being pasted into an issue.
func readmeSnippet(charts []string, runtimeRows []runtimeRow, memoryRows []memoryRow, subtitle string) string {
	var b strings.Builder
	b.WriteString("<!-- generated by lab/cmd/setchart; do not edit by hand -->\n\n")
	fmt.Fprintf(&b, "### Set3 against `map[T]struct{}`\n\n%s\n\n", subtitle)

	if len(runtimeRows) > 0 {
		b.WriteString(headlineTable(runtimeRows))
	}
	if len(memoryRows) > 0 {
		b.WriteString(memoryTable(memoryRows))
	}
	for _, path := range charts {
		name := filepath.Base(path)
		fmt.Fprintf(&b, "![%s](%s)\n\n", strings.TrimSuffix(name, ".svg"), path)
	}
	return b.String()
}

// headlineTable summarises the resolved runtime results per workload: the
// range of differences seen across sizes, and how many cells were resolved.
func headlineTable(rows []runtimeRow) string {
	var b strings.Builder
	b.WriteString("| Workload | key type | sizes measured | Set3 vs. native map | resolved |\n")
	b.WriteString("| --- | --- | --- | --- | --- |\n")
	for _, scenario := range distinct(rows, func(r runtimeRow) string { return r.scenario }) {
		for _, keyType := range distinct(rows, func(r runtimeRow) string { return r.keyType }) {
			group := filter(rows, func(r runtimeRow) bool {
				return r.scenario == scenario && r.keyType == keyType && !r.skipped
			})
			if len(group) == 0 {
				continue
			}
			lo, hi, resolved := 1e9, -1e9, 0
			for _, r := range group {
				lo, hi = minf(lo, r.delta), maxf(hi, r.delta)
				if r.resolved {
					resolved++
				}
			}
			fmt.Fprintf(&b, "| `%s` | %s | %s – %s | %s to %s | %d/%d |\n",
				scenario, keyType, shortCount(group[0].size), shortCount(group[len(group)-1].size),
				signedPercent(lo, 0), signedPercent(hi, 0), resolved, len(group))
		}
	}
	b.WriteString("\nPositive means Set3 is faster. A cell counts as resolved only when its interval excludes zero *and* the difference exceeds the machine's measured noise floor.\n\n")
	return b.String()
}

// memoryTable summarises the footprint per fill history at the largest size
// measured, which is the size where the constant overheads have washed out.
func memoryTable(rows []memoryRow) string {
	var b strings.Builder
	b.WriteString("| Fill history | key type | elements | Set3 | `map[T]struct{}` | ratio |\n")
	b.WriteString("| --- | --- | --- | --- | --- | --- |\n")
	for _, shape := range distinct(rows, func(r memoryRow) string { return r.shape }) {
		for _, keyType := range distinct(rows, func(r memoryRow) string { return r.keyType }) {
			group := filter(rows, func(r memoryRow) bool { return r.shape == shape && r.keyType == keyType })
			if len(group) == 0 {
				continue
			}
			last := group[len(group)-1]
			fmt.Fprintf(&b, "| `%s` | %s | %s | %.1f B/elem | %.1f B/elem | **%.2f×** |\n",
				shape, keyType, shortCount(last.elements), last.set3PerElem, last.mapPerElem, last.ratio)
		}
	}
	b.WriteString("\nRetained heap after a full collection, with the container reachable.\n\n")
	return b.String()
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
