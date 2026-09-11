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

package setcompare

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// The CSV files are semicolon separated, which is what the existing
// lab/results files use and what a European spreadsheet opens without a dialog.
// Fields never contain a semicolon: the only free-text column is the note, and
// writeCSV replaces separators in it.
const csvSep = ";"

// runtimeHeader names every column of runtime.csv. The order is: what was
// measured, what came out, how much to trust it, and under what settings.
var runtimeHeader = []string{
	"scenario", "keytype", "size", "unit", "items_per_op", "raw_block_hash",
	"set3_load_factor", "map_load_factor",
	"set3_ns", "map_ns",
	"delta_pct", "ci_low_pct", "ci_high_pct", "level", "noise_floor_pct", "resolved",
	"repeats", "validation_runs", "inner_loops", "block_length", "autocorrelation", "tie_rate",
	"drift_p_set3", "drift_p_map",
	"set3_alloc_bytes", "map_alloc_bytes", "set3_mallocs", "map_mallocs",
	"skipped", "seconds", "note",
}

// memoryHeader names every column of memory.csv.
var memoryHeader = []string{
	"shape", "keytype", "size", "elements",
	"set3_bytes", "map_bytes", "set3_bytes_per_element", "map_bytes_per_element", "ratio_set3_over_map",
	"set3_build_bytes_per_element", "map_build_bytes_per_element",
	"set3_build_mallocs_per_element", "map_build_mallocs_per_element",
	"note",
}

// WriteResults writes both CSV files and the run manifest into cfg.OutDir,
// creating the directory if it does not exist.
//
// The manifest is the part that makes a recorded result reusable six months
// later: it names the Go version, the architecture, the configuration and the
// scenario documentation, so a CSV found in the repository can be read without
// guessing what produced it. Returns the paths written, and an error from the
// first file that could not be written.
func WriteResults(cfg Config, runtimeRows []RuntimeResult, memoryRows []MemoryResult, elapsed time.Duration) ([]string, error) {
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return nil, fmt.Errorf("create %s: %w", cfg.OutDir, err)
	}
	var written []string

	if len(runtimeRows) > 0 {
		path := filepath.Join(cfg.OutDir, "runtime.csv")
		if err := writeCSV(path, runtimeHeader, runtimeRowsToRecords(runtimeRows)); err != nil {
			return written, err
		}
		written = append(written, path)
	}
	if len(memoryRows) > 0 {
		path := filepath.Join(cfg.OutDir, "memory.csv")
		if err := writeCSV(path, memoryHeader, memoryRowsToRecords(memoryRows)); err != nil {
			return written, err
		}
		written = append(written, path)
	}

	path := filepath.Join(cfg.OutDir, "run.txt")
	if err := os.WriteFile(path, []byte(manifest(cfg, runtimeRows, memoryRows, elapsed)), 0o600); err != nil {
		return written, fmt.Errorf("write %s: %w", path, err)
	}
	return append(written, path), nil
}

// writeCSV writes a header and its records, separator-joined.
func writeCSV(path string, header []string, records [][]string) error {
	var b strings.Builder
	b.WriteString(strings.Join(header, csvSep))
	b.WriteByte('\n')
	for _, rec := range records {
		if len(rec) != len(header) {
			return fmt.Errorf("%s: record has %d fields, header has %d", path, len(rec), len(header))
		}
		b.WriteString(strings.Join(rec, csvSep))
		b.WriteByte('\n')
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

func runtimeRowsToRecords(rows []RuntimeResult) [][]string {
	out := make([][]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, []string{
			r.Scenario, r.KeyType, strconv.Itoa(r.Size), r.Unit, f(r.ItemsPerOp, 0), b(r.RawBlockHash),
			f(r.Set3LoadFactor, 4), f(r.MapLoadFactor, 4),
			f(r.Set3NsPerItem, 4), f(r.MapNsPerItem, 4),
			f(r.DeltaPct, 3), f(r.CILowPct, 3), f(r.CIHighPct, 3), f(r.Level, 3), f(r.NoiseFloorPct, 4), b(r.Resolved),
			strconv.Itoa(r.Repeats), strconv.Itoa(r.ValidationRuns), strconv.FormatUint(r.InnerLoops, 10),
			strconv.Itoa(r.BlockLength), f(r.Autocorrelation, 4), f(r.TieRate, 4),
			f(r.DriftPSet3, 4), f(r.DriftPMap, 4),
			f(r.Set3AllocBytesPerItem, 3), f(r.MapAllocBytesPerItem, 3), f(r.Set3MallocsPerItem, 5), f(r.MapMallocsPerItem, 5),
			b(r.Skipped), f(r.Seconds, 1), clean(r.Note),
		})
	}
	return out
}

func memoryRowsToRecords(rows []MemoryResult) [][]string {
	out := make([][]string, 0, len(rows))
	for _, r := range rows {
		out = append(out, []string{
			r.Shape, r.KeyType, strconv.Itoa(r.Size), strconv.Itoa(r.Elements),
			f(r.Set3Bytes, 0), f(r.MapBytes, 0), f(r.Set3BytesPerElement, 4), f(r.MapBytesPerElement, 4), f(r.RatioSet3OverMap, 5),
			f(r.Set3BuildBytes, 3), f(r.MapBuildBytes, 3),
			f(r.Set3BuildMallocs, 5), f(r.MapBuildMallocs, 5),
			clean(r.Note),
		})
	}
	return out
}

func f(v float64, digits int) string { return strconv.FormatFloat(v, 'f', digits, 64) }

func b(v bool) string { return strconv.FormatBool(v) }

// clean makes a free-text note safe for a separator-joined line.
func clean(s string) string {
	s = strings.ReplaceAll(s, csvSep, ",")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}

// manifest renders everything needed to interpret the CSVs later.
func manifest(cfg Config, runtimeRows []RuntimeResult, memoryRows []MemoryResult, elapsed time.Duration) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Set3 vs. map[T]struct{} — rtcompare suite\n")
	fmt.Fprintf(&b, "generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "elapsed:   %s\n", elapsed.Round(time.Second))
	fmt.Fprintf(&b, "go:        %s %s/%s, %d CPUs\n", runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	fmt.Fprintf(&b, "config:    %s\n", cfg)
	fmt.Fprintf(&b, "rows:      %d runtime, %d memory\n\n", len(runtimeRows), len(memoryRows))

	b.WriteString("Reading a runtime row\n")
	b.WriteString("  delta_pct is positive when Set3 is faster. ci_low_pct/ci_high_pct bound it at\n")
	b.WriteString("  the stated level. A row is only worth quoting when resolved is true: that means\n")
	b.WriteString("  the interval excludes zero AND the difference is larger than noise_floor_pct,\n")
	b.WriteString("  which is what this machine reported as a difference between two runs of\n")
	b.WriteString("  identical code. Read the note column even on a resolved row.\n\n")

	b.WriteString("Scenarios\n")
	for _, s := range Workloads {
		fmt.Fprintf(&b, "  %-15s %s\n", s.name, s.doc)
	}
	b.WriteString("\nMemory shapes\n")
	b.WriteString("  presized      created at the right capacity, then filled\n")
	b.WriteString("  grown         created empty, then filled — includes whatever capacity growth landed on\n")
	b.WriteString("  half-removed  presized and filled, then every second element removed\n")
	b.WriteString("  window-steady a sliding window that has churned through four times its own size\n")
	return b.String()
}

// Summarize renders the short human-readable verdict that goes to the test log
// when a run finishes: how many cells resolved in each direction, and the
// extremes.
//
// It exists because a 90-row CSV is not something anyone reads at the end of a
// 30-minute run, and the one thing worth seeing immediately is whether the
// picture matches the last run or whether something moved.
func Summarize(rows []RuntimeResult) string {
	var wins, losses, unresolved, skipped int
	best, worst := RuntimeResult{}, RuntimeResult{}
	for _, r := range rows {
		switch {
		case r.Skipped:
			skipped++
			continue
		case !r.Resolved:
			unresolved++
			continue
		case r.DeltaPct > 0:
			wins++
		default:
			losses++
		}
		if r.DeltaPct > best.DeltaPct {
			best = r
		}
		if r.DeltaPct < worst.DeltaPct {
			worst = r
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%d cells: Set3 faster in %d, slower in %d, unresolved in %d, skipped %d\n",
		len(rows), wins, losses, unresolved, skipped)
	if best.Scenario != "" {
		fmt.Fprintf(&b, "  best  for Set3: %s/%s n=%d  %+.1f%% [%+.1f%%, %+.1f%%]\n",
			best.Scenario, best.KeyType, best.Size, best.DeltaPct, best.CILowPct, best.CIHighPct)
	}
	if worst.Scenario != "" {
		fmt.Fprintf(&b, "  worst for Set3: %s/%s n=%d  %+.1f%% [%+.1f%%, %+.1f%%]\n",
			worst.Scenario, worst.KeyType, worst.Size, worst.DeltaPct, worst.CILowPct, worst.CIHighPct)
	}
	return b.String()
}

// SummarizeMemory is Summarize for the memory pass: the mean footprint ratio
// and the range it spans.
func SummarizeMemory(rows []MemoryResult) string {
	if len(rows) == 0 {
		return "no memory rows\n"
	}
	var sum, lo, hi float64
	lo, hi = 1e9, 0
	var n int
	for _, r := range rows {
		if r.RatioSet3OverMap <= 0 {
			continue
		}
		sum += r.RatioSet3OverMap
		lo = min(lo, r.RatioSet3OverMap)
		hi = max(hi, r.RatioSet3OverMap)
		n++
	}
	if n == 0 {
		return "no usable memory rows\n"
	}
	return fmt.Sprintf("%d memory cells: Set3 holds %.1f%% of the native map's bytes on average (range %.1f%% to %.1f%%)\n",
		n, sum/float64(n)*100, lo*100, hi*100)
}
