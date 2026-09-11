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

package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// runtimeRow is one row of runtime.csv, in the fields the charts use. The CSV
// carries more than this — the diagnostics that justify a row rather than
// describe it — and this struct deliberately takes only what is drawn.
type runtimeRow struct {
	scenario   string
	keyType    string
	size       int
	unit       string
	set3Ns     float64
	mapNs      float64
	delta      float64 // percent, positive means Set3 is faster
	ciLow      float64
	ciHigh     float64
	noiseFloor float64
	resolved   bool
	skipped    bool
	set3Alloc  float64
	mapAlloc   float64
	note       string
}

// memoryRow is one row of memory.csv.
type memoryRow struct {
	shape       string
	keyType     string
	size        int
	elements    int
	set3PerElem float64
	mapPerElem  float64
	ratio       float64
	set3Build   float64
	mapBuild    float64
}

// table is a parsed separator-delimited file addressed by column name, so that
// adding a column to the suite's output does not renumber anything here.
type table struct {
	index map[string]int
	rows  [][]string
}

// readTable loads a semicolon-separated file with a header line.
//
// It returns an error naming the file for anything malformed, because the most
// likely reason this tool is run against a bad file is that someone pointed it
// at the wrong directory, and "field count mismatch" alone would not say so.
func readTable(path string) (*table, error) {
	raw, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("%s: no data rows", path)
	}

	header := strings.Split(lines[0], ";")
	t := &table{index: make(map[string]int, len(header))}
	for i, name := range header {
		t.index[strings.TrimSpace(name)] = i
	}
	for n, line := range lines[1:] {
		fields := strings.Split(line, ";")
		if len(fields) != len(header) {
			return nil, fmt.Errorf("%s line %d: %d fields, header has %d", path, n+2, len(fields), len(header))
		}
		t.rows = append(t.rows, fields)
	}
	return t, nil
}

func (t *table) str(row []string, col string) string {
	i, ok := t.index[col]
	if !ok || i >= len(row) {
		return ""
	}
	return row[i]
}

func (t *table) num(row []string, col string) float64 {
	v, err := strconv.ParseFloat(t.str(row, col), 64)
	if err != nil {
		return 0
	}
	return v
}

func (t *table) int(row []string, col string) int {
	v, err := strconv.Atoi(t.str(row, col))
	if err != nil {
		return 0
	}
	return v
}

func (t *table) flag(row []string, col string) bool {
	return t.str(row, col) == "true"
}

// loadRuntime reads runtime.csv. Skipped cells are kept: a chart that silently
// omits the sizes that were too expensive to measure tells a more flattering
// story than the run does.
func loadRuntime(path string) ([]runtimeRow, error) {
	t, err := readTable(path)
	if err != nil {
		return nil, err
	}
	out := make([]runtimeRow, 0, len(t.rows))
	for _, r := range t.rows {
		out = append(out, runtimeRow{
			scenario:   t.str(r, "scenario"),
			keyType:    t.str(r, "keytype"),
			size:       t.int(r, "size"),
			unit:       t.str(r, "unit"),
			set3Ns:     t.num(r, "set3_ns"),
			mapNs:      t.num(r, "map_ns"),
			delta:      t.num(r, "delta_pct"),
			ciLow:      t.num(r, "ci_low_pct"),
			ciHigh:     t.num(r, "ci_high_pct"),
			noiseFloor: t.num(r, "noise_floor_pct"),
			resolved:   t.flag(r, "resolved"),
			skipped:    t.flag(r, "skipped"),
			set3Alloc:  t.num(r, "set3_alloc_bytes"),
			mapAlloc:   t.num(r, "map_alloc_bytes"),
			note:       t.str(r, "note"),
		})
	}
	return out, nil
}

// loadMemory reads memory.csv.
func loadMemory(path string) ([]memoryRow, error) {
	t, err := readTable(path)
	if err != nil {
		return nil, err
	}
	out := make([]memoryRow, 0, len(t.rows))
	for _, r := range t.rows {
		out = append(out, memoryRow{
			shape:       t.str(r, "shape"),
			keyType:     t.str(r, "keytype"),
			size:        t.int(r, "size"),
			elements:    t.int(r, "elements"),
			set3PerElem: t.num(r, "set3_bytes_per_element"),
			mapPerElem:  t.num(r, "map_bytes_per_element"),
			ratio:       t.num(r, "ratio_set3_over_map"),
			set3Build:   t.num(r, "set3_build_bytes_per_element"),
			mapBuild:    t.num(r, "map_build_bytes_per_element"),
		})
	}
	return out, nil
}

// distinct returns the values of key across rows, in first-seen order. Order
// matters: the suite emits scenarios in catalogue order and sizes ascending,
// and the charts should read the same way.
func distinct[T any, K comparable](rows []T, key func(T) K) []K {
	seen := make(map[K]bool)
	var out []K
	for _, r := range rows {
		k := key(r)
		if !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	return out
}

// filter returns the rows for which keep is true.
func filter[T any](rows []T, keep func(T) bool) []T {
	var out []T
	for _, r := range rows {
		if keep(r) {
			out = append(out, r)
		}
	}
	return out
}
