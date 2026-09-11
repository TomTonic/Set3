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
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// syntheticRuntimeCSV is a two-workload, two-size run with one resolved result
// and one that is not, so that both branches of the drawing code are exercised.
const syntheticRuntimeCSV = `scenario;keytype;size;unit;items_per_op;raw_block_hash;set3_ns;map_ns;delta_pct;ci_low_pct;ci_high_pct;level;noise_floor_pct;resolved;repeats;validation_runs;inner_loops;block_length;autocorrelation;tie_rate;drift_p_set3;drift_p_map;set3_alloc_bytes;map_alloc_bytes;set3_mallocs;map_mallocs;skipped;seconds;note
lookup-hit30;uint64;1000;ns/lookup;1;true;3.7700;5.7900;34.900;33.600;35.600;0.950;0.6200;true;101;40;8192;1;0.0500;0.0200;0.4000;0.3000;0.000;0.000;0.00000;0.00000;false;8.1;
lookup-hit30;uint64;16384;ns/lookup;1;true;7.9300;12.0800;34.300;34.000;34.600;0.950;0.6800;true;101;40;4096;1;0.0700;0.0100;0.2000;0.1000;0.000;0.000;0.00000;0.00000;false;8.4;
iterate;uint64;1000;ns/element;1000;true;1.5000;1.5200;1.300;-0.400;2.900;0.950;0.9400;false;101;40;64;1;0.0300;0.1100;0.5000;0.6000;0.100;0.000;0.00100;0.00000;false;7.9;
iterate;uint64;16384;ns/element;16384;true;1.8100;6.1500;70.500;69.400;71.200;0.950;1.9900;true;101;24;4;1;0.0900;0.0300;0.2000;0.2000;0.010;0.000;0.00010;0.00000;false;9.2;
`

// syntheticMemoryCSV is one fill history at two sizes.
const syntheticMemoryCSV = `shape;keytype;size;elements;set3_bytes;map_bytes;set3_bytes_per_element;map_bytes_per_element;ratio_set3_over_map;set3_build_bytes_per_element;map_build_bytes_per_element;set3_build_mallocs_per_element;map_build_mallocs_per_element;note
presized;uint64;1000;1000;11600;36990;11.6000;36.9900;0.31360;11.600;36.990;0.00800;0.00800;
presized;uint64;16384;16384;184512;591376;11.2620;36.0900;0.31206;11.262;36.090;0.00400;0.00400;
`

// TestChartsRenderFromARecordedRun verifies that a recorded measurement run
// turns into the SVG files and the markdown snippet the README embeds.
//
// This is the whole job of the setchart command: someone finishes a half-hour
// suite run, points this at the directory, and gets pictures. The test drives
// exactly that path — two CSV files in, charts and a snippet out — because
// every failure mode of this tool (a renamed column, a chart with no data, a
// division by zero on a single-point series) shows up as a missing or empty
// file rather than as a crash.
func TestChartsRenderFromARecordedRun(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "runtime.csv"), syntheticRuntimeCSV)
	write(t, filepath.Join(dir, "memory.csv"), syntheticMemoryCSV)
	write(t, filepath.Join(dir, "run.txt"), "generated: 2026-09-10T12:00:00Z\ngo:        go1.26.8 linux/amd64, 24 CPUs\n")

	if err := run(dir, dir, ""); err != nil {
		t.Fatalf("drawing the charts failed: %v", err)
	}

	for _, name := range []string{"speedup-uint64.svg", "cost-uint64.svg", "memory-uint64.svg", "README-snippet.md"} {
		body := read(t, filepath.Join(dir, name))
		if len(body) < 400 {
			t.Errorf("%s is only %d bytes; it cannot hold a chart", name, len(body))
		}
		if strings.HasSuffix(name, ".svg") {
			if !strings.HasPrefix(body, "<svg") || !strings.HasSuffix(strings.TrimSpace(body), "</svg>") {
				t.Errorf("%s is not a well-formed SVG document", name)
			}
			if !strings.Contains(body, colBackground) {
				t.Errorf("%s has no painted background, so it will be unreadable in GitHub's dark theme", name)
			}
		}
	}

	speedup := read(t, filepath.Join(dir, "speedup-uint64.svg"))
	if !strings.Contains(speedup, colUnresolved) {
		t.Error("the unresolved row is not drawn in the unresolved colour, so a null result reads as a win")
	}
	if !strings.Contains(speedup, colNoise) {
		t.Error("the noise floor band is missing, and it is the thing that makes the chart honest")
	}

	snippet := read(t, filepath.Join(dir, "README-snippet.md"))
	for _, want := range []string{"lookup-hit30", "go1.26.8", "0.31", "speedup-uint64.svg"} {
		if !strings.Contains(snippet, want) {
			t.Errorf("the README snippet does not mention %q", want)
		}
	}
}

// TestMissingMemoryFileIsNotAnError verifies that a run which measured only
// runtime still produces its charts.
//
// It matters because SET3_CMP_SKIP_MEMORY is the normal way to refresh half
// the picture, and a tool that refuses to draw anything without both halves
// would make that setting useless.
func TestMissingMemoryFileIsNotAnError(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "runtime.csv"), syntheticRuntimeCSV)

	if err := run(dir, dir, "just the runtime"); err != nil {
		t.Fatalf("drawing the charts failed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "speedup-uint64.svg")); err != nil {
		t.Errorf("the speedup chart was not written: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "memory-uint64.svg")); !os.IsNotExist(err) {
		t.Error("a memory chart was written although there were no memory measurements")
	}
}

// TestEmptyInputIsReported verifies that pointing the tool at the wrong
// directory says so, instead of writing an empty snippet and exiting happily.
func TestEmptyInputIsReported(t *testing.T) {
	if err := run(t.TempDir(), t.TempDir(), ""); err == nil {
		t.Fatal("a directory with no CSV files should be an error")
	}
}

// TestScalesMapTheirEndpoints verifies that both axis scales put the data
// range exactly on the pixel range they were given.
//
// Everything drawn on these charts goes through scale.at, so an off-by-one
// here would not produce a wrong-looking chart — it would produce a chart that
// looks fine and is wrong, which is the failure this whole suite exists to
// avoid in the measurements.
func TestScalesMapTheirEndpoints(t *testing.T) {
	lin := newLinearScale(-10, 30, 100, 500)
	if got := lin.at(-10); math.Abs(got-100) > 1e-9 {
		t.Errorf("linear scale puts its minimum at %v, want 100", got)
	}
	if got := lin.at(30); math.Abs(got-500) > 1e-9 {
		t.Errorf("linear scale puts its maximum at %v, want 500", got)
	}
	if got := lin.at(10); math.Abs(got-300) > 1e-9 {
		t.Errorf("linear scale puts its midpoint at %v, want 300", got)
	}

	log := newLogScale(1000, 1_000_000, 0, 300)
	if got := log.at(1000); math.Abs(got) > 1e-9 {
		t.Errorf("log scale puts its minimum at %v, want 0", got)
	}
	if got := log.at(1_000_000); math.Abs(got-300) > 1e-9 {
		t.Errorf("log scale puts its maximum at %v, want 300", got)
	}
	if got := log.at(31622.7766); math.Abs(got-150) > 0.01 {
		t.Errorf("log scale puts the geometric midpoint at %v, want 150", got)
	}
	if n := len(log.niceTicks(4)); n != 4 {
		t.Errorf("a range of three decades produced %d ticks, want 4", n)
	}
}

// TestSizeLabelsStayShort verifies the axis labels, which are the one place a
// wrong format makes a chart unreadable rather than merely ugly.
func TestSizeLabelsStayShort(t *testing.T) {
	cases := map[int]string{1: "1", 999: "999", 1000: "1k", 16384: "16k", 262144: "262k", 2097152: "2.1M"}
	for in, want := range cases {
		if got := shortCount(in); got != want {
			t.Errorf("shortCount(%d) = %q, want %q", in, got, want)
		}
	}
	if got := signedPercent(12.34, 1); got != "+12.3%" {
		t.Errorf("signedPercent(12.34) = %q, want +12.3%%", got)
	}
	if got := signedPercent(-4, 0); got != "-4%" {
		t.Errorf("signedPercent(-4) = %q, want -4%%", got)
	}
}

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func read(t *testing.T, path string) string {
	t.Helper()
	body, err := os.ReadFile(path) //nolint:gosec
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(body)
}

// TestNothingIsDrawnOutsideTheCanvas verifies that every element of every
// chart lands inside the document it belongs to.
//
// A chart is the one artefact in this repository that nobody can check by
// reading it: an SVG with a bar running off the right edge or a label at a
// negative coordinate looks perfectly valid as text and is broken as a
// picture. This walks the generated markup and checks the coordinates, which
// is the closest thing to looking at it that a test can do.
func TestNothingIsDrawnOutsideTheCanvas(t *testing.T) {
	charts := map[string]*canvas{
		"speedup": speedupChart(mustLoadRuntime(t), "uint64", "test"),
		"cost":    costChart(mustLoadRuntime(t), "uint64", "test"),
		"memory":  memoryChart(mustLoadMemory(t), "uint64", "test"),
	}
	coord := regexp.MustCompile(`\b(x|x1|x2|cx|y|y1|y2|cy)="(-?[0-9.]+)"`)

	for name, c := range charts {
		if c == nil {
			t.Fatalf("%s chart came out empty", name)
		}
		body := c.String()
		for _, m := range coord.FindAllStringSubmatch(body, -1) {
			v, err := strconv.ParseFloat(m[2], 64)
			if err != nil {
				t.Fatalf("%s chart: %q is not a coordinate", name, m[2])
			}
			limit := c.width
			if strings.HasPrefix(m[1], "y") || m[1] == "cy" {
				limit = c.height
			}
			if v < -1 || v > limit+1 {
				t.Errorf("%s chart: %s=%v falls outside the %vx%v canvas", name, m[1], v, c.width, c.height)
			}
		}
	}
}

func mustLoadRuntime(t *testing.T) []runtimeRow {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "runtime.csv")
	write(t, path, syntheticRuntimeCSV)
	rows, err := loadRuntime(path)
	if err != nil {
		t.Fatalf("loading the synthetic runtime rows failed: %v", err)
	}
	return rows
}

func mustLoadMemory(t *testing.T) []memoryRow {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "memory.csv")
	write(t, path, syntheticMemoryCSV)
	rows, err := loadMemory(path)
	if err != nil {
		t.Fatalf("loading the synthetic memory rows failed: %v", err)
	}
	return rows
}

// TestZeroTicksHaveNoSign verifies that a tick value of negative zero renders
// as "0%" rather than as "+-0%".
//
// It is a one-character bug with a disproportionate effect: the zero tick is
// the reference line of the headline chart, and a reader who sees "+-0%" there
// stops trusting every other number on it. Tick positions are computed by
// floating-point arithmetic, which produces negative zero routinely, so this
// is a case that happens rather than one that could.
func TestZeroTicksHaveNoSign(t *testing.T) {
	for _, v := range []float64{0, math.Copysign(0, -1)} {
		if got := signedPercent(v, 0); got != "0%" {
			t.Errorf("signedPercent(%v) = %q, want \"0%%\"", v, got)
		}
	}
	if got := signedPercent(-0.04, 1); got != "0.0%" {
		t.Errorf("a value that rounds to zero should lose its sign, got %q", got)
	}
}
