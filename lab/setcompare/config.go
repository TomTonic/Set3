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
	"slices"
	"strconv"
	"strings"
	"time"
)

// Size tiers used across the suite. They are spread over three orders of
// magnitude on purpose: the interesting part of a hash table comparison is
// where the table stops fitting into a cache level, and that boundary moves
// with the machine.
const (
	// SizeL1 fits in L1/L2 on every machine — pure instruction cost, no misses.
	SizeL1 = 1_000
	// SizeL2 is around the L2 boundary on a typical desktop core.
	SizeL2 = 16_384
	// SizeL3 sits inside L3 on most server parts and outside it on laptops.
	SizeL3 = 262_144
	// SizeRAM is well past any cache: every probe is a memory access.
	SizeRAM = 2_097_152
	// SizeHuge is the "does this still hold up" size. Off by default because
	// one cell at this size keeps both containers plus their key material
	// resident, which is several gigabytes for the string key type.
	SizeHuge = 8_388_608
)

// Config is the knob board for one suite run. Every field is settable through
// an environment variable so a run can be retargeted without editing code,
// which matters because these runs are long and are usually launched from a
// shell on a quiet machine rather than from an editor.
//
// Call [LoadConfig] to build one; the zero value is not usable.
type Config struct {
	// OutDir is where the CSV files are written. SET3_CMP_OUT.
	OutDir string

	// Budget is the wall-clock time one comparison cell may take. It decides
	// how many validation runs and repeats a cell gets: cheap cells get the
	// rtcompare defaults, expensive ones get fewer. SET3_CMP_BUDGET.
	Budget time.Duration

	// HardCap is the point at which a cell is skipped rather than measured.
	// A cell whose cheapest admissible setting still exceeds this is recorded
	// as skipped, with its estimated cost, instead of stalling the run.
	// SET3_CMP_HARDCAP.
	HardCap time.Duration

	// Sizes are the set sizes to measure. SET3_CMP_SIZES, comma separated.
	Sizes []int

	// LoadCurveSizes are the sizes the load-curve pass sweeps. It is a short
	// list on purpose: the pass measures six occupancies per size, so each
	// entry costs six comparisons. One cache-resident size and one that is not
	// is enough to show whether the curve's shape depends on the working set.
	// SET3_CMP_CURVE_SIZES.
	LoadCurveSizes []int

	// KeyTypes selects which key types run. SET3_CMP_KEYS, comma separated,
	// from uint64, string, struct3x64, structmixed.
	KeyTypes []string

	// Scenarios filters the workload list by name. Empty means all.
	// SET3_CMP_SCENARIOS, comma separated.
	Scenarios []string

	// Resamples is the bootstrap resample count. SET3_CMP_RESAMPLES.
	Resamples uint64

	// Level is the coverage level of the reported interval. SET3_CMP_LEVEL.
	Level float64

	// MemRepeats is how many times the memory pass rebuilds each container
	// before taking the median of the retained heap. SET3_CMP_MEM_REPEATS.
	MemRepeats int

	// SkipMemory and SkipRuntime turn off one of the two passes, for when a
	// run only needs to refresh half the charts. SET3_CMP_SKIP_MEMORY /
	// SET3_CMP_SKIP_RUNTIME.
	SkipMemory  bool
	SkipRuntime bool

	// SkipLoadCurve turns off the load-curve pass. SET3_CMP_SKIP_CURVE.
	SkipLoadCurve bool

	// Cells restricts the runtime pass to named cells, each written
	// "scenario/keytype/size". Empty means every cell the other filters allow.
	// SET3_CMP_CELLS, comma separated.
	//
	// It exists for one job: re-measuring the handful of cells a run reported
	// as untrustworthy, without paying for the whole schedule again. Pair it
	// with Merge and a longer Budget.
	Cells []string

	// Merge folds this run's rows into the CSV files already in OutDir instead
	// of replacing them: a row with the same scenario, key type and size is
	// overwritten, everything else is kept where it was. SET3_CMP_MERGE.
	//
	// Without it a targeted re-measurement would leave an output directory
	// holding four rows.
	Merge bool

	// Tag is an optional label written into the CSV header, e.g. a Go version
	// or "pgo". SET3_CMP_TAG.
	Tag string
}

// LoadConfig builds a Config from the environment, falling back to the
// defaults documented on each field.
//
// short selects the reduced schedule used by `go test -short`: two small
// sizes, one key type, a two-second budget. It exists so that CI can prove the
// suite still compiles and runs end to end in well under a minute without
// pretending the numbers mean anything.
//
// Returns an error only for values that are present but unparseable; a missing
// variable is never an error. Call it once at the top of a test and pass the
// result down.
func LoadConfig(short bool) (Config, error) {
	cfg := Config{
		OutDir:         envString("SET3_CMP_OUT", "../results/setcompare"),
		Budget:         envDuration("SET3_CMP_BUDGET", 30*time.Second),
		HardCap:        envDuration("SET3_CMP_HARDCAP", 300*time.Second),
		Sizes:          []int{SizeL1, SizeL2, SizeL3, SizeRAM},
		LoadCurveSizes: []int{SizeL2, SizeRAM},
		KeyTypes:       []string{keyTypeUint64, keyTypeString, keyTypeStruct},
		Resamples:      envUint("SET3_CMP_RESAMPLES", 5_000),
		Level:          envFloat("SET3_CMP_LEVEL", 0.95),
		MemRepeats:     int(envUint("SET3_CMP_MEM_REPEATS", 7)), //nolint:gosec
		Tag:            envString("SET3_CMP_TAG", ""),
	}
	if envBool("SET3_CMP_HUGE", false) {
		cfg.Sizes = append(cfg.Sizes, SizeHuge)
	}
	if short {
		cfg.Sizes = []int{SizeL1, SizeL2}
		cfg.LoadCurveSizes = []int{SizeL2}
		cfg.KeyTypes = []string{keyTypeUint64}
		cfg.Budget = envDuration("SET3_CMP_BUDGET", 2*time.Second)
		cfg.HardCap = envDuration("SET3_CMP_HARDCAP", 10*time.Second)
		cfg.MemRepeats = 3
	}

	if raw := os.Getenv("SET3_CMP_SIZES"); raw != "" {
		sizes, err := parseInts(raw)
		if err != nil {
			return cfg, fmt.Errorf("SET3_CMP_SIZES: %w", err)
		}
		cfg.Sizes = sizes
	}
	if raw := os.Getenv("SET3_CMP_KEYS"); raw != "" {
		cfg.KeyTypes = splitList(raw)
	}
	if raw := os.Getenv("SET3_CMP_CURVE_SIZES"); raw != "" {
		sizes, err := parseInts(raw)
		if err != nil {
			return cfg, fmt.Errorf("SET3_CMP_CURVE_SIZES: %w", err)
		}
		cfg.LoadCurveSizes = sizes
	}
	if raw := os.Getenv("SET3_CMP_SCENARIOS"); raw != "" {
		cfg.Scenarios = splitList(raw)
	}
	if raw := os.Getenv("SET3_CMP_CELLS"); raw != "" {
		cfg.Cells = splitList(raw)
	}
	cfg.Merge = envBool("SET3_CMP_MERGE", false)
	cfg.SkipMemory = envBool("SET3_CMP_SKIP_MEMORY", false)
	cfg.SkipRuntime = envBool("SET3_CMP_SKIP_RUNTIME", false)
	cfg.SkipLoadCurve = envBool("SET3_CMP_SKIP_CURVE", false)

	return cfg, cfg.validate()
}

// validate rejects the settings that would produce a meaningless run rather
// than an obviously broken one, which is the harder failure to notice.
func (c Config) validate() error {
	if len(c.Sizes) == 0 {
		return fmt.Errorf("no sizes selected")
	}
	for _, n := range c.Sizes {
		if n < 16 {
			return fmt.Errorf("size %d is too small to say anything about a hash table", n)
		}
	}
	if len(c.KeyTypes) == 0 {
		return fmt.Errorf("no key types selected")
	}
	for _, k := range c.KeyTypes {
		if !knownKeyType(k) {
			return fmt.Errorf("unknown key type %q", k)
		}
	}
	if c.Level <= 0.5 || c.Level >= 1 {
		return fmt.Errorf("level %v is not in (0.5, 1)", c.Level)
	}
	if c.MemRepeats < 1 {
		return fmt.Errorf("mem repeats %d must be at least 1", c.MemRepeats)
	}
	if c.Budget <= 0 || c.HardCap < c.Budget {
		return fmt.Errorf("budget %v / hard cap %v is not a usable pair", c.Budget, c.HardCap)
	}
	return nil
}

// wantsScenario reports whether the named workload is in this run's filter.
func (c Config) wantsScenario(name string) bool {
	if len(c.Scenarios) == 0 {
		return true
	}
	for _, s := range c.Scenarios {
		if s == name {
			return true
		}
	}
	return false
}

// String renders the configuration as the one line that goes into the run log
// and the CSV header, so a recorded result can be traced back to what produced
// it.
func (c Config) String() string {
	return fmt.Sprintf("sizes=%v curveSizes=%v keys=%v scenarios=%v budget=%v hardcap=%v resamples=%d level=%.2f memRepeats=%d tag=%q",
		c.Sizes, c.LoadCurveSizes, c.KeyTypes, c.Scenarios, c.Budget, c.HardCap, c.Resamples, c.Level, c.MemRepeats, c.Tag)
}

func parseInts(raw string) ([]int, error) {
	parts := splitList(raw)
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(strings.ReplaceAll(p, "_", ""))
		if err != nil {
			return nil, fmt.Errorf("%q is not an integer: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func splitList(raw string) []string {
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func envString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func envUint(key string, def uint64) uint64 {
	v, err := strconv.ParseUint(os.Getenv(key), 10, 64)
	if err != nil {
		return def
	}
	return v
}

func envFloat(key string, def float64) float64 {
	v, err := strconv.ParseFloat(os.Getenv(key), 64)
	if err != nil {
		return def
	}
	return v
}

func envBool(key string, def bool) bool {
	v, err := strconv.ParseBool(os.Getenv(key))
	if err != nil {
		return def
	}
	return v
}

func envDuration(key string, def time.Duration) time.Duration {
	v, err := time.ParseDuration(os.Getenv(key))
	if err != nil {
		return def
	}
	return v
}

// wantsCell reports whether one cell is in the Cells filter. An empty filter
// admits everything, which is the ordinary case.
func (c Config) wantsCell(scenario, keyType string, size int) bool {
	if len(c.Cells) == 0 {
		return true
	}
	want := fmt.Sprintf("%s/%s/%d", scenario, keyType, size)
	return slices.Contains(c.Cells, want)
}

// CellName spells a cell the way SET3_CMP_CELLS expects it, so that a tool
// reading a runtime.csv can hand the untrustworthy rows straight back.
func CellName(scenario, keyType string, size int) string {
	return fmt.Sprintf("%s/%s/%d", scenario, keyType, size)
}
