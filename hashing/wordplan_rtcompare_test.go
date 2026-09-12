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

// This is lab code — it carries the set3lab tag and is compiled out of an
// ordinary build, like everything under lab/. It lives here rather than under
// lab/ because it measures buildWordClosure against buildClosureFromOps, and
// both are package-internal. The alternative was a hand-written replica of the
// chain in lab/hashalt, which would have been a baseline nobody could check
// against what the generator actually emits.
//
// Run it with:
//
//	go test -tags set3lab -run TestRtcompare_WordPath -v ./hashing
package hashing

import (
	"fmt"
	"math"
	"reflect"
	"runtime"
	"runtime/debug"
	"testing"
	"unsafe"

	"github.com/TomTonic/rtcompare"
)

var rtWordSink uint64

// rtWordShapes are the type shapes the comparison covers. Each must take the
// word path, which buildBothForShape checks.
func rtWordShapes() []struct {
	name string
	typ  reflect.Type
} {
	f64 := reflect.TypeOf(float64(0))
	f32 := reflect.TypeOf(float32(0))
	return []struct {
		name string
		typ  reflect.Type
	}{
		{"2xfloat64", reflect.ArrayOf(2, f64)},
		{"3xfloat64", reflect.ArrayOf(3, f64)},
		{"4xfloat64", reflect.ArrayOf(4, f64)},
		{"6xfloat64", reflect.ArrayOf(6, f64)},
		{"8xfloat64", reflect.ArrayOf(8, f64)},
		{"3xfloat32", reflect.ArrayOf(3, f32)},
		{"6xfloat32", reflect.ArrayOf(6, f32)},
		{"8xfloat32", reflect.ArrayOf(8, f32)},
		{"2xcomplex128", reflect.TypeOf(wpComplex2{})},
		{"int64x2+float64", reflect.TypeOf(wpBlock16Float{})},
		{"2xfloat64+string", reflect.TypeOf(wpFloatString{})},
	}
}

// buildBothForShape returns the word closure and the chain closure the
// generator builds for ty, plus a pointer to a value of that type.
func buildBothForShape(t *testing.T, ty reflect.Type) (words, chain HashFunction, p unsafe.Pointer) {
	t.Helper()
	ops := mergeByteBlocks(flattenTypeOps(ty, 0))
	words = buildWordClosure(ops)
	if words == nil {
		t.Fatalf("%s does not take the word path; the comparison would be chain against chain", ty)
	}
	chain = buildClosureFromOps(ops)

	v := reflect.New(ty)
	fillForHashing(v.Elem())
	return words, chain, v.UnsafePointer()
}

// fillForHashing puts distinguishable, finite values into every field.
func fillForHashing(v reflect.Value) {
	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		v.SetFloat(1.5)
	case reflect.Complex64, reflect.Complex128:
		v.SetComplex(complex(1.5, -2.5))
	case reflect.String:
		v.SetString("vienna")
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(11)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		v.SetUint(11)
	case reflect.Array:
		for i := range v.Len() {
			fillForHashing(v.Index(i))
			if v.Index(i).CanFloat() {
				v.Index(i).SetFloat(float64(i) + 1.5)
			}
		}
	case reflect.Struct:
		for i := range v.NumField() {
			fillForHashing(v.Field(i))
		}
	}
}

// runHashes calls fn count times over the same value, threading the result so
// that nothing can be folded away.
func runHashes(fn HashFunction, p unsafe.Pointer, seed, count uint64) uint64 {
	h := seed
	for range count {
		h ^= fn(p, seed)
	}
	return h
}

// TestRtcompare_WordPathVsChain measures the two closures the generator can
// build for one type, ABBA-interleaved so that neither side owns a particular
// stretch of the machine's behaviour, and reports the paired bootstrap
// confidence rather than a single ratio.
func TestRtcompare_WordPathVsChain(t *testing.T) {
	if testing.CoverMode() != "" {
		t.Skip("coverage instrumentation perturbs hot paths; run without -cover")
	}

	const (
		repeats        = 401
		rounds         = 1_000_000
		precisionLevel = 10_000
		seed           = uint64(0x243f6a8885a308d3)
	)
	speedups := []float64{0.05, 0.10, 0.20, 0.30, 0.40, 0.50}

	fmt.Printf("\nRTCOMPARE word path vs seed-threading chain\n")
	fmt.Printf("  repeats=%d rounds=%d precisionLevel=%d\n\n", repeats, rounds, precisionLevel)

	for _, shape := range rtWordShapes() {
		words, chain, p := buildBothForShape(t, shape.typ)

		// Warm both sides before the first timed round.
		rtWordSink ^= runHashes(words, p, seed, 4096)
		rtWordSink ^= runHashes(chain, p, seed, 4096)

		timesWords := make([]float64, 0, repeats)
		timesChain := make([]float64, 0, repeats)

		gcval := debug.SetGCPercent(-1)
		debug.SetGCPercent(gcval)

		for i := range uint64(repeats) {
			var dWords, dChain int64
			measure := func(fn HashFunction) int64 {
				runtime.GC()
				debug.SetGCPercent(-1)
				t0 := rtcompare.SampleTime()
				rtWordSink ^= runHashes(fn, p, seed, rounds)
				t1 := rtcompare.SampleTime()
				debug.SetGCPercent(gcval)
				return rtcompare.DiffTimeStamps(t0, t1)
			}
			if i%2 == 0 {
				dWords = measure(words)
				dChain = measure(chain)
			} else {
				dChain = measure(chain)
				dWords = measure(words)
			}
			timesWords = append(timesWords, float64(dWords)/float64(rounds))
			timesChain = append(timesChain, float64(dChain)/float64(rounds))
		}
		debug.SetGCPercent(gcval)

		reportWordPathPair(t, shape.name, timesChain, timesWords, speedups, precisionLevel)
	}
}

// reportWordPathPair prints the paired summary for one shape. A is the chain,
// B is the word path, so the interesting direction is B faster than A.
func reportWordPathPair(t *testing.T, name string, timesChain, timesWords, speedups []float64, precisionLevel uint64) {
	t.Helper()

	logRatio := make([]float64, 0, len(timesChain))
	wordsOverChain := make([]float64, 0, len(timesChain))
	ones := make([]float64, 0, len(timesChain))
	for i := range timesChain {
		a, b := timesChain[i], timesWords[i]
		if a <= 0 || b <= 0 {
			t.Fatalf("%s: invalid sample at pair %d: chain=%g words=%g", name, i, a, b)
		}
		lr := math.Log(b) - math.Log(a)
		logRatio = append(logRatio, lr)
		wordsOverChain = append(wordsOverChain, math.Exp(lr))
		ones = append(ones, 1.0)
	}

	// CompareSamples asks "is the first sample set at least X% faster", and
	// faster means smaller, so the ratio to hand it is words/chain — below one
	// exactly when the word path wins.
	faster, err := rtcompare.CompareSamples(wordsOverChain, ones, speedups, precisionLevel)
	if err != nil {
		t.Fatalf("%s: rtcompare.CompareSamples failed: %v", name, err)
	}

	medChain := rtcompare.Median(timesChain)
	medWords := rtcompare.Median(timesWords)
	medLR := rtcompare.Median(logRatio)
	meanLR, _, sdLR := rtcompare.Statistics(logRatio)

	fmt.Printf("  %-18s chain %7.3f ns  words %7.3f ns   words are %.1f%% faster (median), %.1f%% (geometric mean)\n",
		name, medChain, medWords,
		(1-math.Exp(medLR))*100, (1-math.Exp(meanLR))*100)
	fmt.Printf("  %-18s paired log-ratio median=%.6f mean=%.6f sd=%.6f\n", "", medLR, meanLR, sdLR)
	fmt.Printf("  %-18s confidence the word path is faster by at least:", "")
	for _, r := range faster {
		fmt.Printf(" %.0f%%->%.1f%%", r.RelativeSpeedupSampleAvsSampleB*100, r.Confidence*100)
	}
	fmt.Printf("\n\n")

	t.Logf("%s: chain %.3f ns, words %.3f ns, median ratio %.4f", name, medChain, medWords, math.Exp(medLR))
}
