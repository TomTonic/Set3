package performance

import (
	"testing"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/rtcompare"
)

// rtcompareV060Sink keeps the compiler from folding either candidate's loop
// to a constant, per the "sink +=" advice in rtcompare's HOWTO.md.
var rtcompareV060Sink uint64

// TestUint64_WH64Det_vs_Splitmix64 follows the "five-minute version"
// protocol from rtcompare v0.6.0's HOWTO.md (Compare): it calibrates the
// batch size for both candidates, validates the harness's own noise floor
// against itself for each candidate, measures the two interleaved (ABBA),
// checks for drift and autocorrelation, and reports a bootstrap-based
// difference plus confidence at the given thresholds - all in one call,
// instead of the hand-rolled AB/BA loop used by the older tests in
// performance_test.go.
func TestUint64_WH64Det_vs_Splitmix64(t *testing.T) {
	skipRtcomparePerfIfCoverageEnabled(t)

	seed := uint64(0x1234_5678_9abc_def0)

	// Each candidate gets its own DPRNG, seeded identically, so both draw the
	// same sequence of input keys across the run - the RNG draw is per-op
	// work that cannot be hoisted into Setup (see "Attenuation" in the
	// HOWTO), but it is identical for A and B and so never biases the
	// comparison between them.
	rngWHDet := rtcompare.NewDPRNG(seed)
	rngSM := rtcompare.NewDPRNG(seed)

	wh64det := rtcompare.Candidate{
		Name: "WH64Det",
		Batch: func(n uint64) {
			var sum uint64
			for range n {
				val := rngWHDet.Uint64()
				sum ^= hashing.WH64Det(val, seed)
			}
			rtcompareV060Sink ^= sum
		},
	}

	splitmix64 := rtcompare.Candidate{
		Name: "Splitmix64",
		Batch: func(n uint64) {
			var sum uint64
			for range n {
				x := rngSM.Uint64()
				sum ^= hashing.Splitmix64(x ^ seed)
			}
			rtcompareV060Sink ^= sum
		},
	}

	report, err := rtcompare.Compare(wh64det, splitmix64, rtcompare.CompareOptions{
		Thresholds: []float64{0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2},
		// The default MaxQuantizationError (0.001) produced a high bootstrap
		// tie rate on this machine for these two very cheap functions; per
		// the HOWTO's "tie rate is high" troubleshooting entry, tightening
		// this by an order of magnitude lengthens the batches instead of
		// just drawing more samples from the same coarse set of values.
		Collect: rtcompare.CollectOptions{MaxQuantizationError: 0.00001},
	})
	if err != nil {
		t.Fatalf("rtcompare.Compare failed: %v", err)
	}

	t.Log("\n" + report.String())
	for _, w := range report.Warnings {
		t.Logf("warning: %s", w)
	}

	if report.Resolved {
		t.Logf("resolved: WH64Det vs Splitmix64 differ by %s", report.Estimate)
	} else {
		t.Logf("not resolved: this run did not establish a difference between WH64Det and Splitmix64")
	}
}
