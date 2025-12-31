package main

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/TomTonic/rtcompare"
)

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestNormalize(t *testing.T) {
	tol := 1e-12

	cases := []struct {
		name           string
		in             []float64
		expectSumOne   bool
		expectAllZero  bool
		expectSameVals []float64 // if non-nil, expected exact values (approx)
	}{
		{
			name:         "positive values",
			in:           []float64{1, 2, 3},
			expectSumOne: true,
		},
		{
			name:         "already normalized",
			in:           []float64{0.2, 0.3, 0.5},
			expectSumOne: true,
		},
		{
			name:          "zero sum",
			in:            []float64{0, 0, 0},
			expectSumOne:  false,
			expectAllZero: true,
		},
		{
			name:         "single element",
			in:           []float64{5},
			expectSumOne: true,
			expectSameVals: []float64{
				1.0,
			},
		},
		{
			name:         "negative values (sum != 0)",
			in:           []float64{-1, -2, -3},
			expectSumOne: true,
		},
		{
			name:           "empty slice",
			in:             []float64{},
			expectSumOne:   false,
			expectAllZero:  false,
			expectSameVals: []float64{},
		},
	}

	for _, tc := range cases {
		orig := make([]float64, len(tc.in))
		copy(orig, tc.in)

		out := normalize(tc.in)

		// length must match
		if len(out) != len(tc.in) {
			t.Fatalf("%s: length mismatch: got %d want %d", tc.name, len(out), len(tc.in))
		}

		// original must not be modified
		if !reflect.DeepEqual(tc.in, orig) {
			t.Fatalf("%s: input slice was modified", tc.name)
		}

		// check sum properties
		sum := 0.0
		for _, v := range out {
			sum += v
		}

		if tc.expectAllZero {
			for i, v := range out {
				if v != 0 {
					t.Fatalf("%s: expected all zeros but out[%d]=%v", tc.name, i, v)
				}
			}
			continue
		}

		if tc.expectSumOne {
			if !approxEqual(sum, 1.0, 1e-9) {
				t.Fatalf("%s: sum=%v, want 1.0", tc.name, sum)
			}
		} else {
			// when not expecting sum 1 (e.g. zero input or empty), ensure behaviour is consistent:
			// for empty input, sum should be 0; for all-zero input we already checked.
			if len(tc.in) == 0 && sum != 0 {
				t.Fatalf("%s: empty input -> expected sum 0, got %v", tc.name, sum)
			}
		}

		// if exact expected values provided, check them approximately
		if tc.expectSameVals != nil {
			if len(tc.expectSameVals) != len(out) {
				t.Fatalf("%s: expectSameVals length mismatch", tc.name)
			}
			for i := range out {
				if !approxEqual(out[i], tc.expectSameVals[i], tol) {
					t.Fatalf("%s: out[%d]=%v want %v", tc.name, i, out[i], tc.expectSameVals[i])
				}
			}
		}
	}
}

func TestCheckProbs(t *testing.T) {
	cases := []struct {
		name          string
		p             []float64
		argName       string
		wantErr       bool
		wantSubstring string
	}{
		{
			name:    "valid distribution",
			p:       []float64{0.2, 0.3, 0.5},
			argName: "q",
			wantErr: false,
		},
		{
			name:          "negative entry",
			p:             []float64{0.1, -0.01, 0.91},
			argName:       "p",
			wantErr:       true,
			wantSubstring: "< 0",
		},
		{
			name:          "sum not one (too small)",
			p:             []float64{0.1, 0.2, 0.3},
			argName:       "h0",
			wantErr:       true,
			wantSubstring: "sums to",
		},
		{
			name:    "sum within tolerance",
			p:       []float64{0.5, 0.5 + 5e-10},
			argName: "alpha",
			wantErr: false,
		},
		{
			name:          "empty slice -> sum 0",
			p:             []float64{},
			argName:       "empty",
			wantErr:       true,
			wantSubstring: "sums to",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// preserve input
			orig := make([]float64, len(tc.p))
			copy(orig, tc.p)

			err := checkProbs(tc.p, tc.argName)

			// input must not be modified
			if !reflect.DeepEqual(orig, tc.p) {
				t.Fatalf("input was modified")
			}

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				if tc.wantSubstring != "" && !strings.Contains(err.Error(), tc.wantSubstring) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.wantSubstring)
				}
				// also expect the name to appear in the message
				if !strings.Contains(err.Error(), tc.argName) {
					t.Fatalf("error %q does not contain argument name %q", err.Error(), tc.argName)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}

// These are regression tests that compare the result of centralChi2Quantile
// to previously recorded constant values. They do not recompute the
// expected values at test time.
func TestCentralChi2QuantileConstants(t *testing.T) {
	tests := []struct {
		df    float64
		alpha float64
		want  float64
	}{
		// see https://en.wikipedia.org/wiki/Chi-squared_distribution#Table_of_%CF%872_values_vs_p-values
		{1, 0.001, 10.827566170663},  // df=1, 99.9% quantile
		{1, 0.05, 3.841458820694124}, // df=1, 95% quantile
		{1, 0.95, 0.003932140000},    // df=1, 5% quantile
		{2, 0.05, 5.991464547107979}, // df=2, 95% quantile
		{3, 0.1, 6.251388631170},     // df=3, 90% quantile
		{7, 0.7, 4.671330448981},     // df=3, 30% quantile
		{10, 0.001, 29.588298445074}, // df=10, 99.9% quantile
		{10, 0.3, 11.780722627394},   // df=10, 70% quantile
		{10, 0.95, 3.940299136119},   // df=10, 5% quantile
	}

	const tol = 1e-9
	for _, tc := range tests {
		got := centralChi2Quantile(tc.df, tc.alpha)
		if math.Abs(got-tc.want) > tol {
			t.Fatalf("centralChi2Quantile(df=%.0f, alpha=%g) = %.12f; want %.12f", tc.df, tc.alpha, got, tc.want)
		}
	}
}
func TestNoncentralChiSquareCDF_InvalidInputs(t *testing.T) {
	tol := 1e-12

	if !math.IsNaN(noncentralChiSquareCDF(1.0, 2.0, -0.1, tol)) {
		t.Fatalf("expected NaN for negative lambda")
	}
	if !math.IsNaN(noncentralChiSquareCDF(1.0, 0.0, 1.0, tol)) {
		t.Fatalf("expected NaN for non-positive df")
	}
	if !math.IsNaN(noncentralChiSquareCDF(-0.1, 2.0, 1.0, tol)) {
		t.Fatalf("expected NaN for negative x")
	}
}

func TestNoncentralChiSquareCDF_LambdaZeroEqualsCentral(t *testing.T) {
	tol := 1e-12
	cases := []struct {
		df    float64
		alpha float64
	}{
		{1, 0.001},
		{1, 0.05},
		{1, 0.95},
		{2, 0.05},
		{3, 0.1},
		{10, 0.3},
	}

	for _, tc := range cases {
		c := centralChi2Quantile(tc.df, tc.alpha)
		got := noncentralChiSquareCDF(c, tc.df, 0.0, tol)
		want := 1.0 - tc.alpha
		if !approxEqual(got, want, 1e-12) {
			t.Fatalf("df=%.0f alpha=%g: got %v want %v", tc.df, tc.alpha, got, want)
		}
	}
}

func TestNoncentralChiSquareCDF_MonotonicInLambda(t *testing.T) {
	tol := 1e-12
	df := 3.0
	x := centralChi2Quantile(df, 0.5) // median-like point

	got0 := noncentralChiSquareCDF(x, df, 0.0, tol)
	got2 := noncentralChiSquareCDF(x, df, 2.0, tol)
	got10 := noncentralChiSquareCDF(x, df, 10.0, tol)

	if !(got0 > got2 && got2 > got10) {
		t.Fatalf("CDF not decreasing with lambda: got0=%v got2=%v got10=%v", got0, got2, got10)
	}
}

func TestNoncentralChiSquareCDF_Boundaries(t *testing.T) {
	tol := 1e-12
	df := 4.0

	// at x=0 CDF should be 0 for df>0
	gotZero := noncentralChiSquareCDF(0.0, df, 5.0, tol)
	if !approxEqual(gotZero, 0.0, 0.0) {
		t.Fatalf("CDF at x=0 expected 0 got %v", gotZero)
	}

	// at very large x CDF should be ~1
	gotLarge := noncentralChiSquareCDF(1e6, df, 7.0, tol)
	if !approxEqual(gotLarge, 1.0, 1e-12) {
		t.Fatalf("CDF at large x expected ~1 got %v", gotLarge)
	}
}

func TestNoncentralChiSquareCDF_Constants(t *testing.T) {
	tol := 1e-9
	cases := []struct {
		x      float64
		df     float64
		lambda float64
		want   float64
	}{
		// compute values via Google Colab, for example:
		// paste&run the following Python code at https://colab.research.google.com
		//   from scipy.stats import ncx2
		//   x, df, lam = 5.0, 2, 3.0
		//   print(ncx2.cdf(x, df, nc=lam))
		{x: 5.0, df: 2.0, lambda: 3.0, want: 0.5940608030781964},
		{x: 10.0, df: 5.0, lambda: 0.1, want: 0.9190377639698465},
		{x: 3.33, df: 17.0, lambda: 2.7, want: 4.7614699056624424e-05},
		{x: 77.0, df: 2.0, lambda: 21.7, want: 0.9999732101247713},
		{x: 77.0, df: 2.0, lambda: 1.7, want: 0.999999999999895},
		{x: 77.0, df: 22.0, lambda: 41.7, want: 0.8236022067253619},
	}

	for _, tc := range cases {
		got := noncentralChiSquareCDF(tc.x, tc.df, tc.lambda, tol)
		if !approxEqual(got, tc.want, tol) {
			t.Fatalf("CDF(x=%.1f, df=%.1f, lambda=%.1f) = %.10f; want %.10f", tc.x, tc.df, tc.lambda, got, tc.want)
		}
	}
}
func TestMcPower_ExtremesAndDeterminism(t *testing.T) {
	// simple 2-category distributions
	dH0 := []float64{0.5, 0.5}
	dH1 := []float64{0.5, 0.5}
	sampleSize := uint64(100)
	trials := uint64(500)

	// extreme: very low threshold -> always reject
	pwAll := mcPower(sampleSize, -1.0, dH1, dH0, trials)
	if pwAll != 1.0 {
		t.Fatalf("expected always reject (power=1.0) for chi2CriticalVal=-1, got %v", pwAll)
	}

	// extreme: very high threshold -> never reject
	pwNone := mcPower(sampleSize, 1e12, dH1, dH0, trials)
	if pwNone != 0.0 {
		t.Fatalf("expected never reject (power=0.0) for huge chi2CriticalVal, got %v", pwNone)
	}

	// determinism: calling twice with same args yields identical result
	pw1 := mcPower(sampleSize, 1.5, []float64{0.3, 0.7}, []float64{0.5, 0.5}, trials)
	pw2 := mcPower(sampleSize, 1.5, []float64{0.3, 0.7}, []float64{0.5, 0.5}, trials)
	if !approxEqual(pw1, pw2, 0.01) {
		t.Fatalf("expected comparable output for same args, got %v and %v", pw1, pw2)
	}
}

func TestDrawIndex_EmpiricalFrequencies(t *testing.T) {
	probs := []float64{0.1, 0.2, 0.3, 0.4}
	N := 200_000
	rng := rtcompare.NewCPRNG(8192)
	counts := make([]int, len(probs))
	for range N {
		idx := drawIndex(probs, rng)
		counts[idx]++
	}
	tol := 0.01 // empirical tolerance
	for i, want := range probs {
		got := float64(counts[i]) / float64(N)
		if math.Abs(got-want) > tol {
			t.Fatalf("empirical freq for index %d = %v, want %v (tol %v)", i, got, want, tol)
		}
	}
}

func TestNoncentralityLambdaConstants(t *testing.T) {
	tests := []struct {
		n    uint64
		dH1  []float64
		dH0  []float64
		want float64
	}{
		// identical distributions -> lambda == 0
		{10, []float64{0.5, 0.5}, []float64{0.5, 0.5}, 0.0},

		// two categories: dH0=[0.5,0.5], dH1=[0.6,0.4], n=100 =>
		// d=[0.1,-0.1], sum = 2 * (0.1^2 / 0.5) = 0.04 => lambda = 100 * 0.04 = 4
		{100, []float64{0.6, 0.4}, []float64{0.5, 0.5}, 4.0},

		// three categories: dH0=[1/3,1/3,1/3], dH1=[1/2,1/4,1/4], n=50
		// calculation gives sum = 1/8 => lambda = 50 * 1/8 = 6.25
		{50, []float64{0.5, 0.25, 0.25}, []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, 6.25},
	}

	const tol = 1e-12
	for _, tc := range tests {
		got := noncentralityLambda(tc.n, tc.dH1, tc.dH0)
		if math.Abs(got-tc.want) > tol {
			t.Fatalf("noncentralityLambda(n=%d, dH1=%v, dH0=%v) = %g; want %g", tc.n, tc.dH1, tc.dH0, got, tc.want)
		}
	}
}

func TestNoncentralityLambdaLenMismatchReturnsNaN(t *testing.T) {
	got := noncentralityLambda(10, []float64{0.5}, []float64{0.5, 0.5})
	if !math.IsNaN(got) {
		t.Fatalf("expected NaN for mismatched lengths, got %v", got)
	}
}

// Tests for powerChi2GoodnessOfFit using reference values.
// Reference values computed with SciPy (scipy.stats.ncx2).
func TestPowerChi2GoodnessOfFitConstants(t *testing.T) {
	const tol = 1e-9

	tests := []struct {
		name  string
		n     uint64
		alpha float64
		dH1   []float64
		dH0   []float64
		want  float64
	}{
		{
			name:  "identical_distributions_should_give_alpha",
			n:     10,
			alpha: 0.05,
			dH1:   []float64{0.5, 0.5},
			dH0:   []float64{0.5, 0.5},
			want:  0.05, // power ≈ alpha when H0 is true
		},
		{
			name:  "two_cat_small_deviation",
			n:     100,
			alpha: 0.05,
			dH1:   []float64{0.6, 0.4},
			dH0:   []float64{0.5, 0.5},
			// lambda = 100 * ((0.1)^2/0.5 + (-0.1)^2/0.5) = 100 * 0.04 = 4.0
			// df=1, c=3.841459, power from Go implementation
			want: 0.5160052739761745,
		},
		{
			name:  "three_cat_moderate_deviation",
			n:     50,
			alpha: 0.01,
			dH1:   []float64{0.5, 0.25, 0.25},
			dH0:   []float64{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
			// lambda = 50 * sum((p-q)^2/q) = 50 * 1/8 = 6.25
			// df=2, c=qchisq(0.99,2)=9.21034, power from Go implementation
			want: 0.36338744044242033,
		},
		{
			name:  "six_cat_readme_example",
			n:     11000,
			alpha: 0.05,
			dH1:   []float64{0.164, 0.164, 0.164, 0.164, 0.164, 0.18},
			dH0:   []float64{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
			// lambda ≈ 4.07, df=5, c≈11.0705, power from Go implementation
			want: 0.8409809919024405,
		},
		{
			name:  "large_n_high_power",
			n:     1000,
			alpha: 0.05,
			dH1:   []float64{0.7, 0.3},
			dH0:   []float64{0.5, 0.5},
			// lambda = 1000 * ((0.2)^2/0.5 + (-0.2)^2/0.5) = 1000 * 0.16 = 160
			// df=1, very large lambda -> power ≈ 1.0
			want: 1.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := powerChi2GoodnessOfFit(tc.n, tc.alpha, tc.dH1, tc.dH0, 1e-12)
			if math.Abs(got-tc.want) > tol {
				t.Errorf("powerChi2GoodnessOfFit(n=%d, alpha=%g, dH1=%v, dH0=%v) = %g; want %g (diff=%g)",
					tc.n, tc.alpha, tc.dH1, tc.dH0, got, tc.want, math.Abs(got-tc.want))
			}
		})
	}
}

// Regression test for the six-sided-die example from the README.
// This checks that solveNForTargetPower returns the previously recorded
// required sample size (constant-regression style).
func TestSolveNForDieExampleConstant(t *testing.T) {
	p := []float64{0.164, 0.164, 0.164, 0.164, 0.164, 0.18}
	q := []float64{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0}
	target := 0.80
	alpha := 0.05
	tol := 1e-12

	wantN := int64(10022)

	n, pw := solveNForTargetPower(target, alpha, p, q, tol, 0)
	if n != wantN {
		t.Fatalf("solveNForTargetPower returned n=%d; want %d", n, wantN)
	}

	// also check that reported power at that n is near the expected value
	// (value observed from the implementation previously)
	wantPw := 0.80002
	if math.Abs(pw-wantPw) > 1e-5 {
		t.Fatalf("power at n=%d = %g; want approx %g", n, pw, wantPw)
	}
}

// TestMonteCarloComparison runs a short Monte Carlo simulation and verifies
// that the empirical (simulated) power is close to the analytic power.
// Uses deterministic CPRNG inside mcPower for reproducibility.
func TestMonteCarloComparison(t *testing.T) {
	// Use the README six-category example
	dH1 := normalize([]float64{0.164, 0.164, 0.164, 0.164, 0.164, 0.18})
	dH0 := normalize([]float64{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0})
	n := uint64(11000)
	alpha := 0.05
	tol := 1e-12

	analytic := powerChi2GoodnessOfFit(n, alpha, dH1, dH0, tol)

	// Monte Carlo runs: moderate to keep tests fast but reasonably precise.
	trials := uint64(2000)
	chi2CriticalVal := centralChi2Quantile(float64(len(dH1)-1), alpha)
	mc := mcPower(n, chi2CriticalVal, dH1, dH0, trials)

	// allow 1% absolute deviation
	if math.Abs(mc-analytic) > 0.01 {
		t.Fatalf("Monte Carlo power %v differs from analytic %v by >0.01", mc, analytic)
	}
}
