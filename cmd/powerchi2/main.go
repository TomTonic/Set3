package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"strconv"
	"strings"

	"github.com/TomTonic/rtcompare"
	"gonum.org/v1/gonum/stat/distuv"
)

/*
Power and sample-size calculations for the Chi^2 goodness-of-fit test.

Principles:
- Critical value c = χ²_{1-α, df} from the central chi-square (Gonum: distuv.ChiSquared.Quantile).
- Noncentrality parameter λ = n * Σ ((p_i - q_i)^2 / q_i).
- Power = P(χ²_{df,λ} > c) = 1 - F_{χ²_{df,λ}}(c).
	The noncentral chi-square CDF is computed via a Poisson mixture:
	F(x;df,λ) = Σ_{i=0..∞} e^{-λ/2} (λ/2)^i / i! * F_central(x; df+2i).

Example calls:

1) Power with n for "Six is slightly more frequent"
H₀: qi=1/6q_i=1/6qi​=1/6 (uniform for k=6k=6k=6)
H₁: p6=0.18p_6=0.18p6​=0.18, rest equally distributed (0.82/5)(0.82/5)(0.82/5)

# Power with n=11000, α=0.05
go run . \
  -mode power \
  -k 6 \
  -p "0.164,0.164,0.164,0.164,0.164,0.18" \
  -alpha 0.05 \
  -n 11000 \
  -mc 20000


2)  Required sample size for target power 0.80

go run . \
  -mode n \
  -k 6 \
  -p "0.164,0.164,0.164,0.164,0.164,0.18" \
  -alpha 0.05 \
  -target 0.80 \
  -mc 10000


*/

// ---------- Parsing & Utilities ----------

func parseProbList(s string) ([]float64, error) {
	parts := strings.Split(s, ",")
	res := make([]float64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, fmt.Errorf("could not convert %q to float: %w", p, err)
		}
		res = append(res, v)
	}
	if len(res) == 0 {
		return nil, errors.New("empty list")
	}
	return res, nil
}

// normalize: scales a so that the sum is 1.0.
// Returns a new slice.
// If the sum is 0, returns a zero slice.
// Useful for converting unnormalized proportions into probabilities.
func normalize(a []float64) []float64 {
	s := 0.0
	for _, v := range a {
		s += v
	}
	out := make([]float64, len(a))
	if s == 0 {
		return out
	}
	for i := range a {
		out[i] = a[i] / s
	}
	return out
}

// checkProbs: checks whether p is a valid probability distribution.
// Returns an error if...
// - any entry < 0
// - the sum is not 1.0 (within a tolerance of 1e-9)
func checkProbs(p []float64, name string) error {
	for i, v := range p {
		if v < 0 {
			return fmt.Errorf("%s[%d]=%g < 0", name, i, v)
		}
	}
	s := 0.0
	for _, v := range p {
		s += v
	}
	if math.Abs(s-1.0) > 1e-9 {
		return fmt.Errorf("%s sums to %.12f (not 1.0)", name, s)
	}
	return nil
}

// ---------- Statistik: zentrale und nichtzentrale Chi^2 ----------

// centralChi2Quantile liefert den kritischen Wert c = χ²_{1-α, df}
func centralChi2Quantile(df, alpha float64) float64 {
	return distuv.ChiSquared{K: df}.Quantile(1 - alpha)
}

// noncentralChiSquareCDF computes F_{χ²_{df,λ}}(x) via a Poisson mixture.
// tol: tail-weight threshold (e.g. 1e-12)
func noncentralChiSquareCDF(x, df, lambda, tol float64) float64 {
	if lambda < 0 || df <= 0 || x < 0 {
		return math.NaN()
	}
	// Startgewicht i=0
	w := math.Exp(-lambda / 2)
	cdf := distuv.ChiSquared{K: df}.CDF(x)
	sum := w * cdf
	cumW := w

	// heuristische Obergrenze (abhängig von meanI = λ/2)
	meanI := lambda / 2
	maxI := int(meanI + 10*math.Sqrt(meanI) + 50) // großzügig
	if maxI < 100 {
		maxI = 100
	}

	for i := 1; i <= maxI; i++ {
		w *= (lambda / 2) / float64(i)
		cumW += w
		dfi := df + 2*float64(i)
		sum += w * distuv.ChiSquared{K: dfi}.CDF(x)
		if 1.0-cumW < tol {
			break
		}
	}
	return sum
}

// ---------- Test-specific quantities ----------

// lambda = n * Σ ((dH1_i - dH0_i)^2 / dH0_i)
func noncentralityLambda(n uint64, dH1, dH0 []float64) float64 {
	if len(dH1) != len(dH0) {
		return math.NaN()
	}
	s := 0.0
	for i := range dH1 {
		d := dH1[i] - dH0[i]
		s += (d * d) / dH0[i]
	}
	return float64(n) * s
}

// powerChi2GoodnessOfFit computes the power for a given n.
func powerChi2GoodnessOfFit(n uint64, alpha float64, dH1, dH0 []float64, tol float64) float64 {
	df := float64(len(dH1) - 1)
	chi2CriticalVal := centralChi2Quantile(df, alpha)
	lambda := noncentralityLambda(n, dH1, dH0)
	cdf := noncentralChiSquareCDF(chi2CriticalVal, df, lambda, tol)
	return 1.0 - cdf
}

// solveNForTargetPower: find the smallest n with Power >= targetPower
func solveNForTargetPower(targetPower, alpha float64, dH1, dH0 []float64, tol float64, nMax uint64) (int64, float64) {
	if nMax <= 0 {
		nMax = math.MaxUint32
	}
	lo := uint64(1)
	hi := uint64(32)
	// exponential increase to bracket the solution
	for powerChi2GoodnessOfFit(hi, alpha, dH1, dH0, tol) < targetPower && hi < nMax {
		hi *= 2
	}
	if hi >= nMax && powerChi2GoodnessOfFit(hi, alpha, dH1, dH0, tol) < targetPower {
		return -1, powerChi2GoodnessOfFit(hi, alpha, dH1, dH0, tol)
	}
	// bisection
	for iter := 0; iter < 60 && lo < hi; iter++ {
		mid := (lo + hi) / 2
		pw := powerChi2GoodnessOfFit(mid, alpha, dH1, dH0, tol)
		if pw < targetPower {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	n := hi
	return int64(n), powerChi2GoodnessOfFit(n, alpha, dH1, dH0, tol)
}

// ---------- Monte Carlo validation ----------

// multinomial sample: one draw with K categories according to probs
func drawIndex(probs []float64, rng *rtcompare.CPRNG) int {
	u := rng.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if u <= cum {
			return i
		}
	}
	return len(probs) - 1
}

// Monte Carlo power: simulates multinomial data under dH1, tests against dH0.
func mcPower(sampleSize uint64, chi2CriticalVal float64, dH1, dH0 []float64, trials uint64) float64 {
	rng := rtcompare.NewCPRNG(16384)
	rejects := 0
	N := float64(sampleSize)

	for range trials {
		// draw
		counts := make([]uint32, len(dH1))
		for range sampleSize {
			idx := drawIndex(dH1, rng)
			counts[idx]++
		}
		// Chi^2-Statistik
		chi2 := 0.0
		for i := range dH1 {
			exp := N * dH0[i]
			obs := float64(counts[i])
			chi2 += (obs - exp) * (obs - exp) / exp
		}
		if chi2 > chi2CriticalVal {
			rejects++
		}
	}
	return float64(rejects) / float64(trials)
}

// ---------- CLI ----------

func main() {
	// Modes:
	//   -mode power : compute power for a given n
	//   -mode n     : compute minimal n for a target power
	mode := flag.String("mode", "power", "mode: power | n")

	// Hypotheses:
	// q = H0 distribution (comma-separated). If empty, -k must be given → q = uniform.
	qStr := flag.String("q", "", "H0 distribution (comma-separated); alternatively -k")
	k := flag.Int("k", 0, "number of categories (only if -q empty; then q = uniform)")

	// p = H1 distribution (comma-separated; must have same length as q)
	pStr := flag.String("p", "", "H1 distribution (comma-separated) [required]")

	// Test parameters
	alpha := flag.Float64("alpha", 0.05, "significance level alpha")
	samples := flag.Uint64("n", 1000, "sample size n (for mode=power)")
	targetPower := flag.Float64("target", 0.80, "target power (for mode=n)")
	tol := flag.Float64("tol", 1e-12, "tolerance for Poisson tail weight in noncentral CDF")

	// Monte Carlo
	mc := flag.Uint64("mc", 0, "Monte Carlo simulation runs for validation of analytically calculated power (0 = off)")

	flag.Parse()

	// p und dH0 bestimmen
	var dH0 []float64
	var err error

	if *qStr != "" {
		dH0, err = parseProbList(*qStr)
		if err != nil {
			log.Fatal("error parsing -q: ", err)
		}
	} else {
		if *k <= 1 {
			log.Fatal("please provide either -q (comma list) or -k (>=2) for uniform H0")
		}
		dH0 = make([]float64, *k)
		for i := range dH0 {
			dH0[i] = 1.0 / float64(*k)
		}
	}

	if *pStr == "" {
		log.Fatal("please provide -p (comma list of the H1 distribution)")
	}
	dH1, err := parseProbList(*pStr)
	if err != nil {
		log.Fatal("error parsing -p: ", err)
	}

	// Längen abgleichen & normalisieren (falls der Nutzer unnormierte Anteile eingibt)
	if len(dH1) != len(dH0) {
		log.Fatalf("Lengths differ: len(p)=%d != len(q)=%d", len(dH1), len(dH0))
	}
	dH1 = normalize(dH1)
	dH0 = normalize(dH0)

	// Validieren
	if err := checkProbs(dH0, "q"); err != nil {
		log.Fatal("H0 invalid: ", err)
	}
	if err := checkProbs(dH1, "p"); err != nil {
		log.Fatal("H1 invalid: ", err)
	}
	if len(dH1) < 2 {
		log.Fatal("At least 2 categories required.")
	}
	if *alpha <= 0 || *alpha >= 1 {
		log.Fatal("alpha must be in (0,1).")
	}
	df := float64(len(dH1) - 1)

	switch *mode {
	case "power":
		pw := powerChi2GoodnessOfFit(*samples, *alpha, dH1, dH0, *tol)
		c := centralChi2Quantile(df, *alpha)
		lambda := noncentralityLambda(*samples, dH1, dH0)

		fmt.Printf("Mode       : power\n")
		fmt.Printf("Categories : %d (df=%.0f)\n", len(dH1), df)
		fmt.Printf("alpha      : %.4f\n", *alpha)
		fmt.Printf("n          : %d\n", *samples)
		fmt.Printf("critical value c (central) : %.6f\n", c)
		fmt.Printf("noncentrality λ            : %.6f\n", lambda)
		fmt.Printf("Power (analytic)           : %.6f\n", pw)

		if *mc > 0 {
			df := float64(len(dH1) - 1)
			chi2CriticalVal := centralChi2Quantile(df, *alpha)
			pwmc := mcPower(*samples, chi2CriticalVal, dH1, dH0, *mc)
			fmt.Printf("Power (Monte Carlo, %d runs) : %.6f\n", *mc, pwmc)
		}

	case "n":
		nNeeded, pw := solveNForTargetPower(*targetPower, *alpha, dH1, dH0, *tol, 0)
		if nNeeded < 0 {
			log.Fatal("target power not reachable within nMax.")
		}
		chi2CriticalVal := centralChi2Quantile(df, *alpha)
		lambda := noncentralityLambda(uint64(nNeeded), dH1, dH0)

		fmt.Printf("Mode         : n\n")
		fmt.Printf("Categories   : %d (df=%.0f)\n", len(dH1), df)
		fmt.Printf("alpha        : %.4f\n", *alpha)
		fmt.Printf("Target Power : %.4f\n", *targetPower)
		fmt.Printf("required n   : %d\n", nNeeded)
		fmt.Printf("critical value c (central) : %.6f\n", chi2CriticalVal)
		fmt.Printf("noncentrality λ (at n)     : %.6f\n", lambda)
		fmt.Printf("Power (at n)               : %.6f\n", pw)

		if *mc > 0 {
			df := float64(len(dH1) - 1)
			chi2CriticalVal := centralChi2Quantile(df, *alpha)
			pwmc := mcPower(uint64(nNeeded), chi2CriticalVal, dH1, dH0, *mc)
			fmt.Printf("Power (Monte Carlo, %d runs) : %.6f\n", *mc, pwmc)
		}

	default:
		log.Fatalf("Unknown mode=%q (allowed: power | n)", *mode)
	}
}
