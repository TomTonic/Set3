package hashing

// ── Golden Ratio: (sqrt(5)-1)/2 scaled to various bit widths ──────────────

// The golden ratio (sqrt(5)-1)/2, scaled to each bit width and rounded to an
// odd integer. Multiplying by an odd constant is invertible, so no input bits
// are lost; the golden ratio spreads consecutive inputs about as evenly as a
// single multiply can.
const (
	GoldenRatio64 = 0x9E3779B97F4A7C15 // (sqrt(5)-1)/2 * 2^64
	GoldenRatio56 = 0x009E3779B97F4A7D // (sqrt(5)-1)/2 * 2^56
	GoldenRatio48 = 0x00009E3779B97F4B // (sqrt(5)-1)/2 * 2^48
	GoldenRatio40 = 0x0000009E3779B97F // (sqrt(5)-1)/2 * 2^40
	GoldenRatio32 = 0x000000009E3779B9 // (sqrt(5)-1)/2 * 2^32
	GoldenRatio24 = 0x00000000009E3779 // (sqrt(5)-1)/2 * 2^24
	GoldenRatio16 = 0x0000000000009E37 // (sqrt(5)-1)/2 * 2^16
	GoldenRatio08 = 0x000000000000009E // (sqrt(5)-1)/2 * 2^8
)

// ── sqrt(2)-1 scaled to various bit widths ────────────────────────────────

// sqrt(2)-1, scaled to each bit width. A second irrational multiplier, used
// where two independent mixing constants are needed.
const (
	Sqrt2_1_64 = 0x6A09E667F3BCC909 // (sqrt(2)-1) * 2^64
	Sqrt2_1_56 = 0x006A09E667F3BCC9 // (sqrt(2)-1) * 2^56
	Sqrt2_1_48 = 0x00006A09E667F3BD // (sqrt(2)-1) * 2^48
	Sqrt2_1_40 = 0x0000006A09E667F3 // (sqrt(2)-1) * 2^40
	Sqrt2_1_32 = 0x000000006A09E667 // (sqrt(2)-1) * 2^32
	Sqrt2_1_24 = 0x00000000006A09E7 // (sqrt(2)-1) * 2^24
	Sqrt2_1_16 = 0x0000000000006A09 // (sqrt(2)-1) * 2^16
	Sqrt2_1_08 = 0x000000000000006B // (sqrt(2)-1) * 2^8
)

// ── (pi+e)/7 scaled to various bit widths ─────────────────────────────────

// (pi+e)/7, scaled to each bit width and forced odd. A third independent
// multiplier; the derived widths are shifts of Pie7_64 with the low bit set.
const (
	Pie7_64 = 0xD64DD1B3DDCB7509 // (pi+e)/7 * 2^64
	Pie7_56 = Pie7_64>>8 | 1     // (pi+e)/7 * 2^56 & make sure the number is odd
	Pie7_48 = Pie7_64>>16 | 1    // (pi+e)/7 * 2^48 & make sure the number is odd
	Pie7_40 = Pie7_64>>24 | 1    // (pi+e)/7 * 2^40 & make sure the number is odd
	Pie7_32 = Pie7_64>>32 | 1    // (pi+e)/7 * 2^32 & make sure the number is odd
	Pie7_24 = Pie7_64>>40 | 1    // (pi+e)/7 * 2^24 & make sure the number is odd
	Pie7_16 = Pie7_64>>48 | 1    // (pi+e)/7 * 2^16 & make sure the number is odd
	Pie7_08 = Pie7_64>>56 | 1    // (pi+e)/7 * 2^8 & make sure the number is odd
)

// ── Widening/spread constants ─────────────────────────────────────────────

// Spread16to64 is the best multiplier for distributing 16-bit hash values to
// groups when using SplitMix64.
// Tests show that a multiplication with Pie7_48 yields in the best distribution
// of 16-bit hashvalues to groups when using SplitMix64.
// See TestHashingCompare16BitConstantsForSplitMixGroupCountBuckets.
const Spread16to64 = 0x001001001001

// Spread32to64 is the best multiplier for distributing 32-bit hash values to
// groups when using SplitMix64. Equals GoldenRatio32.
// Tests show that a multiplication with GoldenRatio32 yields in the best
// distribution of 32-bit hashvalues to groups when using SplitMix64.
// See TestHashingCompare32BitConstantsForSplitMixGroupCountBuckets.
const Spread32to64 = GoldenRatio32
