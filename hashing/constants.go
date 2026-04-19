package hashing

// ── Golden Ratio: (sqrt(5)-1)/2 scaled to various bit widths ──────────────

const GoldenRatio64 = 0x9E3779B97F4A7C15 // (sqrt(5)-1)/2 * 2^64
const GoldenRatio56 = 0x009E3779B97F4A7D // (sqrt(5)-1)/2 * 2^56
const GoldenRatio48 = 0x00009E3779B97F4B // (sqrt(5)-1)/2 * 2^48
const GoldenRatio40 = 0x0000009E3779B97F // (sqrt(5)-1)/2 * 2^40
const GoldenRatio32 = 0x000000009E3779B9 // (sqrt(5)-1)/2 * 2^32
const GoldenRatio24 = 0x00000000009E3779 // (sqrt(5)-1)/2 * 2^24
const GoldenRatio16 = 0x0000000000009E37 // (sqrt(5)-1)/2 * 2^16
const GoldenRatio08 = 0x000000000000009E // (sqrt(5)-1)/2 * 2^8

// ── sqrt(2)-1 scaled to various bit widths ────────────────────────────────

const Sqrt2_1_64 = 0x6A09E667F3BCC909 // (sqrt(2)-1) * 2^64
const Sqrt2_1_56 = 0x006A09E667F3BCC9 // (sqrt(2)-1) * 2^56
const Sqrt2_1_48 = 0x00006A09E667F3BD // (sqrt(2)-1) * 2^48
const Sqrt2_1_40 = 0x0000006A09E667F3 // (sqrt(2)-1) * 2^40
const Sqrt2_1_32 = 0x000000006A09E667 // (sqrt(2)-1) * 2^32
const Sqrt2_1_24 = 0x00000000006A09E7 // (sqrt(2)-1) * 2^24
const Sqrt2_1_16 = 0x0000000000006A09 // (sqrt(2)-1) * 2^16
const Sqrt2_1_08 = 0x000000000000006B // (sqrt(2)-1) * 2^8

// ── (pi+e)/7 scaled to various bit widths ─────────────────────────────────

const Pie7_64 = 0xD64DD1B3DDCB7509 // (pi+e)/7 * 2^64
const Pie7_56 = Pie7_64>>8 | 1     // (pi+e)/7 * 2^56 & make sure the number is odd
const Pie7_48 = Pie7_64>>16 | 1    // (pi+e)/7 * 2^48 & make sure the number is odd
const Pie7_40 = Pie7_64>>24 | 1    // (pi+e)/7 * 2^40 & make sure the number is odd
const Pie7_32 = Pie7_64>>32 | 1    // (pi+e)/7 * 2^32 & make sure the number is odd
const Pie7_24 = Pie7_64>>40 | 1    // (pi+e)/7 * 2^24 & make sure the number is odd
const Pie7_16 = Pie7_64>>48 | 1    // (pi+e)/7 * 2^16 & make sure the number is odd
const Pie7_08 = Pie7_64>>56 | 1    // (pi+e)/7 * 2^8 & make sure the number is odd

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
