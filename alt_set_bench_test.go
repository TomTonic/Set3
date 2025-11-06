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

package set3

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// see https://en.wikipedia.org/wiki/Xorshift#xorshift*
// This PRNG is deterministic and has a period of 2^64-1. This way we can ensure, we always get a new 'random' number, that is unknown to the set.
type prngState struct {
	state uint64
	round uint64 // for debugging purposes
}

func (thisState *prngState) Uint64() uint64 {
	x := thisState.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	thisState.state = x
	thisState.round++
	return x * 0x2545F4914F6CDD1D
}

func (thisState *prngState) Uint32() uint32 {
	x := thisState.Uint64()
	x >>= 32 // the upper 32 bit have better 'randomness', see https://en.wikipedia.org/wiki/Xorshift#xorshift*
	return uint32(x)
}

type searchDataDriver struct {
	rng       *prngState
	setValues []uint64
	hitRatio  float64
}

func newSearchDataDriver(setSize int, targetHitRatio float64, seed uint64) *searchDataDriver {
	s := prngState{state: seed}
	vals := uniqueRandomNumbers(setSize, &s)
	result := &searchDataDriver{
		rng: &s,
		// Shuffle values deterministically for test coverage
		setValues: shuffleArray(vals, &s, 3),
		hitRatio:  float64(targetHitRatio),
	}
	return result
}

// this function is designed in a way that both paths - table lookup and random number generation only - are about equaly fast/slow.
// the current implementation differs in only 1-2% execution speed for the two paths.
func (thisCfg *searchDataDriver) nextSearchValue() uint64 {
	x := uint64(float64(math.MaxUint64) * thisCfg.hitRatio)
	rndVal := thisCfg.rng.Uint64()
	upper32 := uint32(rndVal >> 32)
	idx := upper32 % uint32(len(thisCfg.setValues))
	tableVal := thisCfg.setValues[idx]
	var result uint64
	if thisCfg.rng.Uint64() < x {
		// this shall be a hit
		result = rndVal ^ tableVal ^ rndVal // use both values to make both paths equally slow/fast
	} else {
		// this shall be a miss
		result = tableVal ^ rndVal ^ tableVal // use both values to make both paths equally slow/fast
	}
	return result
}

func uniqueRandomNumbers(setSize int, rng *prngState) []uint64 {
	result := make([]uint64, setSize)
	for i := 0; i < setSize; i++ {
		result[i] = rng.Uint64()
	}
	return result
}

func shuffleArray(input []uint64, rng *prngState, rounds int) []uint64 {
	a := input // copy array
	for r := 0; r < rounds; r++ {
		for i := len(a) - 1; i > 0; i-- {
			j := rng.Uint32() % uint32(i+1)
			a[i], a[j] = a[j], a[i]
		}
	}
	return a
}

var config = []struct {
	initSetSize       int
	finalSetSize      int
	targetHitRatio    float64
	seed              uint64
	itersPerRoundFill int
	itersPerRoundFind int
	rounds            int
}{
	{initSetSize: 21, finalSetSize: 1, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 400000, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 2, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 395100, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 3, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 390260, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 4, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 385479, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 5, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 380757, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 6, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 376093, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 7, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 371486, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 8, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 366935, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 9, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 362440, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 10, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 358000, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 11, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 353615, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 12, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 349283, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 13, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 345004, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 14, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 340778, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 15, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 336603, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 16, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 332480, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 17, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 328407, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 18, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 324384, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 19, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 320410, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 20, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 316485, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 21, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 312608, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 22, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 308779, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 23, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 304996, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 24, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 301260, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 25, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 297570, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 26, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 293925, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 27, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 290324, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 28, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 286768, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 29, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 283255, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 30, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 279785, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 31, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 276358, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 32, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 272973, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 33, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 269629, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 34, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 266326, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 35, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 263064, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 36, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 259841, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 37, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 256658, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 38, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 253514, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 39, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 250408, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 40, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 247341, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 41, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 244311, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 42, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 241318, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 43, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 238362, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 44, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 235442, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 45, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 232558, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 46, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 229709, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 47, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 226895, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 48, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 224116, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 49, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 221371, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 50, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 218659, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 51, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 215980, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 52, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 213334, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 53, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 210721, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 54, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 208140, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 55, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 205590, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 56, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 203072, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 57, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 200584, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 58, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 198127, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 59, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 195700, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 60, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 193303, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 61, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 190935, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 62, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 188596, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 63, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 186286, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 64, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 184004, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 65, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 181750, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 66, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 179524, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 67, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 177325, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 68, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 175153, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 69, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 173007, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 70, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 170888, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 71, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 168795, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 72, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 166727, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 73, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 164685, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 74, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 162668, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 75, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 160675, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 76, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 158707, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 77, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 156763, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 78, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 154843, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 79, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 152946, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 80, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 151072, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 81, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 149221, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 82, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 147393, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 83, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 145587, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 84, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 143804, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 85, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 142042, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 86, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 140302, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 87, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 138583, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 88, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 136885, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 89, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 135208, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 90, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 133552, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 91, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 131916, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 92, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 130300, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 93, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 128704, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 94, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 127127, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 95, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 125570, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 96, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 124032, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 97, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 122513, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 98, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 121012, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 99, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 119530, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 100, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 118066, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 101, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 116620, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 102, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 115191, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 103, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 113780, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 104, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 112386, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 105, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 111009, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 106, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 109649, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 107, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 108306, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 108, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 106979, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 109, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 105669, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 110, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 104375, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 111, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 103096, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 112, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 101833, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 113, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 100586, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 114, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 99354, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 115, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 98137, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 116, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 96935, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 117, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 95748, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 118, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 94575, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 119, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 93416, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 120, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 92272, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 121, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 91142, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 122, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 90026, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 123, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 88923, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 124, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 87834, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 125, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 86758, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 126, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 85695, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 127, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 84645, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 128, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 83608, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 129, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 82584, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 130, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 81572, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 131, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 80573, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 132, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 79586, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 133, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 78611, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 134, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 77648, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 135, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 76697, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 136, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 75757, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 137, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 74829, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 138, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 73912, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 139, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 73007, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 140, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 72113, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 141, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 71230, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 142, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 70357, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 143, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 69495, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 144, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 68644, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 145, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 67803, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 146, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 66972, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 147, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 66152, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 148, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 65342, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 149, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 64542, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 150, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 63751, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 151, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 62970, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 152, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 62199, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 153, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 61437, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 154, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 60684, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 155, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 59941, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 156, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 59207, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 157, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 58482, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 158, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 57766, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 159, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 57058, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 160, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 56359, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 161, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 55669, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 162, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 54987, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 163, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 54313, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 164, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 53648, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 165, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 52991, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 166, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 52342, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 167, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 51701, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 168, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 51068, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 169, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 50442, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 170, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 49824, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 171, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 49214, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 172, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 48611, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 173, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 48016, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 174, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 47428, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 175, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 46847, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 176, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 46273, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 177, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 45706, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 178, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 45146, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 179, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 44593, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 180, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 44047, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 181, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 43507, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 182, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 42974, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 183, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 42448, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 184, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 41928, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 185, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 41414, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 186, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 40907, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 187, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 40406, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 188, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 39911, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 189, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 39422, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 190, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 38939, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 191, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 38462, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 192, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 37991, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 193, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 37526, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 194, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 37066, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 195, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 36612, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 196, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 36164, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 197, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 35721, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 198, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 35283, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 199, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 34851, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 200, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 34424, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 201, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 34002, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 202, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 33585, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 203, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 33174, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 204, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 32768, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 205, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 32367, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 206, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 31971, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 207, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 31579, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 208, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 31192, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 209, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 30810, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 210, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 30433, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 211, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 30060, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 212, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 29692, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 213, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 29328, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 214, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 28969, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 215, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 28614, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 216, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 28263, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 217, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 27917, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 218, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 27575, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 219, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 27237, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 220, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 26903, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 221, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 26573, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 222, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 26247, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 223, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 25925, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 224, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 25607, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 225, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 25293, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 226, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 24983, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 227, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 24677, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 228, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 24375, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 229, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 24076, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 230, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 23781, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 231, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 23490, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 232, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 23202, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 233, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 22918, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 234, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 22637, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 235, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 22360, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 236, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 22086, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 237, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 21815, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 238, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 21548, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 239, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 21284, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 240, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 21023, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 241, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 20765, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 242, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 20511, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 243, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 20260, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 244, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 20012, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 245, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 19767, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 246, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 19525, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 247, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 19286, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 248, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 19050, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 249, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 18817, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 250, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 18586, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 251, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 18358, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 252, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 18133, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 253, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 17911, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 254, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 17692, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 255, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 17475, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 256, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 17261, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 257, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 17050, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 258, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 16841, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 259, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 16635, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 260, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 16431, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 261, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 16230, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 262, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 16031, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 263, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 15835, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 264, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 15641, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 265, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 15449, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 266, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 15260, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 267, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 15073, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 268, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 14888, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 269, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 14706, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 270, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 14526, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 271, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 14348, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 272, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 14172, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 273, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13998, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 274, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13827, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 275, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13658, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 276, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13491, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 277, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13326, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 278, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13163, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 279, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 13002, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 280, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12843, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 281, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12686, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 282, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12531, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 283, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12377, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 284, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12225, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 285, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 12075, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 286, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11927, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 287, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11781, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 288, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11637, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 289, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11494, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 290, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11353, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 291, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11214, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 292, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 11077, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 293, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10941, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 294, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10807, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 295, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10675, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 296, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10544, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 297, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10415, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 298, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10287, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 299, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10161, itersPerRoundFind: 10_000_000, rounds: 11},
	{initSetSize: 21, finalSetSize: 300, targetHitRatio: 0.3, seed: 0x1234567890ABCDEF, itersPerRoundFill: 10037, itersPerRoundFind: 10_000_000, rounds: 11},
}

func TestPrngSeqLength(t *testing.T) {
	state := prngState{state: 0x1234567890ABCDEF}
	limit := uint32(30_000_000)
	set := EmptyWithCapacity[uint64](limit * 7 / 5)
	counter := uint32(0)
	for set.Size() < limit {
		set.Add(state.Uint64())
		counter++
	}
	assert.True(t, counter == limit, "sequence < limit")
}

func TestPrngDeterminism(t *testing.T) {
	state1 := prngState{state: 0x1234567890ABCDEF}
	state2 := prngState{state: 0x1234567890ABCDEF}
	limit := 30_000_000
	for i := 0; i < limit; i++ {
		v1 := state1.Uint64()
		v2 := state2.Uint64()
		assert.True(t, v1 == v2, "in sync: values not equal in round %d", i)
	}
	_ = state2.Uint64() // skip one value to get both prng out of sync
	for i := 0; i < limit; i++ {
		v1 := state1.Uint64()
		v2 := state2.Uint64()
		assert.False(t, v1 == v2, "out of sync: values equal in round %d", i)
	}
	_ = state1.Uint64() // get both prng back in sync
	for i := 0; i < limit; i++ {
		v1 := state1.Uint64()
		v2 := state2.Uint64()
		assert.True(t, v1 == v2, "back in sync: values not equal in round %d", i)
	}
}

func TestUniqueRandomNumbersDeterministic(t *testing.T) {
	s1 := prngState{state: 0x1234567890ABCDEF}
	s2 := prngState{state: 0x1234567890ABCDEF}

	urn1 := uniqueRandomNumbers(5000, &s1)
	urn2 := uniqueRandomNumbers(5000, &s2)
	assert.True(t, slicesEqual(urn1, urn2), "slices not equal")

}

func TestSearchDataDriver(t *testing.T) {
	setSize := 50_000
	targetHitRatio := 0.3
	seed := uint64(0x1234567890ABCDEF)

	sdd1 := newSearchDataDriver(setSize, targetHitRatio, seed)
	sdd2 := newSearchDataDriver(setSize, targetHitRatio, seed)
	assert.True(t, slicesEqual(sdd1.setValues, sdd2.setValues), "slices not equal")

	set := FromArray(sdd1.setValues)

	rounds := 5_000_000
	hits := 0
	for i := 0; i < rounds; i++ {
		v1 := sdd1.nextSearchValue()
		v2 := sdd2.nextSearchValue()
		assert.True(t, v1 == v2, "values not equal in round %d", i)
		if set.Contains(v1) {
			hits++
		}
	}
	actualHitRatio := float64(hits) / float64(rounds)
	lowerBound := targetHitRatio * 0.99
	upperBound := targetHitRatio * 1.01
	assert.True(t, actualHitRatio > lowerBound && actualHitRatio < upperBound, "actual hit ratio (%d) missed target hit ratio by more than 1 percent", actualHitRatio)
}

func slicesEqual(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

var sddConfig = []struct {
	setSize int
	seed    uint64
}{
	{setSize: 1, seed: 0x1234567890ABCDEF},
	{setSize: 10, seed: 0x1234567890ABCDEF},
	{setSize: 10_000, seed: 0x1234567890ABCDEF},
	{setSize: 10_000_000, seed: 0x1234567890ABCDEF},
}

func BenchmarkSearchDataDriver(b *testing.B) {
	// b.Skip("unskip to benchmark nextSearchValue paths")
	for _, cfg := range sddConfig {
		sdd := newSearchDataDriver(cfg.setSize, 0.0, cfg.seed)
		// Force garbage collection
		runtime.GC()
		// Give the garbage collector some time to complete
		time.Sleep(2 * time.Second)
		var x uint64
		b.Run(fmt.Sprintf("setSize(%d);hit(%.1f)", len(sdd.setValues), sdd.hitRatio), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				x ^= sdd.nextSearchValue()
			}
		})
		sdd.hitRatio = 1.0
		b.Run(fmt.Sprintf("setSize(%d);hit(%.1f)", len(sdd.setValues), sdd.hitRatio), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				x ^= sdd.nextSearchValue()
			}
		})
	}
}

func TestSet3Fill(t *testing.T) {
	t.Skip("unskip for benchmark tests - runs 3-5 minutes")
	fmt.Printf("Implementation;Benchmark;Final Size;Hit Rate;Nanoseconds per Sample;Required Bytes per Element\n")
	for _, cfg := range config {
		timePerIter := make([]float64, cfg.rounds)
		memPerIter := make([]float64, cfg.rounds)
		for i := 0; i < cfg.rounds; i++ {
			sdd := newSearchDataDriver(cfg.finalSetSize, cfg.targetHitRatio, cfg.seed+uint64(i*53))
			sets := make([]*Set3[uint64], cfg.itersPerRoundFill)
			var startMem, endMem runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&startMem)
			startTime := time.Now().UnixNano()
			for j := 0; j < cfg.itersPerRoundFill; j++ {
				set := EmptyWithCapacity[uint64](uint32(cfg.initSetSize))
				for k := 0; k < len(sdd.setValues); k++ {
					set.Add(sdd.setValues[k])
				}
				sets[j] = set // keep it to determine memory consumtion
			}
			endTime := time.Now().UnixNano()
			runtime.GC()
			runtime.ReadMemStats(&endMem)
			// make sure everything is there as expected
			for j := 0; j < cfg.itersPerRoundFill; j++ {
				if sets[j].Size() != uint32(cfg.finalSetSize) {
					t.Fail()
				}
			}

			timePerIter[i] = float64(endTime-startTime) / float64(cfg.itersPerRoundFill)
			memPerIter[i] = float64((endMem.HeapAlloc+endMem.StackInuse+endMem.StackSys)-(startMem.HeapAlloc+startMem.StackInuse+startMem.StackSys)) / float64(cfg.itersPerRoundFill) / float64(cfg.finalSetSize)
		}
		medTime := median(timePerIter)
		medMem := median(memPerIter)
		fmt.Printf("Set3;Fill={BenchMem{BenchTime{EmptyWithCapacity[uint64](%d) + %d*Add(uint64)}}};%d;%.3f;%.3f;%.3f\n", cfg.initSetSize, cfg.finalSetSize, cfg.finalSetSize, 0.0, medTime, medMem)
	}
}

func TestNativeMapFill(t *testing.T) {
	t.Skip("unskip for benchmark tests - runs 3-5 minutes")
	fmt.Printf("Implementation;Benchmark;Final Size;Hit Rate;Nanoseconds per Sample;Required Bytes per Element\n")
	for _, cfg := range config {
		timePerIter := make([]float64, cfg.rounds)
		memPerIter := make([]float64, cfg.rounds)
		for i := 0; i < cfg.rounds; i++ {
			sdd := newSearchDataDriver(cfg.finalSetSize, cfg.targetHitRatio, cfg.seed+uint64(i*53))
			sets := make([]*nativeMap[uint64], cfg.itersPerRoundFill)
			var startMem, endMem runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&startMem)
			startTime := time.Now().UnixNano()
			for j := 0; j < cfg.itersPerRoundFill; j++ {
				set := emptyNativeWithCapacity[uint64](uint32(cfg.initSetSize))
				for k := 0; k < len(sdd.setValues); k++ {
					set.add(sdd.setValues[k])
				}
				sets[j] = set // keep it to determine memory consumtion
			}
			endTime := time.Now().UnixNano()
			runtime.GC()
			runtime.ReadMemStats(&endMem)
			// make sure everything is there as expected
			for j := 0; j < cfg.itersPerRoundFill; j++ {
				if sets[j].count() != uint32(cfg.finalSetSize) {
					t.Fail()
				}
			}

			timePerIter[i] = float64(endTime-startTime) / float64(cfg.itersPerRoundFill)
			memPerIter[i] = float64((endMem.HeapAlloc+endMem.StackInuse+endMem.StackSys)-(startMem.HeapAlloc+startMem.StackInuse+startMem.StackSys)) / float64(cfg.itersPerRoundFill) / float64(cfg.finalSetSize)
		}
		medTime := median(timePerIter)
		medMem := median(memPerIter)
		fmt.Printf("nativeMap;Fill={BenchMem{BenchTime{emptyNativeWithCapacity[uint64](%d) + %d*add(uint64)}}};%d;%.3f;%.3f;%.3f\n", cfg.initSetSize, cfg.finalSetSize, cfg.finalSetSize, 0.0, medTime, medMem)
	}
}

func BenchmarkSet3FindVariance(b *testing.B) {
	for _, cfg := range config {
		for seedUp := 0; seedUp < 10; seedUp++ {
			for round := 0; round < 10; round++ {
				sdd := newSearchDataDriver(cfg.finalSetSize, cfg.targetHitRatio, cfg.seed+uint64(seedUp*51))
				resultSet := FromArray(sdd.setValues)
				// Force garbage collection
				runtime.GC()
				// Give the garbage collector some time to complete
				time.Sleep(1 * time.Second)
				var hit uint64
				var total uint64
				b.Run(fmt.Sprintf("init(%d);final(%d);hit(%f)-s(%d)", len(sdd.setValues), cfg.finalSetSize, cfg.targetHitRatio, seedUp), func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						search := sdd.nextSearchValue()
						if resultSet.Contains(search) {
							hit++
						}
						total++
					}
				})
				b.Logf("Actual hit ratio: %.3f", float32(hit)/float32(total))
			}
		}
	}
}

func median(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)
	sort.Float64s(dataCopy)

	l := len(dataCopy)
	if l%2 == 0 {
		return (dataCopy[l/2-1] + dataCopy[l/2]) / 2
	}
	return dataCopy[l/2]
}

func TestSet3Find(t *testing.T) {
	t.Skip("unskip for benchmark tests - runs 3-5 minutes")
	fmt.Printf("Implementation;Benchmark;Final Size;Hit Rate;Nanoseconds per Sample;Required Bytes per Element\n")
	for _, cfg := range config {
		timePerIter := make([]float64, cfg.rounds)
		memPerRound := make([]float64, cfg.rounds)
		var hit uint64
		var total uint64
		for i := 0; i < cfg.rounds; i++ {
			currentSdd := newSearchDataDriver(cfg.finalSetSize, cfg.targetHitRatio, cfg.seed+uint64(i*53))
			testdata := make([]uint64, cfg.itersPerRoundFind)
			for j := range cfg.itersPerRoundFind {
				testdata[j] = currentSdd.nextSearchValue()
			}
			var startMem, endMem runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&startMem)
			currentSet := FromArray(currentSdd.setValues)
			runtime.GC()
			runtime.ReadMemStats(&endMem)
			memPerRound[i] = float64(endMem.TotalAlloc-startMem.TotalAlloc) / float64(cfg.finalSetSize)

			startTime := time.Now().UnixNano()
			for j := 0; j < cfg.itersPerRoundFind; j++ {
				// search := currentSdd.nextSearchValue()
				search := testdata[j]
				if currentSet.Contains(search) {
					hit++
				}
				total++
			}
			endTime := time.Now().UnixNano()
			timePerIter[i] = float64(endTime-startTime) / float64(cfg.itersPerRoundFind)
		}
		hitRea := float32(hit) / float32(total)
		medTime := median(timePerIter)
		medMem := median(memPerRound)
		fmt.Printf("Set3;Find={BenchMem{FromArray([%d]uint64)} + BenchTime{Contains(uint64)}};%d;%.3f;%.3f;%.3f\n", cfg.finalSetSize, cfg.finalSetSize, hitRea, medTime, medMem)
	}
}

func TestNativeMapFind(t *testing.T) {
	t.Skip("unskip for benchmark tests - runs 3-5 minutes")
	fmt.Printf("Implementation;Benchmark;Final Size;Hit Rate;Nanoseconds per Sample;Required Bytes per Element\n")
	for _, cfg := range config {
		timePerIter := make([]float64, cfg.rounds)
		memPerRound := make([]float64, cfg.rounds)
		var hit uint64
		var total uint64
		for i := 0; i < cfg.rounds; i++ {
			currentSdd := newSearchDataDriver(cfg.finalSetSize, cfg.targetHitRatio, cfg.seed+uint64(i*53))
			testdata := make([]uint64, cfg.itersPerRoundFind)
			for j := range cfg.itersPerRoundFind {
				testdata[j] = currentSdd.nextSearchValue()
			}
			var startMem, endMem runtime.MemStats
			runtime.GC()
			runtime.ReadMemStats(&startMem)
			currentSet := emptyNativeWithCapacity[uint64](uint32(cfg.finalSetSize))
			for j := 0; j < len(currentSdd.setValues); j++ {
				currentSet.add(currentSdd.setValues[j])
			}
			runtime.GC()
			runtime.ReadMemStats(&endMem)
			memPerRound[i] = float64(endMem.TotalAlloc-startMem.TotalAlloc) / float64(cfg.finalSetSize)

			startTime := time.Now().UnixNano()
			for j := 0; j < cfg.itersPerRoundFind; j++ {
				// search := currentSdd.nextSearchValue()
				search := testdata[j]
				if currentSet.contains(search) {
					hit++
				}
				total++
			}
			endTime := time.Now().UnixNano()
			timePerIter[i] = float64(endTime-startTime) / float64(cfg.itersPerRoundFind)
		}
		hitRea := float32(hit) / float32(total)
		medTime := median(timePerIter)
		medMem := median(memPerRound)
		fmt.Printf("nativeMap;Find={BenchMem{emptyNativeWithCapacity[uint64](%d) + %d*add(uint64)} + BenchTime{contains(uint64)}};%d;%.3f;%.3f;%.3f\n", cfg.finalSetSize, cfg.finalSetSize, cfg.finalSetSize, hitRea, medTime, medMem)
	}
}
