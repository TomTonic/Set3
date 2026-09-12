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
	"math"
	"testing"
)

// TestLabLoadConstantMatchesTheLibrary guards the one constant this package
// copies from the library.
//
// set3maxAvgGroupLoad is unexported, so the load-curve pass keeps its own copy
// to invert — and that copy silently went stale twice, reading 6.67 while the
// library moved to 6.5 and then to 4.8. The consequence was not a wrong number
// but a wrong range: the pass asked for occupancies from 0.80 down to 0.25 and
// produced 0.58 down to 0.18, so the part of the curve that compares Set3 at
// the native map's occupancy was simply not measured any more, and nothing
// failed.
//
// This measures what the library actually does instead of reading its source:
// a set built for n elements and filled with n of them sits at exactly the
// average group load over the eight slots of a group.
func TestLabLoadConstantMatchesTheLibrary(t *testing.T) {
	measured := LabLoadConstantCheck()
	// The tolerance covers the prime rounding of the group count, which moves
	// the achieved occupancy by a fraction of a percent, not the constant.
	const tolerance = 0.15
	if math.Abs(measured-set3MaxAvgGroupLoad) > tolerance {
		t.Errorf("the library fills to %.3f elements per group but this package assumes %.3f; "+
			"update set3MaxAvgGroupLoad, or every load-curve point is computed from the wrong inverse",
			measured, set3MaxAvgGroupLoad)
	}
	t.Logf("library fills to %.3f elements per group, lab constant is %.3f", measured, set3MaxAvgGroupLoad)
}

// TestLoadCurveTargetsAreActuallyReachable checks the property that silently
// stopped holding: that asking for an occupancy produces it.
//
// The curve records the load it achieved rather than the one it asked for, so a
// target the library will not hold does not produce a wrong number — it
// produces a right number about a different operating point, filed under the
// wrong label, with the top of the curve quietly collapsing onto the default.
// That is what happened when the top target was 0.80 and the library would not
// go above 0.60.
func TestLoadCurveTargetsAreActuallyReachable(t *testing.T) {
	const size = 1 << 14
	keys := buildKeys(makeUint64Key, memberDomain, size)

	for _, target := range loadCurveTargets {
		if target > set3MaxReachableLoad {
			t.Errorf("target %.3f is above the %.3f Set3 will hold; it can be asked for but not measured",
				target, set3MaxReachableLoad)
			continue
		}
		capacity := set3CapacityForSlots(float64(size) / target)
		got := measureSet3Load(keys, size, capacity)
		// A fifteenth covers the group count rounding up to a prime, which
		// moves the achieved occupancy by a per cent or two at this size.
		if rel := math.Abs(got-target) / target; rel > 1.0/15.0 {
			t.Errorf("asked for occupancy %.3f and got %.3f (%.1f%% off); the curve would record the right "+
				"number under the wrong label", target, got, rel*100)
		}
		t.Logf("target %.3f -> achieved %.3f, capacity %d", target, got, capacity)
	}
}
