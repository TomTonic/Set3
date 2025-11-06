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
	"math/rand"
	"os"
	"runtime/pprof"
	"testing"

	"github.com/stretchr/testify/assert"
)

var randOpsConfig = []struct {
	numSets int
	setSize int
	setVar  int
	mod     int
}{
	{numSets: 7, setSize: 20, setVar: 50, mod: 140},
	{numSets: 21, setSize: 5, setVar: 5, mod: 128},
	{numSets: 7, setSize: 1000, setVar: 50, mod: 1200},
	{numSets: 21, setSize: 5, setVar: 5, mod: 1280},
	{numSets: 70, setSize: 20, setVar: 50, mod: 140},
	{numSets: 210, setSize: 3000, setVar: 3000, mod: 9999},
}

func TestRandomOps(t *testing.T) {
	t.Skip("unskip for cpu profiling - runs 0.5-1 minutes")
	// Start profiling
	f, err := os.Create("cpu.prof")
	if err != nil {
		fmt.Println("Could not create CPU profile:", err)
		return
	}
	defer func() {
		if cerr := f.Close(); cerr != nil {
			t.Fatalf("Could not close CPU profile: %v", cerr)
		}
	}()
	if err := pprof.StartCPUProfile(f); err != nil {
		fmt.Println("Could not start CPU profile:", err)
		return
	}
	defer pprof.StopCPUProfile()

	for _, cfg := range randOpsConfig {
		for iterations := 0; iterations < 200; iterations++ {
			sets := make([]*Set3[uint32], cfg.numSets)
			fillAllSets(cfg.numSets, cfg.setSize, cfg.setVar, cfg.mod, sets)
			deleteSomeElementsAndAddSomeNewOnes(cfg.numSets, cfg.mod, sets)
			deleteSomeSetsCompletely(cfg.numSets, sets)
			refillAllSets(cfg.numSets, cfg.setSize, cfg.setVar, cfg.mod, sets)

			// add all elements to a superset
			superset := Empty[uint32]()
			for i := 0; i < cfg.numSets; i++ {
				superset.AddAll(sets[i])
			}

			supersetTests(cfg.numSets, cfg.mod, superset, sets, t)
		}
	}
}

func fillAllSets(numSets, setSize, setVar, mod int, sets []*Set3[uint32]) {
	for i := 0; i < numSets; i++ {
		sets[i] = Empty[uint32]()
		targetSize := setSize + rand.Intn(setVar)
		for j := 0; j < targetSize; j++ {
			sets[i].Add(rand.Uint32() % uint32(mod))
		}
	}
}

func deleteSomeElementsAndAddSomeNewOnes(numSets, mod int, sets []*Set3[uint32]) {
	for i := 0; i < numSets/3; i++ {
		idx := rand.Intn(numSets)
		sets[idx].AddAllFromArray([]uint32{rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod),
			rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod)})
		for i := 0; i < mod*2; i++ {
			sets[idx].Remove(rand.Uint32() % uint32(mod))
		}
		sets[idx].AddAllOf(rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod),
			rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod))
	}
}

func deleteSomeSetsCompletely(numSets int, sets []*Set3[uint32]) {
	for i := 0; i < numSets/4; i++ {
		idx := rand.Intn(numSets)
		clone := sets[idx].Clone()
		sets[idx].Clear()
		sets[idx].AddAll(clone)
	}
}

func refillAllSets(numSets, setSize, setVar, mod int, sets []*Set3[uint32]) {
	for i := 0; i < numSets; i++ {
		targetSize := uint32(setSize + rand.Intn(setVar))
		for j := sets[i].Size(); j < targetSize; j++ {
			sets[i].Add(rand.Uint32() % uint32(mod))
		}
	}
}

func supersetTests(numSets, mod int, superset *Set3[uint32], sets []*Set3[uint32], t *testing.T) {
	for i := 0; i < numSets; i++ {
		intersect := superset.Intersect(sets[i])
		// intentionally call Contains* to exercise code paths but do not keep the results
		_ = sets[i].ContainsAllOf(rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod))
		_ = sets[i].ContainsAnyOf(rand.Uint32()%uint32(mod), rand.Uint32()%uint32(mod))
		equal := intersect.Equals(sets[i])
		stringIntersect := intersect.String()
		stringSet := sets[i].String()
		assert.True(t, equal, stringIntersect+" != "+stringSet)
		newSet := FromArray([]uint32{rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod),
			rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod), rand.Uint32() % uint32(mod)})
		stringNewSet := newSet.String()
		unequal := intersect.Equals(newSet)
		assert.False(t, unequal, stringIntersect+" == "+stringNewSet)
	}
}
