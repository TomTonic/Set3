// Copyright 2023 Dolthub, Inc.
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

package Set3

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func FuzzStringSet(f *testing.F) {
	f.Add(uint8(1), 14, 50)
	f.Add(uint8(2), 1, 1)
	f.Add(uint8(2), 14, 14)
	f.Add(uint8(2), 14, 15)
	f.Add(uint8(2), 25, 100)
	f.Add(uint8(2), 25, 1000)
	f.Add(uint8(8), 0, 1)
	f.Add(uint8(8), 1, 1)
	f.Add(uint8(8), 14, 14)
	f.Add(uint8(8), 14, 15)
	f.Add(uint8(8), 25, 100)
	f.Add(uint8(8), 25, 1000)
	f.Fuzz(func(t *testing.T, keySz uint8, init, count int) {
		// smaller key sizes generate more overwrites
		fuzzTestStringSet(t, uint32(keySz), uint32(init), uint32(count))
	})
}

func fuzzTestStringSet(t *testing.T, keySz, init, count uint32) {
	const limit = 1024 * 1024
	if count > limit || init > limit {
		t.Skip()
	}
	m := NewSet3WithSize[string](init)
	if count == 0 {
		return
	}
	// make tests deterministic
	setConstSeedSet(m, 1)

	keys := genStringData(int(keySz), int(count))
	golden := make(map[string]int, init)
	for i, k := range keys {
		m.Add(k)
		golden[k] = i
	}
	assert.Equal(t, uint32(len(golden)), m.Count())

	for k := range golden {
		ok := m.Contains(k)
		assert.True(t, ok)
	}
	for _, k := range keys {
		_, ok := golden[k]
		assert.True(t, ok)
		assert.True(t, m.Contains(k))
	}

	deletes := keys[:count/2]
	for _, k := range deletes {
		delete(golden, k)
		m.Remove(k)
	}
	assert.Equal(t, uint32(len(golden)), m.Count())

	for _, k := range deletes {
		assert.False(t, m.Contains(k))
	}
	for k := range golden {
		ok := m.Contains(k)
		assert.True(t, ok)
	}
}

type hasher struct {
	hash func()
	seed uintptr
}

func setConstSeedSet[K comparable](set *Set3[K], seed uintptr) {
	h := (*hasher)((unsafe.Pointer)(&set.hashFunction))
	h.seed = seed
}
