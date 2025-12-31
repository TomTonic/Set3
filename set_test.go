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
	"math/rand"
	"regexp"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/stretchr/testify/assert"
)

func TestSet3VariousSizes(t *testing.T) {
	t.Run("strings=0", func(t *testing.T) {
		testSet3(t, genStringData(16, 0))
	})
	t.Run("strings=100", func(t *testing.T) {
		testSet3(t, genStringData(16, 101))
	})
	t.Run("strings=1000", func(t *testing.T) {
		testSet3(t, genStringData(16, 1000))
	})
	t.Run("strings=10_000", func(t *testing.T) {
		testSet3(t, genStringData(16, 10_000))
	})
	t.Run("strings=100_000", func(t *testing.T) {
		testSet3(t, genStringData(16, 100_000))
	})
	t.Run("uint32=0", func(t *testing.T) {
		testSet3(t, genUint32Data(0))
	})
	t.Run("uint32=100", func(t *testing.T) {
		testSet3(t, genUint32Data(100))
	})
	t.Run("uint32=1000", func(t *testing.T) {
		testSet3(t, genUint32Data(1000))
	})
	t.Run("uint32=10_000", func(t *testing.T) {
		testSet3(t, genUint32Data(10_000))
	})
	t.Run("uint32=100_000", func(t *testing.T) {
		testSet3(t, genUint32Data(100_000))
	})
}

func testSet3[K comparable](t *testing.T, keys []K) {
	// sanity check
	require.Equal(t, len(keys), len(uniq(keys)), keys)
	t.Run("put", func(t *testing.T) {
		testSetPut(t, keys)
	})
	t.Run("has", func(t *testing.T) {
		testSetHas(t, keys)
	})
	t.Run("delete", func(t *testing.T) {
		testSetDelete(t, keys)
	})
	t.Run("clear", func(t *testing.T) {
		testSetClear(t, keys)
	})
	t.Run("iter", func(t *testing.T) {
		testSetIter(t, keys)
	})
	t.Run("grow", func(t *testing.T) {
		testSetGrow(t, keys)
	})
}

func uniq[K comparable](keys []K) []K {
	s := make(map[K]struct{}, len(keys))
	for _, k := range keys {
		s[k] = struct{}{}
	}
	u := make([]K, 0, len(keys))
	for k := range s {
		u = append(u, k)
	}
	return u
}

func genStringData(size, count int) (keys []string) {
	src := rand.New(rand.NewSource(int64(size * count)))
	letters := []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
	r := make([]rune, size*count)
	for i := range r {
		r[i] = letters[src.Intn(len(letters))]
	}
	keys = make([]string, count)
	for i := range keys {
		keys[i] = string(r[:size])
		r = r[size:]
	}
	return
}

func genUint32Data(count int) (keys []uint32) {
	keys = make([]uint32, count)
	var x uint32
	for i := range keys {
		x += (rand.Uint32() % 128) + 1
		keys[i] = x
	}
	return
}

func testSetPut[K comparable](t *testing.T, keys []K) {
	m := EmptyWithCapacity[K](uint32(len(keys)))
	assert.Equal(t, uint32(0), m.Size())
	for _, key := range keys {
		m.Add(key)
	}
	assert.Equal(t, uint32(len(keys)), m.Size())
	// overwrite
	for _, key := range keys {
		m.Add(key)
	}
	assert.Equal(t, uint32(len(keys)), m.Size())
	for _, key := range keys {
		ok := m.Contains(key)
		assert.True(t, ok)
	}
	assert.Equal(t, len(keys), int(m.resident))
}

func testSetHas[K comparable](t *testing.T, keys []K) {
	m := EmptyWithCapacity[K](uint32(len(keys)))
	for _, key := range keys {
		m.Add(key)
	}
	for _, key := range keys {
		ok := m.Contains(key)
		assert.True(t, ok)
	}
}

func testSetDelete[K comparable](t *testing.T, keys []K) {
	m := EmptyWithCapacity[K](uint32(len(keys)))
	assert.Equal(t, uint32(0), m.Size())
	for _, key := range keys {
		m.Add(key)
	}
	assert.Equal(t, uint32(len(keys)), m.Size())
	for _, key := range keys {
		m.Remove(key)
		ok := m.Contains(key)
		assert.False(t, ok)
	}
	assert.Equal(t, uint32(0), m.Size())
	// put keys back after deleting them
	for _, key := range keys {
		m.Add(key)
	}
	assert.Equal(t, uint32(len(keys)), m.Size())
}

func testSetClear[K comparable](t *testing.T, keys []K) {
	m := EmptyWithCapacity[K](0)
	assert.Equal(t, uint32(0), m.Size())
	for _, key := range keys {
		m.Add(key)
	}
	assert.Equal(t, uint32(len(keys)), m.Size())
	m.Clear()
	assert.Equal(t, uint32(0), m.Size())
	for _, key := range keys {
		ok := m.Contains(key)
		assert.False(t, ok)
	}
	var calls int
	for range m.ImmutableRange() {
		calls++
	}
	assert.Equal(t, 0, calls)

	// Assert that the Set was actually cleared...
	/*
		var k K
		for _, s := range m.groupSlot {
			for _, v := range s {
				assert.Equal(t, k, v)
			}
		}
	*/
}

func testSetIter[K comparable](t *testing.T, keys []K) {
	m := EmptyWithCapacity[K](uint32(len(keys)))
	for _, key := range keys {
		m.Add(key)
	}
	visited := make(map[K]uint, len(keys))
	for _, k := range keys {
		visited[k] = 0
	}
	for e := range m.ImmutableRange() {
		visited[e]++
	}
	for _, c := range visited {
		assert.Equal(t, c, uint(1))
	}
}

func testSetGrow[K comparable](t *testing.T, keys []K) {
	n := uint32(len(keys))
	m := EmptyWithCapacity[K](n / 10)
	for _, key := range keys {
		m.Add(key)
	}
	for _, key := range keys {
		ok := m.Contains(key)
		assert.True(t, ok)
	}
}

func TestSet3MutableRange(t *testing.T) {
	tests := []struct {
		name     string
		set      Set3[int]
		expected []int
	}{
		{
			name: "Empty set",
			set: Set3[int]{
				groupCtrl: []uint64{set3AllEmpty},
				groupSlot: [][8]int{{0, 0, 0, 0, 0, 0, 0, 0}},
			},
			expected: []int{},
		},
		{
			name: "Single group with elements",
			set: Set3[int]{
				groupCtrl: []uint64{0x8001800180018001},
				groupSlot: [][8]int{{1, 0, 3, 0, 5, 0, 7, 0}},
			},
			expected: []int{1, 3, 5, 7},
		},
		{
			name: "Multiple groups with elements",
			set: Set3[int]{
				groupCtrl: []uint64{
					0x8001800180018001,
					0x0180018001800180,
					set3AllEmpty,
				},
				groupSlot: [][8]int{
					{1, 2, 3, 4, 5, 6, 7, 8},
					{9, 10, 11, 12, 13, 14, 15, 16},
					{9, 10, 11, 12, 13, 14, 15, 16},
				},
			},
			expected: []int{1, 3, 5, 7, 10, 12, 14, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result []int
			for e := range tt.set.ImmutableRange() {
				result = append(result, e)
			}
			if len(result) != len(tt.expected) {
				t.Errorf("expected length %d, got %d", len(tt.expected), len(result))
			}
			for i, v := range result {
				if v != tt.expected[i] {
					t.Errorf("expected %v at index %d, got %v", tt.expected[i], i, v)
				}
			}
		})
	}
}

func TestSet3MutableRangeTwice(t *testing.T) {
	set := FromArray[string]([]string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"})
	strary := make([]string, set.Size())

	idx := 0
	for s := range set.MutableRange() {
		strary[idx] = s
		idx++
	}

	idx = 0
	for s := range set.MutableRange() {
		// the order is arbitrary, but the same if set is not altered
		assert.True(t, strary[idx] == s, strary[idx]+"!="+s)
		idx++
	}

}

func TestSet3ImmutableRange(t *testing.T) {
	tests := []struct {
		name     string
		set      Set3[int]
		expected []int
	}{
		{
			name: "Empty set",
			set: Set3[int]{
				groupCtrl: []uint64{set3AllEmpty},
				groupSlot: [][8]int{{0, 0, 0, 0, 0, 0, 0, 0}},
			},
			expected: []int{},
		},
		{
			name: "Single group with elements",
			set: Set3[int]{
				groupCtrl: []uint64{0x8001800180018001},
				groupSlot: [][8]int{{1, 0, 3, 0, 5, 0, 7, 0}},
			},
			expected: []int{1, 3, 5, 7},
		},
		{
			name: "Multiple groups with elements",
			set: Set3[int]{
				groupCtrl: []uint64{
					0x8001800180018001,
					0x0180018001800180,
					set3AllEmpty,
				},
				groupSlot: [][8]int{
					{1, 2, 3, 4, 5, 6, 7, 8},
					{9, 10, 11, 12, 13, 14, 15, 16},
					{9, 10, 11, 12, 13, 14, 15, 16},
				},
			},
			expected: []int{1, 3, 5, 7, 10, 12, 14, 16},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result []int
			i := 0
			for e := range tt.set.ImmutableRange() {
				tt.set.groupCtrl[0] = set3AllDeleted
				tt.set.groupSlot[0][i] = i * 20
				result = append(result, e)
				i++
			}
			if len(result) != len(tt.expected) {
				t.Errorf("expected length %d, got %d", len(tt.expected), len(result))
			}
			for i, v := range result {
				if v != tt.expected[i] {
					t.Errorf("expected %v at index %d, got %v", tt.expected[i], i, v)
				}
			}
		})
	}
}

func TestSet3Equals(t *testing.T) {
	set1 := EmptyWithCapacity[int](10)
	sameptr := set1
	if !set1.Equals(sameptr) {
		t.Errorf("Test case 1: Both sets are the same instance: Expected true, got false")
	}

	set2 := EmptyWithCapacity[int](10)
	if !set1.Equals(set2) {
		t.Errorf("Test case 2: Both sets are empty but different instances: Expected true, got false")
	}

	set3 := EmptyWithCapacity[int](10)
	set3.Add(1)
	if set1.Equals(set3) {
		t.Errorf("Test case 3: Sets with different countsExpected false, got true")
	}

	set4 := EmptyWithCapacity[int](10)
	set4.Add(1)
	if !set3.Equals(set4) {
		t.Errorf("Test case 4: Sets with same elements: Expected true, got false")
	}

	set5 := EmptyWithCapacity[int](20)
	set5.Add(1)
	if !set3.Equals(set4) {
		t.Errorf("Test case 5: Sets with same elements but different capacities: Expected true, got false")
	}

	set6 := EmptyWithCapacity[int](10)
	set6.Add(2)
	if set3.Equals(set6) {
		t.Errorf("Test case 6: Sets with different elements: Expected false, got true")
	}
}

func TestSet3FromArray(t *testing.T) {
	empty := Empty[int]()
	s1 := FromArray([]int{})
	eq := empty.Equals(s1)
	assert.Equal(t, eq, true)
	s1.Add(1)
	s2 := FromArray([]int{1})
	eq = s1.Equals(s2)
	assert.Equal(t, eq, true)
	s1.Add(2)
	s3 := FromArray([]int{2, 1})
	eq = s1.Equals(s3)
	assert.Equal(t, eq, true)
}

func TestSet3From(t *testing.T) {
	empty := Empty[int]()
	s1 := From[int]()
	eq := empty.Equals(s1)
	assert.Equal(t, eq, true)
	s1.Add(1)
	s2 := From(1)
	eq = s1.Equals(s2)
	assert.Equal(t, eq, true)
	s1.Add(2)
	s3 := From(2, 1)
	eq = s1.Equals(s3)
	assert.Equal(t, eq, true)
}

func TestSet3String(t *testing.T) {
	tests := []struct {
		name string
		set  Set3[int]
		want string
	}{
		{
			name: "Empty set",
			set:  *Empty[int](),
			want: "^\\{\\}$",
		},
		{
			name: "Single element",
			set:  *FromArray([]int{1}),
			want: "^\\{1\\}$",
		},
		{
			name: "Multiple elements",
			set:  *FromArray([]int{1, 2, 3}),
			want: "^\\{[1-3],[1-3],[1-3]\\}$",
		},
		{
			name: "Multiple groups",
			set:  *FromArray([]int{1, 2, 3, 4, 5, 6, 7, 8, 9}),
			want: "^\\{[1-9],[1-9],[1-9],[1-9],[1-9],[1-9],[1-9],[1-9],[1-9]\\}$",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.set.String()
			pattern := tt.want
			re, err := regexp.Compile(pattern)
			if err != nil {
				t.Errorf("Error compiling regex %v: %v", pattern, err)
			}
			match := re.MatchString(got)
			if !match {
				t.Errorf("Set3.String() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSet3Clone(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set2 := set1.Clone()
	assert.False(t, set1 == set2, "set2 shall not be identical with set1")
	assert.True(t, set1.Equals(set2), "set2 shall be equal to set1")
	set1.Add(4)
	set2.Add(5)
	assert.False(t, set1.Equals(set2), "set2 shall not be equal to set1 anymore")
}

func TestSet3ContainsAll(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set2 := FromArray([]int{1, 2, 3})
	assert.True(t, set1.ContainsAll(set2), "set2 shall be a subset of set1")
	set2.Remove(3)
	assert.True(t, set1.ContainsAll(set2), "set2 shall be a subset of set1")
	set2.Remove(2)
	assert.True(t, set1.ContainsAll(set2), "set2 shall be a subset of set1")
	set2.Remove(1)
	assert.True(t, set1.ContainsAll(set2), "set2 shall be a subset of set1")
	empty := Empty[int]()
	assert.True(t, empty.ContainsAll(set2), "set2 shall be a subset of an empty set")
	set3 := set1.Clone()
	set3.Add(4)
	assert.False(t, set1.ContainsAll(set3), "set3 shall not be a subset of set1")
	assert.False(t, empty.ContainsAll(set3), "set3 shall be a subset of an empty set")
	assert.True(t, set1.ContainsAll(set1), "set1 shall be a subset of set1")
	set4 := set1.Clone()
	assert.True(t, set1.ContainsAll(set4), "set4 shall be a subset of set1")
	set5 := FromArray([]int{3, 4, 5})
	assert.False(t, set1.ContainsAll(set5), "set5 shall not be a subset of set1")
}

func TestSet3ContainsAllFromArray(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	assert.True(t, set1.ContainsAllFromArray([]int{1, 2}), "[]int{1,2} shall be a subset of set1")
	assert.True(t, set1.ContainsAllFromArray([]int{2}), "[]int{2} shall be a subset of set1")
	assert.True(t, set1.ContainsAllFromArray([]int{}), "[]int{} shall be a subset of set1")
	empty := Empty[int]()
	assert.True(t, empty.ContainsAllFromArray([]int{}), "[]int{} shall be a subset of empty")
	assert.True(t, set1.ContainsAllFromArray([]int{2, 2, 2, 2, 2, 2}), "[]int{2,2,2,2,2,2} shall be a subset of set1")
	assert.False(t, set1.ContainsAllFromArray([]int{2, 4}), "[]int{2,4} shall be a subset of set1")
	assert.False(t, set1.ContainsAllFromArray([]int{4}), "[]int{4} shall be a subset of set1")
}

func TestSet3ContainsAllOf(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	assert.True(t, set1.ContainsAllOf(1, 2), "1,2 shall be a subset of set1")
	assert.True(t, set1.ContainsAllOf(2), "2 shall be a subset of set1")
	assert.True(t, set1.ContainsAllOf(), "no args shall be a subset of set1")
	assert.True(t, set1.ContainsAllOf(nil...), "nil shall be a subset of set1")
	empty := Empty[int]()
	assert.True(t, empty.ContainsAllOf(), "[]int{} shall be a subset of empty")
	assert.True(t, set1.ContainsAllOf(2, 2, 2, 2, 2, 2), "[]int{2,2,2,2,2,2} shall be a subset of set1")
	assert.False(t, set1.ContainsAllOf(2, 4), "[]int{2,4} shall be a subset of set1")
	assert.False(t, set1.ContainsAllOf(4), "[]int{4} shall be a subset of set1")
}

func TestSet3Unite(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set2 := FromArray([]int{4, 5, 6})
	set3 := FromArray([]int{1, 2, 3, 4, 5, 6})
	u1 := set1.Unite(set2)
	assert.False(t, set1.Equals(u1), "set1 shall not be altered by union")
	assert.True(t, u1.Equals(set3), "u1 and set3 shall be equal")
	u2 := set2.Unite(set1)
	assert.True(t, u2.Equals(set3), "u2 and set3 shall be equal")
	empty := Empty[int]()
	u3 := set1.Unite(empty)
	assert.True(t, set1.Equals(u3), "set1 shall be equal to u3")
	u4 := empty.Unite(set1)
	assert.True(t, set1.Equals(u4), "set1 shall be equal to u4")
	set4 := FromArray([]int{2, 3, 4, 5, 6})
	set5 := FromArray([]int{1, 2, 3, 4, 5, 6})
	u5 := set1.Unite(set4)
	assert.True(t, u5.Equals(set5), "u5 and set5 shall be equal")
}

func TestSet3AddAll(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set2 := FromArray([]int{4, 5, 6})
	set3 := FromArray([]int{1, 2, 3, 4, 5, 6})
	set1.AddAll(set2)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	empty := Empty[int]()
	set1.AddAll(empty)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAll(set2)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
}

func TestSet3AddAllFromArray(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set3 := FromArray([]int{1, 2, 3, 4, 5, 6})
	set1.AddAllFromArray([]int{4, 5, 6})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAllFromArray([]int{})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAllFromArray([]int{4, 5, 6})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
}

func TestSet3AddAllOf(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set3 := FromArray([]int{1, 2, 3, 4, 5, 6})
	set1.AddAllOf(4, 5, 6)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAllOf()
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAllOf(nil...)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.AddAllOf(4, 5, 6)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
}

func TestSet3Intersect(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3, 4})
	clone := set1.Clone()
	set2 := FromArray([]int{3, 4, 5, 6})
	set3 := FromArray([]int{3, 4})
	i1 := set1.Intersect(set2)
	assert.True(t, set1.Equals(clone), "set1 shall not be altered by intersection")
	assert.True(t, i1.Equals(set3), "i1 and set3 shall be equal")

	empty := Empty[int]()
	i2 := set1.Intersect(empty)
	assert.True(t, empty.Equals(i2), "empty shall be equal to i2")
	i3 := empty.Intersect(set1)
	assert.True(t, empty.Equals(i3), "empty shall be equal to i3")

	set4 := FromArray([]int{1, 2, 3})
	set5 := FromArray([]int{4, 5, 6})
	i4 := set4.Intersect(set5)
	assert.True(t, empty.Equals(i4), "empty shall be equal to i4")
}

func TestSet3IntersectWithArray(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3, 4})
	clone := set1.Clone()
	set3 := FromArray([]int{3, 4})
	i1 := set1.IntersectWithArray([]int{3, 4, 5, 6})
	assert.True(t, set1.Equals(clone), "set1 shall not be altered by intersection")
	assert.True(t, i1.Equals(set3), "i1 and set3 shall be equal")

	empty := Empty[int]()
	i2 := set1.IntersectWithArray([]int{})
	assert.True(t, empty.Equals(i2), "empty shall be equal to i2")
	i3 := empty.IntersectWithArray([]int{3, 4, 5, 6})
	assert.True(t, empty.Equals(i3), "empty shall be equal to i3")

	set4 := FromArray([]int{1, 2, 3})
	i4 := set4.IntersectWithArray([]int{4, 5, 6})
	assert.True(t, empty.Equals(i4), "empty shall be equal to i4")
}

func TestSet3RemoveAll(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3})
	set2 := FromArray([]int{3, 4, 5, 6})
	set3 := FromArray([]int{1, 2})
	set1.RemoveAll(set2)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	empty := Empty[int]()
	set1.RemoveAll(empty)
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.RemoveAll(set3)
	assert.True(t, set1.Equals(empty), "set1 and empty shall be equal")
}

func TestSet3RemoveAllFromArray(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3, 4, 5, 6})
	set3 := FromArray([]int{1, 2, 3, 4})
	set1.RemoveAllFromArray([]int{5, 6, 7, 8})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.RemoveAllFromArray([]int{})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
	set1.RemoveAllFromArray([]int{9, 10})
	assert.True(t, set1.Equals(set3), "set1 and set3 shall be equal")
}

func TestSet3Subtract(t *testing.T) {
	set1 := FromArray([]int{1, 2, 3, 4})
	set2 := FromArray([]int{3, 4, 5, 6})
	set3 := FromArray([]int{1, 2})
	i1 := set1.Subtract(set2)
	assert.False(t, set1.Equals(i1), "set1 shall not be altered by intersection")
	assert.True(t, i1.Equals(set3), "i1 and set3 shall be equal")

	empty := Empty[int]()
	i2 := set1.Subtract(empty)
	assert.True(t, set1.Equals(i2), "set1 shall be equal to i2")
}

func TestSet3Rehash(t *testing.T) {
	data := genUint32Data(53)
	set := FromArray(data)
	assert.True(t, len(set.groupCtrl) == 12, "set shall contain 12 groups")
	set.RehashToCapacity(200)
	assert.True(t, len(set.groupCtrl) == 31, "set shall contain 30 groups")
	for _, e := range data {
		assert.True(t, set.Contains(e), "set shall contain %v", e)
	}
	set.RehashToCapacity(20)
	assert.True(t, len(set.groupCtrl) == 31, "set shall contain 30 groups")
	for _, e := range data {
		assert.True(t, set.Contains(e), "set shall contain %v", e)
	}
	set.Rehash()
	assert.True(t, len(set.groupCtrl) == 9, "set shall contain 9 groups")
	for _, e := range data {
		assert.True(t, set.Contains(e), "set shall contain %v", e)
	}
}

func TestSet3ToArray(t *testing.T) {
	set := FromArray([]int{1, 2, 2, 3})
	ary := set.ToArray()
	assert.True(t, len(ary) == 3, "the array shall contain 3 elements")
	newSet := FromArray(ary)
	assert.True(t, set.Equals(newSet), "both sets shall contain the same 3 elements")
	empty := Empty[int]()
	ary = empty.ToArray()
	assert.True(t, len(ary) == 0, "the array shall contain 0 elements")
}

func TestExample(t *testing.T) {
	// create a new Set3
	set1 := Empty[int]()
	// add some elements
	set1.Add(1)
	set1.Add(2)
	set1.Add(3)
	// add some more elements
	set1.AddAllOf(4, 5, 6)
	// create a second set directly from an array
	set2 := FromArray([]int{2, 3, 4, 5})
	// check if set2 is a subset of set1. must be true in this case
	isSubset := set1.ContainsAll(set2)
	assert.True(t, isSubset, "%v is not a subset of %v", set2, set1)
	// mathematical operations like Unite, Subtract and Intersect
	// do not manipulate a Set3 but return a new set
	intersect := set1.Intersect(set2)
	// compare sets. as set2 is a subset of set1, intersect must be equal to set2
	equal := intersect.Equals(set2)
	assert.True(t, equal, "%v is not equal to %v", intersect, set2)
}

func TestNil(t *testing.T) {
	empty := Empty[int]()

	set := FromArray[int](nil)
	assert.True(t, empty.Equals(set), "%v is not equal to %v", set, empty)

	set.AddAllFromArray([]int{1, 2, 3})
	ref := set.Clone()

	b := set.ContainsAll(nil)
	assert.True(t, b, "every set shall contain the empty set")

	c := set.ContainsAllFromArray(nil)
	assert.True(t, c, "every set shall contain the empty set")

	d := empty.Equals(nil)
	assert.True(t, d, "an empty set shall be equal to nil")

	d2 := set.Equals(nil)
	assert.False(t, d2, "a non-empty set shall not be equal to nil")

	set.AddAll(nil)
	assert.True(t, set.Equals(ref), "%v is not equal to %v", set, ref)

	set.AddAllFromArray(nil)
	assert.True(t, set.Equals(ref), "%v is not equal to %v", set, ref)

	u := set.Unite(nil)
	assert.True(t, u.Equals(set), "%v is not equal to %v", u, set)

	set.RemoveAll(nil)
	assert.True(t, set.Equals(ref), "%v is not equal to %v", set, ref)

	set.RemoveAllFromArray(nil)
	assert.True(t, set.Equals(ref), "%v is not equal to %v", set, ref)

	diff := set.Subtract(nil)
	assert.True(t, diff.Equals(set), "%v is not equal to %v", diff, set)

	intersect := set.Intersect(nil)
	assert.True(t, intersect.Equals(empty), "%v is not equal to %v", intersect, empty)

	intersectFrom := set.IntersectWithArray(nil)
	assert.True(t, intersectFrom.Equals(empty), "%v is not equal to %v", intersect, empty)

	bany := set.ContainsAny(nil)
	assert.False(t, bany, "set cannot contain any elements from nil")

	banyfrom := set.ContainsAnyFromArray(nil)
	assert.False(t, banyfrom, "set cannot contain any elements from nil")

	var nilSet *Set3[int]
	s := nilSet.String()
	assert.True(t, s == "{nil}", "value shall be '{nil}'")
}

func TestSet3ContainsAny(t *testing.T) {
	tests := []struct {
		name     string
		thisSet  *Set3[int]
		thatSet  *Set3[int]
		expected bool
	}{
		{
			name:     "No common elements",
			thisSet:  FromArray[int]([]int{1, 2, 3}),
			thatSet:  FromArray[int]([]int{4, 5, 6}),
			expected: false,
		},
		{
			name:     "Some common elements (this bigger)",
			thisSet:  FromArray[int]([]int{1, 2, 3, 4}),
			thatSet:  FromArray[int]([]int{3, 4, 5}),
			expected: true,
		},
		{
			name:     "Some common elements (that bigger)",
			thisSet:  FromArray[int]([]int{1, 2, 3}),
			thatSet:  FromArray[int]([]int{3, 4, 5, 6}),
			expected: true,
		},
		{
			name:     "All elements common",
			thisSet:  FromArray[int]([]int{1, 2, 3}),
			thatSet:  FromArray[int]([]int{1, 2, 3}),
			expected: true,
		},
		{
			name:     "Empty sets",
			thisSet:  FromArray[int]([]int{}),
			thatSet:  FromArray[int]([]int{}),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.thisSet.ContainsAny(tt.thatSet)
			if result != tt.expected {
				t.Errorf("ContainsAny() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestSet3ContainsAnyFromArray(t *testing.T) {
	tests := []struct {
		name     string
		set      *Set3[int]
		array    []int
		expected bool
	}{
		{
			name:     "No common elements",
			set:      FromArray[int]([]int{1, 2, 3}),
			array:    []int{4, 5, 6},
			expected: false,
		},
		{
			name:     "Some common elements",
			set:      FromArray[int]([]int{1, 2, 3, 4}),
			array:    []int{0, 3, 4, 5},
			expected: true,
		},
		{
			name:     "All elements common",
			set:      FromArray[int]([]int{1, 2, 3}),
			array:    []int{1, 2, 3},
			expected: true,
		},
		{
			name:     "Empty sets",
			set:      FromArray[int]([]int{}),
			array:    []int{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.set.ContainsAnyFromArray(tt.array)
			if result != tt.expected {
				t.Errorf("ContainsAnyFrom() = %v, expected %v", result, tt.expected)
			}
		})
	}
}

func TestContainsAnyOf(t *testing.T) {
	// Test case 1: Contains one of the elements
	set := From(1, 2, 3)
	if !set.ContainsAnyOf(2, 4) {
		t.Errorf("ContainsAnyOf failed to return true for element 2")
	}

	// Test case 2: Does not contain any of the elements
	set = From(1, 2)
	if set.ContainsAnyOf(3, 4) {
		t.Errorf("ContainsAnyOf incorrectly returned true for elements 3 and 4")
	}

	// Test case 3: Contains all of the elements
	set = From(1, 2)
	if !set.ContainsAnyOf(1, 2) {
		t.Errorf("ContainsAnyOf failed to return true for elements 1 and 2")
	}

	// Test case 4: Empty set
	set = Empty[int]()
	if set.ContainsAnyOf(1, 2) {
		t.Errorf("ContainsAnyOf incorrectly returned true for an empty set")
	}

	// Test case 5: No arguments
	set = From(1)
	if set.ContainsAnyOf() {
		t.Errorf("ContainsAnyOf incorrectly returned true with no arguments")
	}

	// Test case 6: Nil arguments
	set = From(1)
	if set.ContainsAnyOf(nil...) {
		t.Errorf("ContainsAnyOf incorrectly returned true with nil arguments")
	}
}

func TestRemoveAllOf(t *testing.T) {
	// Test case 1: Remove multiple elements
	set := From(1, 2, 3)
	set.RemoveAllOf(1, 2)
	if set.Contains(1) || set.Contains(2) {
		t.Errorf("RemoveAllOf failed to remove elements 1 and 2")
	}
	if !set.Contains(3) {
		t.Errorf("RemoveAllOf incorrectly removed element 3")
	}

	// Test case 2: Remove elements not in the set
	set = From(1)
	set.RemoveAllOf(2, 3)
	if !set.Contains(1) {
		t.Errorf("RemoveAllOf incorrectly removed element 1")
	}

	// Test case 3: Remove from an empty set
	set = Empty[int]()
	set.RemoveAllOf(1, 2)
	if set.Size() != 0 {
		t.Errorf("RemoveAllOf failed on empty set")
	}

	// Test case 4: Remove with no arguments
	set = From(1)
	set.RemoveAllOf()
	if !set.Contains(1) {
		t.Errorf("RemoveAllOf incorrectly removed element 1 with no arguments")
	}

	// Test case 5: Remove with nil arguments
	set = From(1)
	set.RemoveAllOf(nil...)
	if !set.Contains(1) {
		t.Errorf("RemoveAllOf incorrectly removed element 1 with nil arguments")
	}
}

func expectedHasZeroByte(x uint64) uint64 {
	var res uint64
	for i := range 8 {
		if ((x >> (8 * i)) & 0xFF) == 0 {
			res |= uint64(0x80) << (8 * i)
		}
	}
	return res
}

func TestSet3HasZeroByte_Table(t *testing.T) {
	cases := []uint64{
		0x0000000000000000,
		0xFFFFFFFFFFFFFFFF,
		0x00FF00FF00FF00FF,
		0xFF00FF00FF00FF00,
		0x1122334455667788,
		0x1122330044556677, // one zero byte in middle
		0x8000000000000080, // zero bytes in between high bytes
	}

	for _, in := range cases {
		got := set3hasZeroByte(in)
		want := expectedHasZeroByte(in)
		if got != want {
			t.Fatalf("set3hasZeroByte(0x%016x) = 0x%016x; want 0x%016x", in, got, want)
		}
	}
}

func TestSet3HasZeroByte_Random(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		v := r.Uint64()
		got := set3hasZeroByte(v)
		want := expectedHasZeroByte(v)
		if got != want {
			t.Fatalf("random #%d: set3hasZeroByte(0x%016x) = 0x%016x; want 0x%016x", i, v, got, want)
		}
	}
}

func TestSet3NextMatch_SingleBits(t *testing.T) {
	cases := []int{0, 1, 7, 8, 15, 16, 31, 32, 63}
	for _, pos := range cases {
		var b uint64 = 1 << uint(pos)
		orig := b
		got := set3nextMatch(&b)
		want := pos >> 3
		if got != want {
			t.Fatalf("pos=%d: got index %d, want %d", pos, got, want)
		}
		if b != 0 {
			t.Fatalf("pos=%d: expected bit cleared, remaining 0x%016x (orig 0x%016x)", pos, b, orig)
		}
	}
}

func TestSet3NextMatch_SequenceOrderAndClear(t *testing.T) {
	var mask uint64 = (1 << 7) | (1 << 15) | (1 << 31) | (1 << 63) // bytes 0,1,3,7 (high bit of each byte)
	expected := []int{0, 1, 3, 7}
	for i, want := range expected {
		got := set3nextMatch(&mask)
		if got != want {
			t.Fatalf("step %d: got %d, want %d", i, got, want)
		}
	}
	if mask != 0 {
		t.Fatalf("mask not fully cleared, remaining 0x%016x", mask)
	}
}

func TestSet3NextMatch_HighLowMix(t *testing.T) {
	// bits at positions 0,7,8,15 -> bytes 0,0,1,1 -> expected order: bit0 (byte0), bit7 (byte0, higher pos still in same byte order), then byte1 bits
	var b uint64 = (1 << 0) | (1 << 7) | (1 << 8) | (1 << 15)
	// Expected sequence: least significant set bit positions are 0,7,8,15 -> bytes 0,0,1,1 -> returns 0,0,1,1
	wantSeq := []int{0, 0, 1, 1}
	for i, want := range wantSeq {
		got := set3nextMatch(&b)
		if got != want {
			t.Fatalf("step %d: got %d, want %d", i, got, want)
		}
	}
	if b != 0 {
		t.Fatalf("expected b cleared, got 0x%016x", b)
	}
}
