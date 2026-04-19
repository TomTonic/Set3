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

// nolint:gosec // Test file: intentional use of math/rand and uint32 conversions for deterministic testing
package set3

import (
	"math/bits"
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/stretchr/testify/assert"
)

func BenchmarkStringSets(b *testing.B) {
	const keySz = 8
	sizes := []int{16, 128, 1024, 8192, 131072}
	for _, n := range sizes {
		keys := genStringData(keySz, n)
		b.Run("n="+strconv.Itoa(n), func(b *testing.B) {
			b.Run("runtime Set", func(b *testing.B) {
				benchmarkRuntimeSet(b, keys)
			})
			b.Run("swiss.Set", func(b *testing.B) {
				benchmarkSwissSet(b, keys)
			})
		})
	}
}

func BenchmarkInt64Sets(b *testing.B) {
	sizes := []int{16, 128, 1024, 8192, 131072, 524288}
	for _, n := range sizes {
		b.Run("n="+strconv.Itoa(n), func(b *testing.B) {
			keys := generateInt64Data(n)
			b.Run("runtime Set", func(b *testing.B) {
				benchmarkRuntimeSet(b, keys)
			})
			b.Run("swiss.Set", func(b *testing.B) {
				benchmarkSwissSet(b, keys)
			})
		})
	}
}

func TestMemoryFootprintSet(t *testing.T) {
	t.Skip("unskip for memory footprint stats - runs 1-2 minutes")
	var samples []float64
	for n := 10; n <= 350_000; n += 20 {
		b1 := testing.Benchmark(func(b *testing.B) {
			// max load factor 6.66666/8
			m := EmptyWithCapacity[int](uint32(n))
			require.NotNil(b, m)
		})
		b2 := testing.Benchmark(func(b *testing.B) {
			// max load factor 6.5/8
			m := make(map[int]struct{}, n)
			require.NotNil(b, m)
		})
		x := float64(b1.MemBytes) / float64(b2.MemBytes)
		samples = append(samples, x)
	}
	t.Logf("mean size ratio: %.3f", mean(samples))
}

func benchmarkRuntimeSet[K comparable](b *testing.B, keys []K) {
	n := uint32(len(keys))
	mod := n - 1 // power of 2 fast modulus
	require.Equal(b, 1, bits.OnesCount32(n))
	m := make(map[K]K, n)
	b.ResetTimer()
	for _, k := range keys {
		m[k] = k
	}
	var ok bool
	for i := 0; i < b.N; i++ {
		_, ok = m[keys[uint32(i)&mod]]
	}
	//	assert.True(b, ok)
	for i := b.N; i < b.N*2; i++ {
		_, ok = m[keys[uint32(i-b.N)&mod]]
	}
	assert.True(b, ok)
	b.ReportAllocs()
}

func benchmarkSwissSet[K comparable](b *testing.B, keys []K) {
	n := uint32(len(keys))
	mod := n - 1 // power of 2 fast modulus
	require.Equal(b, 1, bits.OnesCount32(n))
	m := EmptyWithCapacity[K](n)
	b.ResetTimer()
	for _, k := range keys {
		m.Add(k)
	}
	var ok bool
	for i := 0; i < b.N; i++ {
		ok = m.Contains(keys[uint32(i)&mod])
	}
	//	assert.True(b, ok)
	for i := b.N; i < b.N*2; i++ {
		ok = m.Contains(keys[uint32(i-b.N)&mod])
	}
	assert.True(b, ok)
	b.ReportAllocs()
}

func generateInt64Data(n int) (data []int64) {
	data = make([]int64, n)
	var x int64
	for i := range data {
		x += rand.Int63n(128) + 1
		data[i] = x
	}
	return
}

func mean(samples []float64) (m float64) {
	for _, s := range samples {
		m += s
	}
	return m / float64(len(samples))
}
