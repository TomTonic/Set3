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

package hashperf

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
	"github.com/TomTonic/Set3/lab/hashalt"
)

var laneSink uint64

// laneSeed is fixed so that a run is reproducible.
const laneSeed = uint64(0x243f6a8885a308d3)

// makeStrings builds count distinct strings of exactly size bytes.
func makeStrings(size, count int) []string {
	const digits = "0123456789abcdefghijklmnopqrstuvwxyz"
	out := make([]string, count)
	for i := range out {
		buf := make([]byte, size)
		v := uint64(i)*0x9e3779b97f4a7c15 + 1 //nolint:gosec
		for j := range buf {
			buf[j] = digits[v%36]
			v = v/36 + uint64(j)
		}
		out[i] = string(buf)
	}
	return out
}

// BenchmarkHashThroughput measures independent hashes back to back, which is
// what a stream of unrelated lookups produces: the processor can overlap one
// hash with the next, so this reports how much *work* each routine is, not how
// long it takes to get an answer.
func BenchmarkHashThroughput(b *testing.B) {
	for _, size := range []int{8, 12, 16, 20, 24, 32, 64} {
		keys := makeStrings(size, 4096)
		b.Run(fmt.Sprintf("len=%d", size), func(b *testing.B) {
			b.Run("production HashString", func(b *testing.B) {
				var acc uint64
				for i := 0; i < b.N; i++ {
					s := keys[i&4095]
					acc ^= hashing.HashString(unsafe.Pointer(&s), laneSeed)
				}
				laneSink ^= acc
			})
			b.Run("lane-parallel", func(b *testing.B) {
				var acc uint64
				for i := 0; i < b.N; i++ {
					s := keys[i&4095]
					acc ^= hashalt.WHLaneString(unsafe.Pointer(&s), laneSeed)
				}
				laneSink ^= acc
			})
		})
	}
}

// BenchmarkHashLatency measures a chain in which each hash chooses the next
// input, so nothing can overlap. This is the dependency chain the routines
// differ in, and it is the number that bounds a lookup whose probe cannot
// start until the hash is done.
func BenchmarkHashLatency(b *testing.B) {
	for _, size := range []int{8, 12, 16, 20, 24, 32, 64} {
		keys := makeStrings(size, 4096)
		b.Run(fmt.Sprintf("len=%d", size), func(b *testing.B) {
			b.Run("production HashString", func(b *testing.B) {
				h := laneSeed
				for i := 0; i < b.N; i++ {
					s := keys[h&4095]
					h = hashing.HashString(unsafe.Pointer(&s), laneSeed)
				}
				laneSink ^= h
			})
			b.Run("lane-parallel", func(b *testing.B) {
				h := laneSeed
				for i := 0; i < b.N; i++ {
					s := keys[h&4095]
					h = hashalt.WHLaneString(unsafe.Pointer(&s), laneSeed)
				}
				laneSink ^= h
			})
		})
	}
}

// BenchmarkFixedBlockLatency compares the fixed-size raw-block helpers, which
// is the path a struct key takes. The 24-byte case is a three-field uint64
// struct, the shape the suite measures as struct3x64.
func BenchmarkFixedBlockLatency(b *testing.B) {
	var k16 [16]byte
	var k24 [24]byte
	var k32 [32]byte

	b.Run("16 bytes", func(b *testing.B) {
		b.Run("production", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k16), hashing.HashAsByteArray[[16]byte]) })
		b.Run("lane-parallel", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k16), hashalt.WHLaneBlock16) })
	})
	b.Run("24 bytes", func(b *testing.B) {
		b.Run("production", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k24), hashing.HashAsByteArray[[24]byte]) })
		b.Run("lane-parallel", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k24), hashalt.WHLaneBlock24) })
	})
	b.Run("32 bytes", func(b *testing.B) {
		b.Run("production", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k32), hashing.HashAsByteArray[[32]byte]) })
		b.Run("lane-parallel", func(b *testing.B) { benchFixed(b, unsafe.Pointer(&k32), hashalt.WHLaneBlock32) })
	})
}

// benchFixed chains the hash into its own next seed, so the measurement is the
// routine's latency rather than the loop's throughput.
func benchFixed(b *testing.B, p unsafe.Pointer, h func(unsafe.Pointer, uint64) uint64) {
	acc := laneSeed
	for i := 0; i < b.N; i++ {
		acc = h(p, acc)
	}
	laneSink ^= acc
}
