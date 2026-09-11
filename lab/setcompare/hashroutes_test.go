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
	"hash/maphash"
	"testing"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

var hashSink uint64

// HashSinkValue exists for the same reason as SinkValue: to be read.
func HashSinkValue() uint64 { return hashSink }

// BenchmarkStringHashRoutes measures what Set3 pays to hash a string against
// what Go's own runtime pays for the same string.
//
// It exists to settle a question the suite raises but cannot answer. Set3 loses
// the string lookup workloads, and the obvious explanation is the hash: the Go
// runtime hashes strings with AES instructions on amd64, and Set3 uses a
// portable wyhash routine that has no such shortcut. This benchmark checks that
// explanation instead of assuming it, and largely rules it out — measured on an
// AMD Ryzen 9 7900 with Go 1.26.8 over 20-byte keys, the two routines came to
// 3.83 ns and 3.48 ns. A third of a nanosecond accounts for the gap at a
// thousand elements (0.47 ns) and comes nowhere near the gap at a quarter of a
// million (12.6 ns).
//
// What is left as the candidate explanation is the load factor. Set3 fills to
// 83% where the native map runs near 47%, which is the whole of Set3's memory
// advantage and is bought with longer probe sequences. With a string key every
// candidate slot in a probe costs a pointer dereference into scattered string
// data, so a longer chain costs cache misses rather than instructions — which
// is invisible in L1 and expensive out of it. That is a hypothesis this
// benchmark does not test; it is recorded here so that the next person does not
// re-derive the part that has already been ruled out.
func BenchmarkStringHashRoutes(b *testing.B) {
	keys := buildKeys(makeStringKey, memberDomain, 4096)
	const mask = 4095
	seed := uint64(0x243f6a8885a308d3)
	runtimeSeed := maphash.MakeSeed()

	b.Run("Set3 hashing.HashString", func(b *testing.B) {
		var acc uint64
		for i := 0; i < b.N; i++ {
			s := keys[i&mask]
			acc += hashing.HashString(unsafe.Pointer(&s), seed)
		}
		hashSink += acc
	})
	b.Run("Go runtime via maphash.String", func(b *testing.B) {
		var acc uint64
		for i := 0; i < b.N; i++ {
			acc += maphash.String(runtimeSeed, keys[i&mask])
		}
		hashSink += acc
	})
}
