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

package hashalt

import (
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

// Lane-parallel candidates for the multi-word hash routines.
//
// # The problem they address
//
// Every multi-word routine in the production hashing package is a strict
// dependency chain. HashString, HashBytesBlock and the fixed-size
// hashByteBlockNN helpers all have the shape
//
//	h = WH64Det(word0, h); h = WH64Det(word1, h); ...
//
// and WH64Det is itself two dependent widening multiplies. So an N-word input
// costs 2N multiplies *in series*, and the processor cannot start the second
// until the first has retired. The fixed-size helpers unroll the loop, which
// removes the loop overhead and leaves the chain exactly as it was — the chain
// is the cost, not the loop.
//
// The arithmetic matches the measurement. On Zen 4 a 64x64 widening multiply
// has a latency of three cycles, so a 20-byte string is two loop rounds plus a
// tail round, six dependent multiplies, about 20 cycles, about 3.9 ns at
// 5 GHz. The production routine measures 3.93 ns.
//
// # What these do instead
//
// Independent lanes. Words are consumed in pairs by separate accumulators that
// do not depend on one another, and the accumulators are combined once at the
// end. A 24-byte key goes from eight dependent multiplies to two levels: two
// that issue together, then one to combine. The structure is wyhash's own, not
// an invention — wyhash processes 48-byte blocks through three independent
// lanes for exactly this reason.
//
// Short inputs additionally avoid the byte-at-a-time tail. The production
// routines end with a switch of seven fallthrough cases that assembles the
// remaining bytes with shifts and ors, each dependent on the last. These read
// the first eight bytes and the *last* eight bytes instead, overlapping in the
// middle, which costs two loads and no branches. Below eight bytes the same
// trick works with four-byte reads, and below four with wyhash's three-byte
// gather.
//
// # What is not settled here
//
// Speed is only half of it. A hash with fewer mixing rounds can be faster and
// worse, and "worse" for a Swiss table means the low seven bits that pick the
// H2 tag stop being uniform, or nearby keys stop landing in different groups.
// Nothing here should be promoted into the production package until the
// hashquality suite — determinism, distinct outputs, uniformity of the lowest
// seven bits, bucket mapping and avalanche — says it is at least as good as
// what it would replace.

// P0..P3 and M5 are the production package's mixing constants, reused so that
// these candidates differ from it in structure only.
const (
	p0 = hashing.P0
	p1 = hashing.P1
	p2 = hashing.P2
	p3 = hashing.P3
	m5 = hashing.M5
)

// mix is the production package's Mix: a widening multiply folded to 64 bits.
// One multiply, and the unit this file counts.
func mix(a, b uint64) uint64 { return hashing.Mix(a, b) }

// read8 and read4 load unaligned words at an offset.
func read8(p unsafe.Pointer, off int) uint64 {
	return *(*uint64)(unsafe.Add(p, off)) //nolint:gosec
}

func read4(p unsafe.Pointer, off int) uint64 {
	return uint64(*(*uint32)(unsafe.Add(p, off))) //nolint:gosec
}

// WHLaneBlock16 hashes exactly 16 bytes with one level of parallel work and
// one to combine: two multiplies deep, against the production helper's six.
func WHLaneBlock16(p unsafe.Pointer, seed uint64) uint64 {
	a := mix(read8(p, 0)^p1, read8(p, 8)^seed)
	return mix(a^p0, m5^16)
}

// WHLaneBlock24 hashes exactly 24 bytes. Two lanes issue together, one
// multiply combines them: two levels deep against the production helper's
// eight. This is the size of a three-field uint64 struct key, which is the
// shape the raw-block hasher sees most often.
func WHLaneBlock24(p unsafe.Pointer, seed uint64) uint64 {
	a := mix(read8(p, 0)^p1, read8(p, 8)^seed)
	b := mix(read8(p, 16)^p3, seed^p2)
	return mix(a^b, m5^24)
}

// WHLaneBlock32 hashes exactly 32 bytes through two independent lanes.
func WHLaneBlock32(p unsafe.Pointer, seed uint64) uint64 {
	a := mix(read8(p, 0)^p1, read8(p, 8)^seed)
	b := mix(read8(p, 16)^p3, read8(p, 24)^seed)
	return mix(a^b, m5^32)
}

// WHLaneShort hashes 0 to 16 bytes without a loop and without a byte-wise
// tail, by reading the first and last words and letting them overlap.
//
// The overlap is what removes the tail: for any length in [8,16] the two reads
// together cover every byte, and no byte is missed or duplicated in a way the
// length mixed into the final step cannot separate. Below eight bytes the same
// holds for four-byte reads, and below four the three-byte gather gives first,
// middle and last, which distinguishes every length and every content at those
// sizes.
func WHLaneShort(p unsafe.Pointer, n int, seed uint64) uint64 {
	var a, b uint64
	switch {
	case n >= 8:
		a = read8(p, 0)
		b = read8(p, n-8)
	case n >= 4:
		a = read4(p, 0)
		b = read4(p, n-4)
	case n > 0:
		first := uint64(*(*byte)(p))
		mid := uint64(*(*byte)(unsafe.Add(p, n>>1))) //nolint:gosec
		last := uint64(*(*byte)(unsafe.Add(p, n-1))) //nolint:gosec
		a = first<<16 | mid<<8 | last
		b = 0
	default:
		a, b = p1, p2
	}
	return mix(mix(a^p1, b^seed), m5^uint64(n)) //nolint:gosec
}

// WHLaneString is the length-dispatched, lane-parallel counterpart to
// hashing.HashString.
//
// The dispatch matters as much as the lanes: the production routine runs its
// eight-byte loop and then its seven-case tail for every length, so a 20-byte
// key pays for three dependent rounds and a chain of shifts. Here the common
// sizes are straight-line code, and only inputs past 32 bytes enter a loop —
// which then runs three independent lanes so that the chain does not come back
// for long inputs either.
func WHLaneString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	n := len(s)
	dp := unsafe.Pointer(unsafe.StringData(s)) //nolint:gosec
	return whLaneBytes(dp, n, seed)
}

// WHLaneBytes is WHLaneString for a byte slice, the counterpart to
// hashing.HashBytesBlock.
func WHLaneBytes(seed uint64, b []byte) uint64 {
	if len(b) == 0 {
		return whLaneBytes(nil, 0, seed)
	}
	return whLaneBytes(unsafe.Pointer(&b[0]), len(b), seed) //nolint:gosec
}

// whLaneBytes is the shared body. It is one function so that strings and byte
// slices cannot drift apart in either speed or hash value.
func whLaneBytes(p unsafe.Pointer, n int, seed uint64) uint64 {
	switch {
	case n <= 16:
		return WHLaneShort(p, n, seed)
	case n <= 32:
		// Four reads covering everything, overlapping in the middle, through
		// two lanes that issue together.
		a := mix(read8(p, 0)^p1, read8(p, 8)^seed)
		b := mix(read8(p, n-16)^p3, read8(p, n-8)^seed)
		return mix(a^b, m5^uint64(n)) //nolint:gosec
	}

	// Longer inputs: three lanes over 24-byte strides, so the dependency chain
	// per lane is one multiply per 24 bytes rather than two per 8.
	l1, l2, l3 := seed^p1, seed^p2, seed^p3
	off, rest := 0, n
	for rest >= 24 {
		l1 = mix(read8(p, off)^p1, l1)
		l2 = mix(read8(p, off+8)^p2, l2)
		l3 = mix(read8(p, off+16)^p3, l3)
		off += 24
		rest -= 24
	}
	// Whatever is left is covered by the last 16 bytes of the input, which
	// overlap back into what the loop already read. Overlap costs nothing and
	// removes the tail entirely.
	a := mix(read8(p, n-16)^p1, read8(p, n-8)^seed)
	return mix(l1^l2^l3^a, m5^uint64(n)) //nolint:gosec
}
