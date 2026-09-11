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
	"encoding/binary"
	"unsafe"

	"github.com/TomTonic/Set3/hashing"
)

// The serial byte-block routines that the hashing package used until the
// lane-parallel body replaced them.
//
// They are kept here, verbatim, for one reason: the speed claim in the README
// is a comparison, and a comparison needs both sides to stay runnable. Without
// this file the benchmarks would measure the new routine against nothing, and
// the next person to touch the hash would have no way to check whether they had
// given the two to four times back.
//
// They are not a fallback and nothing should call them outside the lab. Two of
// the defects that motivated the replacement are visible here if you look for
// them: SerialBytesBlock("") returns zero for every seed, because the tail
// constant for the empty case is P1 and WH64Det starts by XORing P1 into its
// input — so the chain is multiplied by zero before the seed ever reaches it.

// SerialBytesBlock is the superseded hashing.HashBytesBlock: an eight-byte loop
// over a strict dependency chain, closed by a seven-case fallthrough tail that
// assembles the remaining bytes with shifts and ors.
func SerialBytesBlock(seed uint64, b []byte) uint64 {
	h := seed ^ hashing.P0
	i, n := 0, len(b)
	for i+8 <= n {
		v := binary.NativeEndian.Uint64(b[i:])
		h = hashing.WH64Det(v, h)
		i += 8
	}
	var tail uint64
	switch n - i {
	case 7:
		tail |= uint64(b[i+6]) << 48
		fallthrough
	case 6:
		tail |= uint64(b[i+5]) << 40
		fallthrough
	case 5:
		tail |= uint64(b[i+4]) << 32
		fallthrough
	case 4:
		tail |= uint64(b[i+3]) << 24
		fallthrough
	case 3:
		tail |= uint64(b[i+2]) << 16
		fallthrough
	case 2:
		tail |= uint64(b[i+1]) << 8
		fallthrough
	case 1:
		tail |= uint64(b[i])
	case 0:
		tail = hashing.P1
	}
	return hashing.WH64Det(tail^uint64(n)*hashing.P2, h) //nolint:gosec
}

// SerialString is the superseded hashing.HashString.
func SerialString(p unsafe.Pointer, seed uint64) uint64 {
	s := *(*string)(p)
	n := len(s)
	h := seed ^ hashing.P0
	if n == 0 {
		return hashing.WH64Det(hashing.P1^uint64(n)*hashing.P2, h) //nolint:gosec
	}
	dp := unsafe.Pointer(unsafe.StringData(s)) //nolint:gosec
	i := 0
	for i+8 <= n {
		v := *(*uint64)(unsafe.Add(dp, i)) //nolint:gosec
		h = hashing.WH64Det(v, h)
		i += 8
	}
	var tail uint64
	switch n - i {
	case 7:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+6))) << 48 //nolint:gosec
		fallthrough
	case 6:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+5))) << 40 //nolint:gosec
		fallthrough
	case 5:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+4))) << 32 //nolint:gosec
		fallthrough
	case 4:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+3))) << 24 //nolint:gosec
		fallthrough
	case 3:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+2))) << 16 //nolint:gosec
		fallthrough
	case 2:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i+1))) << 8 //nolint:gosec
		fallthrough
	case 1:
		tail |= uint64(*(*byte)(unsafe.Add(dp, i))) //nolint:gosec
	case 0:
		tail = hashing.P1
	}
	return hashing.WH64Det(tail^uint64(n)*hashing.P2, h) //nolint:gosec
}

// SerialBlock16, SerialBlock24 and SerialBlock32 are the superseded fixed-size
// helpers: the loop unrolled, which removes the loop and leaves the dependency
// chain exactly as long as it was. That is the point the measurement makes —
// the chain is the cost, not the loop.
func SerialBlock16(p unsafe.Pointer, seed uint64) uint64 {
	b := unsafe.Slice((*byte)(p), 16) //nolint:gosec
	h := seed ^ hashing.P0
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[:8]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[8:16]), h)
	n := uint64(len(b))
	return hashing.WH64Det(uint64(hashing.P1)^n*uint64(hashing.P2), h)
}

func SerialBlock24(p unsafe.Pointer, seed uint64) uint64 {
	b := unsafe.Slice((*byte)(p), 24) //nolint:gosec
	h := seed ^ hashing.P0
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[:8]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[8:16]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[16:24]), h)
	n := uint64(len(b))
	return hashing.WH64Det(uint64(hashing.P1)^n*uint64(hashing.P2), h)
}

func SerialBlock32(p unsafe.Pointer, seed uint64) uint64 {
	b := unsafe.Slice((*byte)(p), 32) //nolint:gosec
	h := seed ^ hashing.P0
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[:8]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[8:16]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[16:24]), h)
	h = hashing.WH64Det(binary.NativeEndian.Uint64(b[24:32]), h)
	n := uint64(len(b))
	return hashing.WH64Det(uint64(hashing.P1)^n*uint64(hashing.P2), h)
}
