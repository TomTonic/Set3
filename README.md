# Set3

[![Go Report Card](https://goreportcard.com/badge/github.com/TomTonic/Set3)](https://goreportcard.com/report/github.com/TomTonic/Set3)
[![Go Reference](https://pkg.go.dev/badge/github.com/TomTonic/Set3.svg)](https://pkg.go.dev/github.com/TomTonic/Set3)
[![Tests](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml/badge.svg?branch=main)](https://github.com/TomTonic/Set3/actions/workflows/coverage.yml)
![coverage](https://raw.githubusercontent.com/TomTonic/Set3/badges/.badges/main/coverage.svg)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9470/badge)](https://www.bestpractices.dev/projects/9470)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/TomTonic/Set3/badge)](https://scorecard.dev/viewer/?uri=github.com/TomTonic/Set3)

Set3 is a high-performance, native Golang set implementation. It offers a significant improvement in speed and memory efficiency,
being 10%-30% faster and utilizing 25% less memory compared to `map[type]struct{}`. Additionally, Set3 provides the flexibility to
optimize for either space consumption or speed through the RehashToCapacity(newCapacity) function. This level of performance and
adaptability is unattainable with implementations based on `map[type]struct{}`, which is the standard foundation for most set implementations in Go.

The code is derived from [SwissMap](https://github.com/dolthub/swiss) and it implements the "Fast, Efficient, Cache-friendly Hash Table" found in [Abseil](https://abseil.io/blog/20180927-swisstables).
For details on the algorithm see the [CppCon 2017 talk by Matt Kulukundis](https://www.youtube.com/watch?v=ncHmEUmJZf4).
The dependency on x86 assembler for [SSE2/SSE3](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions) instructions has been removed for portability and speed; the code runs faster without SSE and the necessary additional stack frame.
As hash function, Set3 uses the original hash function from `map[type]struct{}` via [dolthub/maphash](https://github.com/dolthub/maphash).

The name "Set3" comes from the fact that this was the 3rd attempt for an optimized datastructure/code-layout to get the best runtime performance.

## Installation

To use the `Set3` package in your Go project, follow these steps:

1. **Initialize a Go module** (if you haven't already):

   ```sh
   go mod init your-module-name
   ```

2. **Add the package**: Simply import the package in your Go code, and Go modules will handle the rest:

   ```go
   import "github.com/TomTonic/Set3"
   ```

3. **Download dependencies**: Run the following command to download the dependencies:

   ```sh
   go mod tidy
   ```

   This will automatically download and install the Set3 package along with any other dependencies.

## Using Set3

The following test case creates two sets and demonstrates some operations. For a full list of operations on the `Set3` type, see [API doc](https://pkg.go.dev/github.com/TomTonic/Set3#Set3).

```go
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
```

## Performance

The following benchmarks have been performed with [v0.4.0](https://github.com/TomTonic/Set3/releases/tag/v0.4.0) to compare `Set3[uint64]` with `map[uint64]struct{}` with the command:

```sh
go test -v -count=1 -run "^(TestSet3Fill|TestNativeMapFill|TestSet3Find|TestNativeMapFind)$" github.com/TomTonic/Set3 -timeout=120m > benchresult.txt
```

(Raw benchmark results are available [in plain text](https://raw.githubusercontent.com/TomTonic/Set3/main/benchresult.txt). Go version 1.23.1, no PGO.
Please note that you have to comment out the instructions to skip the tests first (`t.Skip("...")`). The whole benchmark runs about 45 minutes.)

### Inserting Nodes into an Empty Set

The following chart illustrates the time required to insert random uint64 values into newly allocated sets.
The displayed times encompass the set allocation process.
All sets were allocated with an initial capacity of 21 elements, which is the current default in Set3.
As a result, rehashing occurs—sometimes multiple times—for larger sets.
This effect is clearly visible in the two charts below.

n = 1 ... 300 (step size +1, linear scale)
![Time for Inserting n Random Elements into an Empty Set, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/b2496fb4-2ff8-4539-9e95-748d108df830)

Please note that the memory chart displays the total memory consumption divided by the number of elements in the set.
This effectively represents the memory usage for storing a single element, i.e., 8 bytes.
Additionally, be aware of the lower bound of 10 bytes and the logarithmic scale of the y-axis.

n = 1 ... 300 (step size +1, linear scale)
![Memory required to store an Element in a Set of Size n, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/ba04f5cf-bca1-453b-9f90-e55d9ede58e5)

### Searching Nodes in a Populated Set

The following chart illustrates the time required to determine whether a random value is present in the set.
The test driver maintains a 30% hit ratio, ensuring that 30% of the queried values are contained within the set, while the remaining 70% are not.
The x-axis represents sets of varying sizes, and the y-axis indicates the average time taken to look up a random value in a set of the corresponding size.

n = 1 ... 300 (step size +1, linear scale)
![Time for Searching Random Values in a Set of Size n, 30% Hit Rate, n = 1 ... 300 (step size +1, linear scale)](https://github.com/user-attachments/assets/bf77efc4-fb60-4de4-a65e-087318e3958c)
