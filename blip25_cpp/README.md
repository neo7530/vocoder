# blip25_cpp

Portable C++20 port of the [blip25-vocoder](https://github.com/openBLIP25/blip25-vocoder)
P25 codec: IMBE (full rate) and AMBE+2 (half rate), encoder and decoder, all four
wire rates (7200x4400, 4400x4400, 3600x2450, 2450x2450). No dependencies beyond the
standard library, no exceptions, no allocation in the DSP core.

```cpp
#include "b25/vocoder.h"
b25::Vocoder tx(b25::Rate::HalfRate3600x2450), rx(b25::Rate::HalfRate3600x2450);
std::vector<uint8_t> wire; std::vector<int16_t> pcm;
tx.encode_pcm(frame160, wire);               // 160 samples @ 8 kHz -> 9 bytes
rx.decode_bits(wire.data(), wire.size(), pcm);   // -> 160 samples
```

## Layout

| path | content |
|---|---|
| `src/dsp.cpp decode.cpp encode.cpp engine.cpp` | fixed-point codec core (port of `src/engine/`) |
| `src/*_tables.cpp`, `wire_tables.cpp` | tables, generated from the Rust sources by `tools/` |
| `src/fec.cpp`, `src/wire.cpp` | Golay / Hamming, PN modulation, interleave, bit packing |
| `src/vocoder.cpp` | façade (`Vocoder`), erasure frames, soft decode |
| `include/b25/rs.h` | Rust integer semantics (wrapping ops, casts) the port depends on |

Build: `cmake -S . -B build && cmake --build build` (`-DB25_BOUNDS_CHECK=ON` enables
`_GLIBCXX_ASSERTIONS` bounds checks). The build needs `-fwrapv`; CMake sets it.

## Verification

* `ref_dump_test <ref.b25dump>`: replays 10800 frame checks recorded from the Rust crate
  (payload bits, PCM and every state word of decoder and encoder). 0 mismatches.
* `golden_corpus_test test/data/golden_corpus.b25corp`: the upstream regression corpus,
  36 sources: encoder output at all four rates, decoder output (full and half rate), the
  concealment severity ramp, soft-vs-hard decode. 219 checks, 0 failing, also under
  bounds checking.

The corpus is a regression fixture produced by the Rust crate, so it proves equality
with that crate, not correctness against a hardware reference.

## Not ported

Enhancement post-filter, rate transcoder, live/stream wrappers, `SpectrumProbe`,
parameter read-back. Hard-decision FEC is exact; the soft Chase decoder breaks
reliability ties by index order where the Rust original uses an unstable sort.

## Licence and patents

Derived from blip25-vocoder (MIT, (c) Chance Lindsey): see `LICENSE.upstream-MIT`.
Speech in the corpus is credited in `test/data/ATTRIBUTION.md`. **Read
`PATENT_NOTICE.md`**: the AMBE+/AMBE+2 algorithms are subject to patents (US8359197
runs to 2028); the MIT licence grants no patent rights.
