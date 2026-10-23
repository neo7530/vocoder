// Golay(23,12), extended Golay(24,12) and Hamming(15,11), hard and soft decision.
// Port of blip25-vocoder src/fec.rs (MIT, (c) Chance Lindsey).
#pragma once

#include "b25/rs.h"

namespace b25::fec {

struct Decoded {
  u16 info;
  u8 errors;  // corrected bit errors; 255 = unrecoverable (24,12 only)
};

u32 golay_23_12_encode(u16 info);
Decoded golay_23_12_decode(u32 codeword);
u32 golay_24_12_encode(u16 info);
Decoded golay_24_12_decode(u32 codeword);
u16 hamming_15_11_encode(u16 info);
Decoded hamming_15_11_decode(u16 codeword);

// Soft decision: one LLR per code bit, MSB first, positive = bit 1. Chase-style
// search over the least reliable positions. Ties in reliability are broken by
// index order (the Rust original uses an unstable sort there).
Decoded golay_23_12_decode_soft(const i8 (&soft)[23]);
Decoded golay_24_12_decode_soft(const i8 (&soft)[24]);
Decoded hamming_15_11_decode_soft(const i8 (&soft)[15]);

}  // namespace b25::fec
