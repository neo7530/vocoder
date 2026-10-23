// Port of blip25-vocoder src/fec.rs.
#include "b25/fec.h"

#include <algorithm>
#include <array>
#include <vector>

namespace b25::fec {

namespace {

constexpr u32 GOLAY_23_12_GEN[12] = {
    0b100000000000'11000111010, 0b010000000000'01100011101, 0b001000000000'11110110100, 0b000100000000'01111011010,
    0b000010000000'00111101101, 0b000001000000'11011001100, 0b000000100000'01101100110, 0b000000010000'00110110011,
    0b000000001000'11011100011, 0b000000000100'10101001011, 0b000000000010'10010011111, 0b000000000001'10001110101,
};

inline int popcount32(u32 x) { return __builtin_popcount(x); }

u16 golay_23_12_syndrome(u32 codeword) {
  u32 cw = codeword & 0x7F'FFFF;
  u32 info = (cw >> 11) & 0xFFF;
  u32 received_parity = cw & 0x7FF;
  u32 reconstructed = 0;
  for (int i = 0; i < 12; i++) {
    if ((info >> (11 - i)) & 1) reconstructed ^= GOLAY_23_12_GEN[i] & 0x7FF;
  }
  return static_cast<u16>(received_parity ^ reconstructed);
}

// Syndrome -> error pattern for every error of weight <= 3 (the code is perfect).
const std::array<u32, 2048>& golay_23_12_syndrome_table() {
  static const std::array<u32, 2048> table = [] {
    std::array<u32, 2048> t{};
    for (int a = 0; a < 23; a++) t[golay_23_12_syndrome(1u << a)] = 1u << a;
    for (int a = 0; a < 23; a++)
      for (int b = a + 1; b < 23; b++) {
        u32 pat = (1u << a) | (1u << b);
        t[golay_23_12_syndrome(pat)] = pat;
      }
    for (int a = 0; a < 23; a++)
      for (int b = a + 1; b < 23; b++)
        for (int c = b + 1; c < 23; c++) {
          u32 pat = (1u << a) | (1u << b) | (1u << c);
          t[golay_23_12_syndrome(pat)] = pat;
        }
    return t;
  }();
  return table;
}

constexpr u8 HAMMING_15_11_PARITY[11] = {0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b0111, 0b0110, 0b0101, 0b0011};

template <int N>
std::vector<int> least_reliable_positions(const i8 (&soft)[N], int k) {
  std::array<std::pair<int, int>, N> ranked;
  for (int i = 0; i < N; i++) ranked[i] = {i, std::abs(static_cast<int>(soft[i]))};
  std::stable_sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.second < b.second; });
  std::vector<int> out;
  for (int i = 0; i < std::min(k, N); i++) out.push_back(ranked[i].first);
  return out;
}

template <int N>
u32 soft_to_codeword(const i8 (&soft)[N]) {
  u32 cw = 0;
  for (int i = 0; i < N; i++) cw = (cw << 1) | (soft[i] > 0 ? 1u : 0u);
  return cw;
}

template <int N>
u32 soft_distance(const i8 (&soft)[N], u32 candidate) {
  u32 dist = 0;
  for (int i = 0; i < N; i++) {
    u32 cb = (candidate >> (N - 1 - i)) & 1;
    u32 hard = soft[i] > 0 ? 1u : 0u;
    if (cb != hard) dist += static_cast<u32>(std::abs(static_cast<int>(soft[i])));
  }
  return dist;
}

// Generic Chase decoder: flips every subset (<= T flips) of the `weak` positions,
// hard-decodes each, and keeps the decode whose re-encoded codeword is closest
// to the soft input.
template <int N, int T, class HardDecode>
Decoded chase_decode(const i8 (&soft)[N], u32 hard_cw, const std::vector<int>& weak, HardDecode hard_decode) {
  bool have = false;
  Decoded best{};
  u32 best_dist = 0;
  u32 num_patterns = 1u << weak.size();
  for (u32 pattern = 0; pattern < num_patterns; pattern++) {
    if (popcount32(pattern) > T) continue;
    u32 candidate = hard_cw;
    for (usz bit_idx = 0; bit_idx < weak.size(); bit_idx++) {
      if ((pattern >> bit_idx) & 1) candidate ^= 1u << (N - 1 - weak[bit_idx]);
    }
    auto [decoded, reencoded] = hard_decode(candidate);
    u32 dist = soft_distance<N>(soft, reencoded);
    if (!have || dist < best_dist) {
      have = true;
      best = decoded;
      best_dist = dist;
    }
  }
  return best;
}

constexpr int CHASE_GOLAY_DEPTH = 6;
constexpr int CHASE_HAMMING_DEPTH = 3;

}  // namespace

u32 golay_23_12_encode(u16 info) {
  u32 cw = 0;
  u32 in = static_cast<u32>(info) & 0xFFF;
  for (int i = 0; i < 12; i++) {
    if ((in >> (11 - i)) & 1) cw ^= GOLAY_23_12_GEN[i];
  }
  return cw;
}

Decoded golay_23_12_decode(u32 codeword) {
  u32 cw = codeword & 0x7F'FFFF;
  u16 syndrome = golay_23_12_syndrome(cw);
  u32 pattern = golay_23_12_syndrome_table()[syndrome];
  u32 corrected = cw ^ pattern;
  return {static_cast<u16>((corrected >> 11) & 0xFFF), static_cast<u8>(popcount32(pattern))};
}

u32 golay_24_12_encode(u16 info) {
  u32 cw23 = golay_23_12_encode(info);
  u32 parity = popcount32(cw23) & 1;
  return (cw23 << 1) | parity;
}

Decoded golay_24_12_decode(u32 codeword) {
  u32 cw = codeword & 0xFF'FFFF;
  u32 cw23 = (cw >> 1) & 0x7F'FFFF;
  u32 received_parity = cw & 1;
  Decoded d23 = golay_23_12_decode(cw23);
  u32 corrected23 = golay_23_12_encode(d23.info);
  u32 reconstructed_parity = popcount32(corrected23) & 1;
  bool parity_matches = reconstructed_parity == received_parity;
  u8 errors;
  if (d23.errors == 3 && !parity_matches) errors = 255;  // 4 errors: detected, not correctable
  else if (parity_matches) errors = d23.errors;
  else errors = static_cast<u8>(d23.errors + 1);
  return {d23.info, errors};
}

u16 hamming_15_11_encode(u16 info) {
  info &= 0x7FF;
  u16 parity = 0;
  for (int i = 0; i < 11; i++) {
    if ((info >> (10 - i)) & 1) parity ^= HAMMING_15_11_PARITY[i];
  }
  return static_cast<u16>((info << 4) | (parity & 0xF));
}

Decoded hamming_15_11_decode(u16 codeword) {
  u16 cw = codeword & 0x7FFF;
  u16 info = (cw >> 4) & 0x7FF;
  u16 received_parity = cw & 0xF;
  u16 reconstructed = 0;
  for (int i = 0; i < 11; i++) {
    if ((info >> (10 - i)) & 1) reconstructed ^= HAMMING_15_11_PARITY[i];
  }
  u16 syndrome = (received_parity ^ reconstructed) & 0xF;
  if (syndrome == 0) return {info, 0};
  for (int i = 0; i < 11; i++) {
    if (HAMMING_15_11_PARITY[i] == syndrome) return {static_cast<u16>(info ^ (1u << (10 - i))), 1};
  }
  return {info, 1};  // the error is in a parity bit
}

Decoded golay_23_12_decode_soft(const i8 (&soft)[23]) {
  auto weak = least_reliable_positions<23>(soft, CHASE_GOLAY_DEPTH);
  u32 hard_cw = soft_to_codeword<23>(soft);
  return chase_decode<23, 3>(soft, hard_cw, weak, [](u32 cw) {
    Decoded d = golay_23_12_decode(cw);
    return std::pair<Decoded, u32>{d, golay_23_12_encode(d.info)};
  });
}

Decoded golay_24_12_decode_soft(const i8 (&soft)[24]) {
  auto weak = least_reliable_positions<24>(soft, CHASE_GOLAY_DEPTH);
  u32 hard_cw = soft_to_codeword<24>(soft);
  return chase_decode<24, 3>(soft, hard_cw, weak, [](u32 cw) {
    Decoded d = golay_24_12_decode(cw);
    return std::pair<Decoded, u32>{d, golay_24_12_encode(d.info)};
  });
}

Decoded hamming_15_11_decode_soft(const i8 (&soft)[15]) {
  auto weak = least_reliable_positions<15>(soft, CHASE_HAMMING_DEPTH);
  u32 hard_cw = soft_to_codeword<15>(soft);
  return chase_decode<15, 1>(soft, hard_cw, weak, [](u32 cw) {
    Decoded d = hamming_15_11_decode(static_cast<u16>(cw));
    return std::pair<Decoded, u32>{d, hamming_15_11_encode(d.info)};
  });
}

}  // namespace b25::fec
