// Wire layer: FEC, PN modulation and dibit interleave for the two codecs, plus
// the bit packing the façade uses. Port of blip25-vocoder halfrate/frame.rs,
// fullrate/frame.rs and fullrate/fec.rs (MIT, (c) Chance Lindsey).
#pragma once

#include <array>

#include "b25/rs.h"

namespace b25::wire {

// ---- half rate (AMBE+2), 4 info vectors, 36 dibits / 72 bits --------------------
namespace half {

constexpr usz DIBITS = 36;
constexpr usz SOFT_BITS = 72;
constexpr std::array<u8, 4> INFO_WIDTHS = {12, 12, 11, 14};

struct Frame {
  std::array<u16, 4> info{};
  std::array<u8, 4> errors{};
  u16 error_total() const;
};

struct SoftCodeVectors {
  i8 c0[24]{};
  i8 c1[23]{};
  i8 c2[11]{};
  i8 c3[14]{};
};

std::array<u32, 4> modulation_masks(u16 u0);
std::array<u32, 4> deinterleave(const std::array<u8, DIBITS>& dibits);
std::array<u8, DIBITS> interleave(const std::array<u32, 4>& codewords);
SoftCodeVectors soft_deinterleave(const std::array<i8, SOFT_BITS>& soft);

Frame decode_code_vectors(const std::array<u32, 4>& c);
Frame decode_code_vectors_soft(SoftCodeVectors c);
Frame decode_frame(const std::array<u8, DIBITS>& dibits);
Frame decode_frame_soft(const std::array<i8, SOFT_BITS>& soft);
std::array<u32, 4> encode_code_vectors(const std::array<u16, 4>& info);
std::array<u8, DIBITS> encode_frame(const std::array<u16, 4>& info);

// Natural packing: the 49 info bits MSB first into 7 bytes, pad bits zero.
std::array<u8, 7> pack_natural(const std::array<u16, 4>& info);
std::array<u16, 4> unpack_natural(const u8* bytes);

// AMBE+2 bit prioritisation: quantised parameters b0..b8 -> info vectors.
constexpr usz PARAM_COUNT = 9;
std::array<u16, 4> prioritize(const std::array<u16, PARAM_COUNT>& b);

}  // namespace half

// ---- full rate (IMBE), 8 info vectors, 72 dibits / 144 bits ----------------------
namespace full {

constexpr usz DIBITS = 72;
constexpr usz SOFT_BITS = 144;
constexpr std::array<u8, 8> INFO_WIDTHS = {12, 12, 12, 12, 11, 11, 11, 7};

struct Frame {
  std::array<u16, 8> info{};
  std::array<u8, 8> errors{};
  u16 error_total() const;
};

std::array<u32, 8> modulation_masks(u16 u0);
std::array<u32, 8> deinterleave(const std::array<u8, DIBITS>& dibits);
std::array<u8, DIBITS> interleave(const std::array<u32, 8>& codewords);

Frame decode_frame(const std::array<u8, DIBITS>& dibits);
Frame decode_frame_soft(const std::array<i8, SOFT_BITS>& soft);
std::array<u8, DIBITS> encode_frame(const std::array<u16, 8>& info);

}  // namespace full

// ---- byte packing ----------------------------------------------------------------
// Dibits, two bits each, MSB first.
template <usz N>
std::array<u8, N> unpack_dibits(const u8* bytes) {
  std::array<u8, N> out{};
  usz bit = 0;
  for (auto& slot : out) {
    u8 d = 0;
    for (int k = 0; k < 2; ++k, ++bit) d = u8((d << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1));
    slot = d;
  }
  return out;
}

template <usz N, usz B>
std::array<u8, B> pack_dibits(const std::array<u8, N>& dibits) {
  std::array<u8, B> out{};
  usz bit = 0;
  for (u8 d : dibits)
    for (int pos = 1; pos >= 0; --pos, ++bit) out[bit / 8] |= u8(((d >> pos) & 1) << (7 - bit % 8));
  return out;
}

// Info fields of the given widths, MSB first, into a zero-initialised buffer.
template <usz N>
void pack_info_bits(const std::array<u16, N>& info, const std::array<u8, N>& widths, u8* out) {
  usz bit = 0;
  for (usz i = 0; i < N; ++i)
    for (int k = widths[i] - 1; k >= 0; --k, ++bit)
      out[bit / 8] |= u8(((info[i] >> k) & 1) << (7 - bit % 8));
}

template <usz N>
std::array<u16, N> unpack_info_bits(const u8* bytes, const std::array<u8, N>& widths) {
  std::array<u16, N> out{};
  usz bit = 0;
  for (usz i = 0; i < N; ++i) {
    u16 v = 0;
    for (int k = 0; k < widths[i]; ++k, ++bit) v = u16((v << 1) | ((bytes[bit / 8] >> (7 - bit % 8)) & 1));
    out[i] = v;
  }
  return out;
}

}  // namespace b25::wire
