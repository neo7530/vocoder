#include "b25/wire.h"

#include <algorithm>

#include "b25/fec.h"
#include "b25/wire_tables.h"

namespace b25::wire {

namespace {

// Linear congruential PN sequence shared by both codecs (u0 seeds it).
template <usz N>
std::array<u16, N> pn_sequence(u16 u0) {
  u0 &= 0x0FFF;
  std::array<u16, N> pr{};
  pr[0] = u16(u0 * 16u);
  for (usz n = 1; n < N; ++n) pr[n] = u16((173u * u32(pr[n - 1]) + 13849u) & 0xFFFFu);
  return pr;
}

template <usz W>
void soft_demodulate(i8 (&soft)[W], u32 mask) {
  for (usz i = 0; i < W; ++i)
    if ((mask >> (W - 1 - i)) & 1) soft[i] = soft[i] == -128 ? i8(127) : i8(-soft[i]);
}

template <usz N, usz D>
std::array<u32, N> deinterleave_with(const wire::InterleaveEntry (&tab)[D], const std::array<u8, D>& dibits) {
  std::array<u32, N> c{};
  for (usz sym = 0; sym < D; ++sym) {
    const auto& e = tab[sym];
    c[e.bit1_vec] |= u32((dibits[sym] >> 1) & 1) << e.bit1_idx;
    c[e.bit0_vec] |= u32(dibits[sym] & 1) << e.bit0_idx;
  }
  return c;
}

template <usz N, usz D>
std::array<u8, D> interleave_with(const wire::InterleaveEntry (&tab)[D], const std::array<u32, N>& cw) {
  std::array<u8, D> d{};
  for (usz sym = 0; sym < D; ++sym) {
    const auto& e = tab[sym];
    u8 hi = u8((cw[e.bit1_vec] >> e.bit1_idx) & 1);
    u8 lo = u8((cw[e.bit0_vec] >> e.bit0_idx) & 1);
    d[sym] = u8((hi << 1) | lo);
  }
  return d;
}

}  // namespace

// =============================== half rate ========================================
namespace half {

u16 Frame::error_total() const {
  u16 t = 0;
  for (u8 e : errors) t = u16(t + e);
  return t;
}

std::array<u32, 4> modulation_masks(u16 u0) {
  auto pr = pn_sequence<24>(u0);
  u32 m = 0;
  for (int k = 0; k < 23; ++k) m |= u32(pr[1 + k] >> 15) << (22 - k);
  return {0, m, 0, 0};
}

std::array<u32, 4> deinterleave(const std::array<u8, DIBITS>& dibits) {
  return deinterleave_with<4, DIBITS>(ANNEX_S, dibits);
}

std::array<u8, DIBITS> interleave(const std::array<u32, 4>& cw) { return interleave_with<4, DIBITS>(ANNEX_S, cw); }

SoftCodeVectors soft_deinterleave(const std::array<i8, SOFT_BITS>& soft) {
  SoftCodeVectors out;
  auto place = [&](u8 vec, u8 idx, i8 s) {
    switch (vec) {
      case 0: out.c0[24 - 1 - idx] = s; break;
      case 1: out.c1[23 - 1 - idx] = s; break;
      case 2: out.c2[11 - 1 - idx] = s; break;
      default: out.c3[14 - 1 - idx] = s; break;
    }
  };
  for (usz sym = 0; sym < DIBITS; ++sym) {
    const auto& e = ANNEX_S[sym];
    place(e.bit1_vec, e.bit1_idx, soft[2 * sym]);
    place(e.bit0_vec, e.bit0_idx, soft[2 * sym + 1]);
  }
  return out;
}

Frame decode_code_vectors(const std::array<u32, 4>& c) {
  auto d0 = fec::golay_24_12_decode(c[0]);
  auto masks = modulation_masks(d0.info);
  auto d1 = fec::golay_23_12_decode(c[1] ^ masks[1]);
  Frame f;
  f.info = {d0.info, d1.info, u16(c[2] & 0x7FF), u16(c[3] & 0x3FFF)};
  f.errors = {d0.errors, d1.errors, 0, 0};
  return f;
}

Frame decode_code_vectors_soft(SoftCodeVectors c) {
  auto d0 = fec::golay_24_12_decode_soft(c.c0);
  auto masks = modulation_masks(d0.info);
  soft_demodulate(c.c1, masks[1]);
  auto d1 = fec::golay_23_12_decode_soft(c.c1);
  u16 u2 = 0, u3 = 0;
  for (int i = 0; i < 11; ++i)
    if (c.c2[i] > 0) u2 |= u16(1u << (11 - 1 - i));
  for (int i = 0; i < 14; ++i)
    if (c.c3[i] > 0) u3 |= u16(1u << (14 - 1 - i));
  Frame f;
  f.info = {d0.info, d1.info, u2, u3};
  f.errors = {d0.errors, d1.errors, 0, 0};
  return f;
}

Frame decode_frame(const std::array<u8, DIBITS>& dibits) { return decode_code_vectors(deinterleave(dibits)); }
Frame decode_frame_soft(const std::array<i8, SOFT_BITS>& soft) {
  return decode_code_vectors_soft(soft_deinterleave(soft));
}

std::array<u32, 4> encode_code_vectors(const std::array<u16, 4>& info) {
  std::array<u16, 4> u{};
  for (usz i = 0; i < 4; ++i) u[i] = u16(info[i] & ((1u << INFO_WIDTHS[i]) - 1));
  auto masks = modulation_masks(u[0]);
  return {fec::golay_24_12_encode(u[0]), fec::golay_23_12_encode(u[1]) ^ masks[1], u32(u[2]), u32(u[3])};
}

std::array<u8, DIBITS> encode_frame(const std::array<u16, 4>& info) {
  return interleave(encode_code_vectors(info));
}

std::array<u8, 7> pack_natural(const std::array<u16, 4>& info) {
  std::array<u8, 7> out{};
  pack_info_bits(info, INFO_WIDTHS, out.data());
  return out;
}

std::array<u16, 4> unpack_natural(const u8* bytes) { return unpack_info_bits<4>(bytes, INFO_WIDTHS); }

std::array<u16, 4> prioritize(const std::array<u16, PARAM_COUNT>& b) {
  std::array<u16, 4> u{};
  for (const auto& m : AMBE_BIT_MAP) {
    u16 bit = (b[m.src_param] >> m.src_bit) & 1;
    u[m.dst_vec] |= u16(bit << m.dst_bit);
  }
  return u;
}

}  // namespace half

// =============================== full rate ========================================
namespace full {

u16 Frame::error_total() const {
  u16 t = 0;
  for (u8 e : errors) t = u16(t + e);
  return t;
}

std::array<u32, 8> modulation_masks(u16 u0) {
  auto pr = pn_sequence<115>(u0);
  struct L { usz vec, start, len; };
  static constexpr L LAYOUT[6] = {{1, 1, 23}, {2, 24, 23}, {3, 47, 23}, {4, 70, 15}, {5, 85, 15}, {6, 100, 15}};
  std::array<u32, 8> masks{};
  for (const auto& l : LAYOUT) {
    u32 m = 0;
    for (usz k = 0; k < l.len; ++k) m |= u32(pr[l.start + k] >> 15) << (l.len - 1 - k);
    masks[l.vec] = m;
  }
  return masks;
}

std::array<u32, 8> deinterleave(const std::array<u8, DIBITS>& dibits) {
  return deinterleave_with<8, DIBITS>(ANNEX_H, dibits);
}

std::array<u8, DIBITS> interleave(const std::array<u32, 8>& cw) { return interleave_with<8, DIBITS>(ANNEX_H, cw); }

namespace {
constexpr u8 VECTOR_LENGTHS[8] = {23, 23, 23, 23, 15, 15, 15, 7};
struct Soft {
  i8 golay[4][23]{};
  i8 hamming[3][15]{};
  i8 uncoded[7]{};
};
Soft soft_deinterleave(const std::array<i8, SOFT_BITS>& soft) {
  Soft out;
  auto place = [&](u8 vec, u8 idx, i8 s) {
    usz pos = VECTOR_LENGTHS[vec] - 1u - idx;
    if (vec <= 3) out.golay[vec][pos] = s;
    else if (vec <= 6) out.hamming[vec - 4][pos] = s;
    else out.uncoded[pos] = s;
  };
  for (usz sym = 0; sym < DIBITS; ++sym) {
    const auto& e = ANNEX_H[sym];
    place(e.bit1_vec, e.bit1_idx, soft[2 * sym]);
    place(e.bit0_vec, e.bit0_idx, soft[2 * sym + 1]);
  }
  return out;
}
}  // namespace

Frame decode_frame(const std::array<u8, DIBITS>& dibits) {
  auto c = deinterleave(dibits);
  auto d0 = fec::golay_23_12_decode(c[0]);
  auto m = modulation_masks(d0.info);
  auto d1 = fec::golay_23_12_decode(c[1] ^ m[1]);
  auto d2 = fec::golay_23_12_decode(c[2] ^ m[2]);
  auto d3 = fec::golay_23_12_decode(c[3] ^ m[3]);
  auto d4 = fec::hamming_15_11_decode(u16(c[4] ^ m[4]));
  auto d5 = fec::hamming_15_11_decode(u16(c[5] ^ m[5]));
  auto d6 = fec::hamming_15_11_decode(u16(c[6] ^ m[6]));
  Frame f;
  f.info = {d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, u16(c[7] & 0x7F)};
  f.errors = {d0.errors, d1.errors, d2.errors, d3.errors, d4.errors, d5.errors, d6.errors, 0};
  return f;
}

Frame decode_frame_soft(const std::array<i8, SOFT_BITS>& soft) {
  Soft c = soft_deinterleave(soft);
  auto d0 = fec::golay_23_12_decode_soft(c.golay[0]);
  auto m = modulation_masks(d0.info);
  soft_demodulate(c.golay[1], m[1]);
  soft_demodulate(c.golay[2], m[2]);
  soft_demodulate(c.golay[3], m[3]);
  soft_demodulate(c.hamming[0], m[4]);
  soft_demodulate(c.hamming[1], m[5]);
  soft_demodulate(c.hamming[2], m[6]);
  auto d1 = fec::golay_23_12_decode_soft(c.golay[1]);
  auto d2 = fec::golay_23_12_decode_soft(c.golay[2]);
  auto d3 = fec::golay_23_12_decode_soft(c.golay[3]);
  auto d4 = fec::hamming_15_11_decode_soft(c.hamming[0]);
  auto d5 = fec::hamming_15_11_decode_soft(c.hamming[1]);
  auto d6 = fec::hamming_15_11_decode_soft(c.hamming[2]);
  u16 u7 = 0;
  for (int i = 0; i < 7; ++i)
    if (c.uncoded[i] > 0) u7 |= u16(1u << (7 - 1 - i));
  Frame f;
  f.info = {d0.info, d1.info, d2.info, d3.info, d4.info, d5.info, d6.info, u7};
  f.errors = {d0.errors, d1.errors, d2.errors, d3.errors, d4.errors, d5.errors, d6.errors, 0};
  return f;
}

std::array<u8, DIBITS> encode_frame(const std::array<u16, 8>& info) {
  std::array<u16, 8> u{};
  for (usz i = 0; i < 8; ++i) u[i] = u16(info[i] & ((1u << INFO_WIDTHS[i]) - 1));
  auto m = modulation_masks(u[0]);
  std::array<u32, 8> c = {
      fec::golay_23_12_encode(u[0]),
      fec::golay_23_12_encode(u[1]) ^ m[1],
      fec::golay_23_12_encode(u[2]) ^ m[2],
      fec::golay_23_12_encode(u[3]) ^ m[3],
      u32(fec::hamming_15_11_encode(u[4])) ^ m[4],
      u32(fec::hamming_15_11_encode(u[5])) ^ m[5],
      u32(fec::hamming_15_11_encode(u[6])) ^ m[6],
      u32(u[7]),
  };
  return interleave(c);
}

}  // namespace full
}  // namespace b25::wire
