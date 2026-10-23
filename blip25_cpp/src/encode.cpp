// Port of blip25-vocoder src/engine/encode.rs (MIT, (c) Chance Lindsey):
// analysis, voicing/pitch decisions, quantisation and bit packing. Function
// names and order follow the Rust source; the long per-function documentation
// lives there. Everything is in b25::eng::enc so that local helpers shadow the
// shared dsp/decode ones exactly like module-local items do in Rust.
#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <tuple>

#include "b25/dsp.h"
#include "b25/decode.h"
#include "b25/dsp_tables.h"
#include "b25/encode.h"
#include "b25/encode_tables.h"
#include "b25/engine_tables.h"

namespace b25::eng::enc {

using namespace rs;

// Snapshot helpers: copy `src` into `buf` and return the copy (the Rust code
// uses them to read a block while writing to the buffer it came from).
static CS16 snap(S16 buf, CS16 src) {
  usz n = src.size();
  std::copy_n(src.begin(), n, buf.begin());
  return CS16(buf.first(n));
}
static std::span<const i32> snap32(std::span<i32> buf, std::span<const i32> src) {
  usz n = src.size();
  std::copy_n(src.begin(), n, buf.begin());
  return std::span<const i32>(buf.first(n));
}

static void normalise_to_exp_inplace(S16 v, u16 n, i16& exp) {
  u16 e = static_cast<u16>(norm_exp_absmax(v, n));
  i16 sh = static_cast<i16>(wneg<u16>(e));
  if (sh != 0 && static_cast<i16>(n) > 0) {
    u32 b;
    if (sh < 0) b = sh < -0x1e ? 0x1fu : (static_cast<u32>(static_cast<u8>(wneg<i16>(sh))) & 0x1f);
    else b = 0;
    for (usz i = 0; i < n; i++) {
      i32 w = static_cast<i32>(v[i]) << 16;
      w = sh < 0 ? (w >> b) : wshl<i32>(w, static_cast<u32>(static_cast<i32>(sh)) & 0x1f);
      v[i] = static_cast<i16>(w >> 16);
    }
  }
  exp = wadd<i16>(exp, static_cast<i16>(e));
}

static i32 bsr_keep(u32 v) { return v == 0 ? 0 : 31 - __builtin_clz(v); }

static i32 norm_exp(i32 x) {
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  return 0x1e - bsr_keep(m);
}

static i32 shift32_bL2b1(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  u32 abs_s = static_cast<u32>(s < 0 ? -static_cast<i32>(s) : static_cast<i32>(s));
  return v >> (abs_s & 0x1f);
}
static i16 high16_bL2b1(i32 v) { return static_cast<i16>(v >> 16); }
static i32 up16_bL2b1(i16 v) { return static_cast<i32>(v) << 16; }
static i64 allshl_bL2b1(i64 v, u8 count) { return allshl_al2b3(v, count); }  // identical code in the original
static i32 mul32x16_bL2b1(i32 v, i32 w) {
  i32 lo = wmul<i32>(static_cast<u16>(v), w) >> 15;
  i32 hi = wmul<i32>(wmul<i32>(static_cast<i16>(v >> 16), w), 2);
  return wadd<i32>(lo, hi);
}

i32 atan2_bfp(i16 y, i16 y_exp, i16 x, i16 x_exp, i16& out_exp) {
  u32 quad = 0;
  if (x < 0) quad |= 2;
  if (y < 0) quad |= 1;
  if (x == 0) {
    if (quad & 1) {
      out_exp = 1;
      return static_cast<i32>(0xffff9b78u);  // -pi/2
    }
    if (y != 0) {
      out_exp = 1;
      return 0x6488;  // +pi/2
    }
    out_exp = 0;
    return 0;
  }
  if (y == 0) {
    if (quad & 2) {
      out_exp = 2;
      return 0x6488;  // pi
    }
    out_exp = 0;
    return 0;
  }
  i16 off;
  i32 off_exp;
  switch (quad) {
    case 2: off = 0x6488; off_exp = 2; break;
    case 3: off = -0x6488; off_exp = 2; break;
    default: off = 0; off_exp = 0; break;
  }
  i32 xw = up16_bL2b1(x);
  u32 sh_x = static_cast<u16>(norm_exp(xw));
  i32 xe = wsub<i32>(x_exp, static_cast<i32>(sh_x));
  i16 xm = high16_bL2b1(static_cast<i32>(static_cast<u32>(xw) << (sh_x & 0x1f)));
  i32 yw = up16_bL2b1(y);
  u32 sh_y = static_cast<u16>(norm_exp(yw));
  i32 num = static_cast<i32>(static_cast<u32>(yw) << (sh_y & 0x1f));
  i16 e = 0;
  i16 q = div_norm(num, static_cast<i16>(wsub<i32>(y_exp, static_cast<i32>(sh_y))), xm, static_cast<i16>(xe), e);
  i16 a = atan_bfp(q, e, e);
  i16 cnt = wsub<i16>(wsub<i16>(e, static_cast<i16>(off_exp)), 1);
  i32 v = wadd<i32>(shift32_bL2b1(up16_bL2b1(a), cnt), up16_bL2b1(off) >> 1);
  u16 sh = static_cast<u16>(v == 0 ? 0 : norm_exp(v));
  out_exp = wadd<i16>(wsub<i16>(static_cast<i16>(off_exp), static_cast<i16>(sh)), 1);
  return static_cast<u16>(static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(sh) & 0x1f)) >> 16);
}

void harmonic_log_mags(S16 out, i16 step, u16 n, CS16 spec, i16 spec_exp, i16 offset) {
  i32 inc = (static_cast<i32>(step) << 9) >> 4;
  i32 acc0 = wadd<i32>(inc >> 1, 0x8000);
  if (static_cast<i16>(n) <= 0) return;
  u32 count = n;
  u16 pos_hi = static_cast<u16>(high16_bL2b1(acc0));
  u16 pos_lo = static_cast<u16>(acc0);
  usz oi = 0;
  for (;;) {
    u32 base = static_cast<u32>(up16_bL2b1(static_cast<i16>(pos_hi))) | static_cast<u32>(pos_lo);
    i32 acc = static_cast<i32>(wadd<u32>(base, static_cast<u32>(inc)));
    u16 raw_hi = static_cast<u16>(high16_bL2b1(acc));
    u16 next_hi = 0x7f < static_cast<i16>(raw_hi) ? static_cast<u16>(0x7f) : raw_hi;
    i32 w0 = static_cast<i32>(static_cast<u32>(pos_lo) >> 1);
    u16 span = wsub<u16>(next_hi, pos_hi);
    i32 idx = static_cast<i16>(pos_hi);
    i32 v0 = rd32(spec, 2 * static_cast<usz>(idx));
    i32 p0 = mul32x16_bL2b1(v0, w0);
    p0 = p0 == INT32_MIN ? INT32_MAX : wneg<i32>(p0);
    i32 first = wadd<i32>(p0, v0);
    i32 w1 = static_cast<i16>((static_cast<u32>(acc) >> 1) & 0x7fff);
    i32 v1 = rd32(spec, 2 * static_cast<usz>(idx + static_cast<i32>(static_cast<i16>(span))));
    i32 last = mul32x16_bL2b1(v1, w1);
    i64 acc64 = wadd<i64>(static_cast<i64>(first), static_cast<i64>(last));
    if (static_cast<i16>(span) > 1) {
      u16 whole = wsub<u16>(span, 1);
      for (usz k = 0; k < whole; k++) acc64 = wadd<i64>(acc64, static_cast<i64>(rd32(spec, 2 * (static_cast<usz>(idx) + 1 + k))));
    }
    i16 s = static_cast<i16>(norm64(static_cast<u32>(acc64), static_cast<u32>(acc64 >> 32)));
    i32 shifted = static_cast<i32>(s < 0 ? allshr(acc64, static_cast<u8>(static_cast<u32>(wneg<i32>(s)) & 0xff))
                                         : allshl_bL2b1(acc64, static_cast<u8>(static_cast<u32>(static_cast<i32>(s)) & 0xff)));
    u16 d = wsub<u16>(static_cast<u16>(spec_exp), static_cast<u16>(s));
    i16 mant = high16_bL2b1(shifted);
    if (static_cast<i16>(d) < -0x3b || mant == 0) out[oi] = -0x7800;
    else out[oi] = wadd<i16>(static_cast<i16>(log2_at_q(mant, static_cast<i16>(wadd<u16>(d, 0x1e)), 5)), offset);
    oi++;
    count--;
    if (count == 0) break;
    pos_hi = next_hi;
    pos_lo = static_cast<u16>(acc);
  }
}

u16 quantise_voicing_bL2b1(u32& out0, u32& out1, i32 prev_bits, CS16 w_a, CS16 w_b, CS16 w_c, i32 nbits) {
  u32 prev = static_cast<u16>(gather_dibits(prev_bits));
  u32 book[312] = {};
  build_voicing_table(std::span<u32>(book), prev, nbits);
  u16 idx = static_cast<u16>(voicing_codebook_search(w_a, w_b, w_c, std::span<const u32>(book), static_cast<u8>(nbits), 0));
  u32 pat = voicing_pattern(static_cast<i16>(idx), prev, static_cast<i16>(nbits));
  expand_dibits(out0, out1, pat);
  return idx;
}

i32 log_amp_clamped(i16 mant, i32 exp, i16 offset) {
  if (static_cast<i16>(exp) >= -0x3b && mant != 0) {
    i16 l = static_cast<i16>(log2_at_q(mant, wadd<i16>(static_cast<i16>(exp), 0x1e), 5));
    i16 v = wadd<i16>(l, offset);
    if (v > 0x7800) return 0x7800;
    if (v >= -0x1400) return v;
  }
  return -0x1400;
}

i32 pitch_gate_score(std::span<const i32> spec, i16 pitch, i16 sh) {
  i16 e1 = 0;
  i32 start = 0;
  i16 m1 = band_energy_walk(e1, spec, 0x10000, pitch, sh, start);
  i32 back = shift32_bL2b1(wmul<i32>(pitch, 0x6000), wsub<i16>(sh, 7));
  u16 idx1 = static_cast<u16>(high16_bL2b1(wsub<i32>(start, back)));
  i16 e2 = 0;
  i16 m2 = sum_mant_exp(spec.subspan(static_cast<usz>(static_cast<i16>(idx1))), static_cast<i16>(0x80 - static_cast<i32>(idx1)), e2);
  i32 lhs = shift32_bL2b1(up16_bL2b1(m1), wsub<i16>(e1, e2));
  if (wsub<i32>(lhs, wmul<i32>(m2, 0xe8ba)) <= 0) return -1;
  u16 idx2 = static_cast<u16>(high16_bL2b1(wadd<i32>(shift32_bL2b1(up16_bL2b1(pitch), wsub<i16>(sh, 8)), 0x10000)));
  i16 e3 = 0;
  i16 m3 = sum_mant_exp(spec.subspan(static_cast<usz>(static_cast<i16>(idx2))), static_cast<i16>(0x80 - static_cast<i32>(idx2)), e3);
  i32 lhs2 = shift32_bL2b1(up16_bL2b1(m2), wsub<i16>(e2, e3));
  if (wsub<i32>(lhs2, wmul<i32>(m3, 0x1998)) <= 0) return -1;
  i16 e4 = 0;
  i32 start2 = 0;
  i16 m4 = band_energy_walk(e4, spec, pitch, pitch, sh, start2);
  return one_minus_ratio(m4, e4, m3, e3);
}

static i32 read_i32(CS16 state, usz byte_off) {
  usz i = byte_off / 2;
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(state[i])) | (static_cast<u32>(static_cast<i32>(state[i + 1])) << 16));
}
static void write_i32(S16 state, usz byte_off, i32 v) {
  usz i = byte_off / 2;
  state[i] = static_cast<i16>(v);
  state[i + 1] = static_cast<i16>(v >> 16);
}

i32 band_level_floor(S16 out, S16 state) {
  constexpr usz MANT = 0x52 / 2;
  constexpr usz EXPS = 0x72 / 2;
  i32 total = 0;
  i16 tilt = -0x400;
  for (usz k = 0; k < 16; k++) {
    i32 v = shift32_bL2b1(up16_bL2b1(state[MANT + k]), wsub<i16>(state[EXPS + k], 5));
    v = wadd<i32>(v, static_cast<i32>(0x78000000u));
    i16 level = high16_bL2b1(v);
    u16 t = static_cast<u16>(high16_bL2b1(wsub<i32>(v, up16_bL2b1(tilt))));
    if (static_cast<i16>(t) < 0x66) t = 0x66;
    i16 l = static_cast<i16>(log2_at_q(static_cast<i16>(t), 5, 5));
    i16 corr = high16_bL2b1(wsub<i32>(up16_bL2b1(l), 0x21c0001));
    if (corr < 0) {
      i32 a = up16_bL2b1(corr);
      i32 b = up16_bL2b1(level);
      i32 s = wadd<i32>(a, b);
      if (((a ^ s) & (s ^ b)) < 0) s = static_cast<i32>(wadd<u32>(0x7fffffffu, b < 0 ? 1u : 0u));
      level = high16_bL2b1(s);
    }
    out[k] = level;
    total = wadd<i32>(total, level);
    tilt = high16_bL2b1(wsub<i32>(up16_bL2b1(tilt), 0x3f0001));
  }
  i32 margin = total > 0x10000 ? 0x2000 : 0x4000;
  i32 m = static_cast<i16>(margin);
  i32 hyst = wsub<i32>(m << 4, m);  // 15*margin
  i32 prev = read_i32(state, 0x40);
  i32 d = wshl<i32>(wsub<i32>(total, prev), 6);
  if (wadd<i32>(d, wmul<i32>(hyst, 2)) < 0) {
    write_i32(state, 0x40, total);
    return 2;
  }
  return 0;
}

static i32 sat_add_batch1(i32 a, i32 b) {
  i32 s = wadd<i32>(a, b);
  if (((s ^ a) & (s ^ b)) < 0) return a < 0 ? INT32_MIN : INT32_MAX;
  return s;
}

static i16 nrm30(i32 x) {
  if (x == 0) return 0;
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  i32 b = m == 0 ? 0 : 31 - __builtin_clz(m);
  return static_cast<i16>(0x1e - b);
}
static i32 shift32_bL2b2(i32 v, i16 s) {
  if (s >= 0) return wshl<i32>(v, static_cast<u32>(static_cast<i32>(s)) & 0x1f);
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f);
}
static i16 high16_bL2b2(i32 x) { return static_cast<i16>(x >> 16); }
static i32 up16_bL2b2(i16 x) { return static_cast<i32>(x) << 16; }
static u16 shl16_bL2b2(u16 x, u16 c) {
  u32 cc = c & 0x1f;
  return cc >= 16 ? static_cast<u16>(0) : static_cast<u16>(x << cc);
}
static i32 get32_bL2b2(CS16 s, usz w) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(s[w])) | (static_cast<u32>(static_cast<i32>(s[w + 1])) << 16));
}
static void put32_bL2b2(S16 s, usz w, i32 v) {
  s[w] = static_cast<i16>(v);
  s[w + 1] = static_cast<i16>(v >> 16);
}

void frame_band_voicing(S16 bands, S16 mid, CS16 spec, S16 out, i16 mode) {
  interp_mid(mid.subspan(5), mid.first(5));
  i16 exp = 0;
  i32 exps = rd32(spec, 2 * 387);
  i16 ratio = ratio_of_block_sums(spec.subspan(2 * 129), static_cast<u16>(exps >> 16), spec, static_cast<i16>(exps), exp);
  bands[0x51] = wadd<i16>(bands[0x50], exp);
  i16 a[8], c[8], d[8], f[8];
  std::copy_n(bands.begin() + 0, 8, a);
  std::copy_n(bands.begin() + 16, 8, c);
  std::copy_n(bands.begin() + 40, 8, d);
  std::copy_n(bands.begin() + 56, 8, f);
  i16 metric[8], weights[8];
  std::copy_n(bands.begin() + 8, 8, metric);
  std::copy_n(bands.begin() + 48, 8, weights);
  band_ratio_weights(CS16(a), S16(metric), CS16(c), CS16(d), S16(weights), CS16(f), static_cast<u16>(bands[0x50]),
                     static_cast<u16>(bands[0x52]), ratio);
  std::copy_n(metric, 8, bands.begin() + 8);
  std::copy_n(weights, 8, bands.begin() + 48);
  i16 src[8], levels[8];
  std::copy_n(out.begin() + 0x0c, 8, src);
  std::copy_n(out.begin() + 0x24, 8, levels);
  i16 cur[8], alt[8];
  std::copy_n(out.begin() + 0x14, 8, cur);
  std::copy_n(out.begin() + 0x1c, 8, alt);
  band_floor(mid, CS16(src), S16(cur), S16(alt), CS16(levels), mode);
  std::copy_n(cur, 8, out.begin() + 0x14);
  std::copy_n(alt, 8, out.begin() + 0x1c);
}

void log_band_levels(S16 out, CS16 src, i16 exp, i16 tilt) {
  for (usz i = 0; i < 8; i++) {
    i16 x = src[i];
    if (x == 0) {
      out[i] = 0x7fff;
      continue;
    }
    i16 l = static_cast<i16>(log2_at_q(x, exp, 6));
    i32 acc = wshl<i32>(wsub<i32>(-0x4a3, l), 16);
    if (tilt != 0 && static_cast<i16>(i) > 3) acc = wsub<i32>(acc, 0x0200'0000);
    if (acc < 0) acc = 0;
    out[i] = high16_bL2b2(shl_sat(acc, 5));
  }
}

u16 band_spectral_match(CS16 meas, i16 meas_exp, CS16 reference, i16 ref_exp, S16 out) {
  i16 err[8] = {}, err_exp[8] = {}, energy[8] = {}, energy_exp[8] = {};
  u32 lo = 0;
  u32 hi = 0xd;
  for (usz i = 0; i < 8; i++) {
    if (i == 7) hi = 0x7f;
    usz base = static_cast<usz>(static_cast<isz>(static_cast<i32>(static_cast<i16>(lo)) * 4));  // byte offset
    usz word = base / 2;
    i16 n = static_cast<i16>(wsub<u32>(hi, lo));
    err[i] = sq_diff_norm(err_exp[i], reference.subspan(word), ref_exp, meas.subspan(word), meas_exp, n);
    i16 e = 0;
    i32 dd = dot_norm64(e, meas.subspan(word), meas.subspan(word), wmul<u16>(static_cast<u16>(n), 2));
    energy_exp[i] = wadd<i16>(e, wmul<i16>(meas_exp, 2));
    energy[i] = high16_bL2b2(dd);
    lo = hi & 0xffff;
    hi = wadd<u32>(hi, 0x10u);
  }
  i16 src_a[8];
  std::copy_n(err, 8, src_a);
  u16 ec = static_cast<u16>(align_block_exp(S16(err), CS16(src_a), CS16(err_exp), 8));
  std::copy_n(energy, 8, src_a);
  u16 en = static_cast<u16>(align_block_exp(S16(energy), CS16(src_a), CS16(energy_exp), 8));
  i16 err_aligned[8], energy_aligned[8];
  std::copy_n(err, 8, err_aligned);
  std::copy_n(energy, 8, energy_aligned);
  i16 ratio_exp[8] = {};
  for (usz i = 0; i < 8; i++) {
    i32 se = err_aligned[i];
    i32 sn = energy_aligned[i];
    if (i > 0) {
      se = wadd<i32>(se, err_aligned[i - 1]);
      sn = wadd<i32>(sn, energy_aligned[i - 1]);
    }
    if (i < 7) {
      se = wadd<i32>(se, err_aligned[i + 1]);
      sn = wadd<i32>(sn, energy_aligned[i + 1]);
    }
    u16 ne = static_cast<u16>(nrm30(se));
    u16 nn = static_cast<u16>(nrm30(sn));
    i32 den = wshl<i32>(sn, static_cast<u32>(nn) & 0x1f);
    if (den == 0) {
      out[0x0c + i] = 0x4000;
      ratio_exp[i] = 1;
    } else {
      i16 e = 0;
      out[0x0c + i] = div_norm(wshl<i32>(se, static_cast<u32>(ne) & 0x1f), static_cast<i16>(wsub<u16>(ec, ne)), high16_bL2b2(den),
                               static_cast<i16>(wsub<u16>(en, nn)), e);
      ratio_exp[i] = e;
    }
  }
  i16 ratios[8];
  std::copy_n(out.begin() + 0x0c, 8, ratios);
  i16 src_r[8];
  std::copy_n(ratios, 8, src_r);
  u16 r = static_cast<u16>(align_block_exp(S16(ratios), CS16(src_r), CS16(ratio_exp), 8));
  std::copy_n(ratios, 8, out.begin() + 0x0c);
  out[9] = static_cast<i16>(en);
  return r;
}

u16 two_group_mask(CS16 cfg, i16 groups, i16 tbl_base, CS16 cos_tbl, CS16 parts) {
  i8 cls = frame_size_class(cfg[1]);
  u16 l = wadd<u16>(static_cast<u16>(band_count_for_mode(cls, cfg[1], tbl_base, cos_tbl)), 1);
  i16 start = high16_bL2b2(wadd<i32>(
      wmul<i32>(wmul<i32>(static_cast<i16>(static_cast<u16>(static_cast<u32>(l) << 4)), cfg[6]), 2) >> 4, 0x4000));
  i16 pick = 0;
  bool fell_through = true;
  if (0 < groups) {
    i16 acc = start;
    i16 i = 0;
    for (;;) {
      i32 idx = static_cast<i32>(i) + static_cast<i32>(groups) * 8 - 0x18;
      acc = wadd<i16>(acc, parts[static_cast<usz>(idx)]);
      if (acc >= 8) {
        if (i < groups) {
          pick = i;
          fell_through = false;
        }
        break;
      }
      i = wadd<i16>(i, 1);
      if (i >= groups) break;
    }
    if (fell_through) pick = wsub<i16>(groups, 1);
  } else {
    pick = wsub<i16>(groups, 1);
  }
  i16 q = static_cast<i16>(div32(wadd<i32>(static_cast<i16>(l), 2), 3));
  i16 d = static_cast<i16>(static_cast<i32>(groups) - static_cast<i32>(q));
  u16 c = d < 0 ? static_cast<u16>(0) : static_cast<u16>(d);
  return static_cast<u16>(shl16_bL2b2(1, c) | shl16_bL2b2(1, static_cast<u16>(pick)));
}

i16 refine_pitch_by_harmonics(CS16 corr, i16 lo, i16 n) {
  i16 peak = wadd<i16>(argmax(corr.subspan(static_cast<usz>(lo)), wsub<i16>(n, lo)), lo);
  i32 v = parabolic_peak(peak, corr, n);
  i16 s = nrm30(v);
  i16 st[10] = {};
  st[0] = high16_bL2b2(wshl<i32>(v, static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  st[1] = static_cast<i16>(0xf - s);
  st[4] = st[0];
  st[5] = st[1];
  i16 edges[4] = {};
  i16 k = 1;
  for (;;) {
    i32 exp = static_cast<i32>((static_cast<u32>(static_cast<u16>(st[2])) << 16) | static_cast<u32>(static_cast<u16>(st[1])));
    if (band_edges(S16(edges), st[0], exp, k, n) == 1) break;
    i16 width1 = wadd<i16>(wsub<i16>(edges[3], edges[2]), 1);
    if (k > 1) {
      i16 p = wadd<i16>(argmax(corr.subspan(static_cast<usz>(edges[2])), width1), edges[2]);
      i32 v2 = parabolic_peak(p, corr, n);
      i16 s2 = nrm30(v2);
      st[5] = static_cast<i16>(0xf - s2);
      st[4] = high16_bL2b2(wshl<i32>(v2, static_cast<u32>(static_cast<i32>(s2)) & 0x1f));
    }
    i16 width0 = wadd<i16>(wsub<i16>(edges[1], edges[0]), 1);
    i32 s0 = sum_i16(corr.subspan(static_cast<usz>(edges[0])), width0);
    i32 s1 = sum_i16(corr.subspan(static_cast<usz>(edges[2])), width1);
    i16 e = 0;
    st[2] = ratio_percent(static_cast<u32>(s1), s0, e);
    st[3] = e;
    update_ratio_accum(k, S16(st));
    k = wadd<i16>(k, 1);
    if (k >= 0x10) break;
  }
  return high16_bL2b2(shift32_bL2b2(up16_bL2b2(st[0]), wsub<i16>(st[1], 6)));
}

u16 band_voicing_tracker(S16 state, std::span<const i32> nw, i16 new_exp, i16 f0, i16 passthru, i16& peak, i16 gain) {
  i16 acc_m = 0, acc_e = 0, noise_m = 0, noise_e = 0, cur_m = 0, cur_e = 0, peak_e = 0, dist_m = 0, dist_e = 0;
  i16 wsum_m = 32000, wsum_e = 10, prev_m = 0, prev_e = 0;
  usz out_at = 0x81;
  peak = 0;
  i32 fv = up16_bL2b2(f0);
  u16 nf = static_cast<u16>(nrm30(fv));
  i16 dens_e = 0;
  u16 dens = static_cast<u16>(div_norm(static_cast<i32>(0x5b95'0000u), 0, high16_bL2b2(wshl<i32>(fv, static_cast<u32>(nf) & 0x1f)),
                                       static_cast<i16>(wsub<u32>(0xfu, nf)), dens_e));
  i16 env[16] = {};
  {
    i16 m[16], e[16];
    std::copy_n(state.begin() + 0x29, 16, m);
    std::copy_n(state.begin() + 0x39, 16, e);
    band_gain_envelope(CS16(m), CS16(e), S16(env));
  }
  i16 lvl = state[0x24];
  i32 step;
  if (lvl < -0x18ff) {
    step = 0x200;
  } else {
    i16 d = static_cast<i16>(wsub<i32>(lvl, state[0x28]));
    i16 t = static_cast<i16>(wmul<i32>(d, 0x3332) >> 16);
    if (t > 0x200) step = 0x200;
    else if (t <= 0x100) step = 0x100;
    else step = static_cast<u16>(t);
  }
  i64 sum = 0;
  for (i16 band = 0; band < 16; band = static_cast<i16>(band + 1)) {
    usz i = static_cast<usz>(band);
    u16 f2 = static_cast<u16>(state[0x79]);
    i32 a = nw[i];
    i32 b = get32_bL2b2(state, 2 * i);
    u16 na = static_cast<u16>(nrm30(a));
    u16 nb = static_cast<u16>(nrm30(b));
    i16 e_sum = 0;
    u16 m_sum = static_cast<u16>(add_norm(high16_bL2b2(wshl<i32>(a, static_cast<u32>(na) & 0x1f)),
                                          static_cast<i16>(wsub<u16>(static_cast<u16>(new_exp), na)),
                                          high16_bL2b2(wshl<i32>(b, static_cast<u32>(nb) & 0x1f)),
                                          static_cast<i16>(wsub<u16>(f2, nb)), e_sum));
    e_sum = wsub<i16>(e_sum, 1);
    u16 lac = static_cast<u16>(state[0x29 + i]);
    u16 la0 = static_cast<u16>(state[0x39 + i]);
    u16 l94 = static_cast<u16>(state[0x49 + i]);
    u16 la4 = static_cast<u16>(state[0x59 + i]);
    i16 d_mant = 0, d_exp = 0, log_e = 0;
    i16 log_m = static_cast<i16>(static_cast<u16>(
        log_scaled_product(d_mant, d_exp, log_e, band, static_cast<i16>(m_sum), e_sum, static_cast<i16>(dens), dens_e)));
    i16 thr_e = 0;
    u16 thr = static_cast<u16>(add_norm(static_cast<i16>(lac), static_cast<i16>(la0), static_cast<i16>(step), 7, thr_e));
    i16 hang = state[0x9c];
    u16 ret;
    if (hang < 8) {
      i16 c = static_cast<i16>(l94);
      i16 ag = static_cast<i16>(la4);
      i16 v = static_cast<i16>(lac);
      u16 e = la0;
      ret = static_cast<u16>(update_pitch_track(v, e, c, ag, log_m, log_e, passthru, hang));
      lac = static_cast<u16>(v);
      la0 = e;
      l94 = static_cast<u16>(c);
      la4 = static_cast<u16>(ag);
    } else if (compare_scaled(log_m, log_e, static_cast<i16>(thr), thr_e)) {
      i16 oa = static_cast<i16>(l94);
      i16 ob = static_cast<i16>(la4);
      i16 v = static_cast<i16>(lac);
      u16 e = la0;
      ret = blend_toward(v, e, oa, ob, log_m, static_cast<u16>(log_e), static_cast<u16>(passthru));
      lac = static_cast<u16>(v);
      la0 = e;
      l94 = static_cast<u16>(oa);
      la4 = static_cast<u16>(ob);
    } else {
      i16 age = state[0x69 + i];
      if (age > 0) {
        i16 m = static_cast<i16>(l94);
        i16 e = static_cast<i16>(la4);
        ret = static_cast<u16>(track_level_ar(m, e, log_m, log_e, age, static_cast<i16>(step)));
        l94 = static_cast<u16>(m);
        la4 = static_cast<u16>(e);
      } else {
        lac = l94;
        la0 = la4;
        ret = static_cast<u16>(passthru);
        l94 = 0x7fff;
        la4 = 5;
      }
    }
    state[0x69 + i] = static_cast<i16>(ret);
    state[0x49 + i] = static_cast<i16>(l94);
    state[0x59 + i] = static_cast<i16>(la4);
    state[0x29 + i] = static_cast<i16>(lac);
    state[0x39 + i] = static_cast<i16>(la0);
    put32_bL2b2(state, 2 * i, nw[i]);
    i16 diff_m = 0, diff_e = 0;
    accumulate_weighted_diff(dist_m, dist_e, wsum_m, wsum_e, diff_m, diff_e, static_cast<i16>(lac), static_cast<i16>(la0), log_m,
                             log_e, static_cast<i16>(m_sum), e_sum, band);
    noise_track_update(acc_m, acc_e, noise_m, noise_e, static_cast<i16>(m_sum), e_sum, log_m, log_e, static_cast<i16>(lac),
                       static_cast<i16>(la0), cur_m, cur_e, peak, peak_e, band, gain, env[i], state[0x24], state[0x28]);
    i16 dm = diff_m;
    i16 de = diff_e;
    if ((band & 1) == 0) {
      prev_m = dm;
      prev_e = de;
    } else {
      i16 e = prev_e;
      i16 r = add_norm(prev_m, prev_e, dm, de, e);
      prev_m = r;
      prev_e = e;
      state[out_at] = high16_bL2b2(shift32_bL2b2(up16_bL2b2(r), wsub<i16>(e, 7)));
      out_at += 1;
    }
    if (band == 0) state[0x99] = high16_bL2b2(shift32_bL2b2(up16_bL2b2(dm), wsub<i16>(de, 5)));
    sum = wadd<i64>(sum, static_cast<i64>(shift32_bL2b2(up16_bL2b2(log_m), wsub<i16>(log_e, 7))));
  }
  i16 mean = static_cast<i16>(wadd<i32>(static_cast<i32>(sum >> 4), 0x0700'0000) >> 16);
  state[0x9d] = high16_bL2b2(wadd<i32>(wmul<i32>(mean, 0x9a1c), wmul<i32>(state[0x9d], 0xcccc)));
  i16 e88 = 0;
  i16 q = div_norm(up16_bL2b2(dist_m), dist_e, wsum_m, wsum_e, e88);
  state[0x7a] = high16_bL2b2(shl_sat(up16_bL2b2(q), wsub<i16>(e88, 2)));
  i32 v = shl_sat(wmul<i32>(peak, 0x3332), wsub<i16>(peak_e, 3));
  i32 r = wadd<i32>(v, 0x8000);
  if (((r ^ v) & r) < 0) r = wadd<i32>(0x7fff'ffff, v < 0 ? 1 : 0);
  peak = high16_bL2b2(r);
  i16 fm, fe;
  if (noise_e < 3) { fm = 0x4000; fe = 3; } else { fm = noise_m; fe = noise_e; }
  i16 e80 = acc_e;
  i16 s = div_norm(up16_bL2b2(acc_m), acc_e, fm, fe, e80);
  v = shl_sat(up16_bL2b2(s), wsub<i16>(e80, 3));
  r = wadd<i32>(v, 0x8000);
  if (((r ^ v) & r) < 0) r = wadd<i32>(0x7fff'ffff, v < 0 ? 1 : 0);
  state[0x79] = new_exp;
  return static_cast<u16>(r >> 16);
}

static u16 norm_shift_bL2b3(i32 v) {
  u32 m = v < 0 ? ~static_cast<u32>(v) : static_cast<u32>(v);
  u32 b = m == 0 ? 0u : static_cast<u32>(31 - __builtin_clz(m));
  return static_cast<u16>(wsub<u32>(0x1eu, b));
}
static std::pair<i16, i16> norm_pair(i32 x) {
  u16 n = x == 0 ? static_cast<u16>(0) : norm_shift_bL2b3(x);
  i16 m = static_cast<i16>(static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(n) & 0x1f)) >> 16);
  return {m, wneg<i16>(static_cast<i16>(n))};
}
static i32 shift32_bL2b3(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f);
}
static i16 high16_bL2b3(i32 x) { return static_cast<i16>(x >> 16); }

u32 boost_gain_if_above_floor(i16& gain, i16 lo, i16 level, i16 level_exp, i16 hi, i16 floor_, i16 rate, i16 bias) {
  i16 at_scale = high16_bL2b3(shift32_bL2b3(static_cast<i32>(level) << 16, wsub<i16>(level_exp, 5)));
  if (at_scale <= floor_) return 1;
  i32 headroom = wsub<i32>(wsub<i32>(static_cast<i32>(hi) << 16, static_cast<i32>(lo) << 16), static_cast<i32>(bias) << 16);
  if (headroom <= 0) return 1;
  i16 h = high16_bL2b3(headroom);
  if (h > 0x200) h = 0x200;
  i32 p = wmul<i32>(wmul<i32>(h, rate), 2);
  u16 sh = p == 0 ? static_cast<u16>(0) : norm_shift_bL2b3(p);
  i16 mant = static_cast<i16>(static_cast<i32>(static_cast<u32>(p) << (static_cast<u32>(sh) & 0x1f)) >> 16);
  i16 g = static_cast<i16>(pow2_scaled(mant, wsub<i16>(8, static_cast<i16>(sh)), 3));
  i32 prod = wmul<i32>(wmul<i32>(g, gain), 2);
  gain = high16_bL2b3(shl_sat(prod, 3));
  return 0;
}

void slot_band_weights(S16 out, S16 clamped, CS16 bands, CS16 mags, i32 active_mask, i32 slot, i32 stride, u16& flag, CS16 cos_tbl) {
  i32 s = static_cast<i16>(slot);
  i32 d = static_cast<i16>(stride);
  CS16 m = mags.subspan(static_cast<usz>(s) * 8, 8);
  for (usz j = 0; j < 8; j++) {
    i16 v = m[j];
    clamped[j] = v < 1 ? static_cast<i16>(1) : v;
  }
  i32 mask16 = static_cast<i16>(active_mask);
  u32 sh_cur = static_cast<u32>(static_cast<u8>(wadd<u32>(static_cast<u32>(slot), 2u)) & 0x1f);
  if (((mask16 >> sh_cur) & 1) == 0) {
    std::fill_n(out.begin(), 8, static_cast<i16>(0));
    flag = 0;
    return;
  }
  copy16_batch3(out, m, 8);
  harmonic_dft(out, 8, 2, cos_tbl);
  i32 t = (static_cast<i32>(out[0]) << 16) >> 1;
  t = t == INT32_MIN ? INT32_MAX : wneg<i32>(t);
  u16 f = (static_cast<i32>(out[1]) << 16) < t ? 1 : 0;
  flag = f;
  i16 shifts[8] = {0, 0, 0, 0, static_cast<i16>(f), static_cast<i16>(f), static_cast<i16>(f), static_cast<i16>(f)};
  u32 sh_prev = static_cast<u32>(static_cast<u8>(wadd<u8>(static_cast<u8>(wsub<u32>(static_cast<u32>(slot), static_cast<u32>(stride))), 2)) & 0x1f);
  bool has_prev = static_cast<i16>(slot) != 0 && ((mask16 >> sh_prev) & 1) != 0;
  i32 acc[8] = {};
  CS16 cur = bands.subspan(static_cast<usz>(s) * 8);
  CS16 next = bands.subspan(static_cast<usz>(s + d) * 8);
  if (has_prev) {
    CS16 prev = bands.subspan(static_cast<usz>(s - d) * 8);
    band_min_of_max(std::span<i32>(acc), cur, prev, next, static_cast<i16>(active_mask), static_cast<i8>(stride), static_cast<i8>(slot));
  } else {
    voiced_scale8(std::span<i32>(acc), cur, next, static_cast<i16>(active_mask), static_cast<i16>(stride), static_cast<i16>(slot));
  }
  log_weight8(out, std::span<const i32>(acc), 0, CS16(shifts));
}

u32 log_spectrum_pairs(S16 dst, CS16 src, i32 exp, u16 n) {
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 v = rd32(src, 2 * i);
      u16 sh = v == 0 ? static_cast<u16>(0) : norm_shift_bL2b3(v);
      i16 mant = static_cast<i16>(static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(sh) & 0x1f)) >> 16);
      i16 e = static_cast<i16>(wsub<i32>(exp, sh));
      dst[2 * i] = (mant == 0 || e <= -0x20) ? static_cast<i16>(0x8000u) : static_cast<i16>(log2_at_q(mant, e, 5));
      dst[2 * i + 1] = 0;
    }
  }
  return 4;
}

void block_normalise_to_q15(S16 dst, std::span<const i32> src, i16 n, i16& exp) {
  i16 be = static_cast<i16>(i32_block_exp(src, n));
  round_block_i32(dst, static_cast<u16>(be), src, 0, static_cast<u16>(n));
  exp = wadd<i16>(exp, be);
}

void pick_pitch_by_band_ratio(i16& num, i16& num_exp, i16& den, i16& den_exp, CS16 v, i32 step, i32 start1, i32 start2) {
  i16 n1 = static_cast<i16>(wsub<i32>(0x20, start1));
  auto [den1, den1_e] = norm_pair(sum_i16(v.subspan(static_cast<usz>(static_cast<i16>(start1))), n1));
  auto [num1, num1_e] = norm_pair(sum_bands(v, static_cast<i16>(step), 6, 0x40));
  if (static_cast<i16>(step) > 0x0ccd) {
    i16 n2 = static_cast<i16>(wsub<i32>(0x20, start2));
    auto [den2, den2_e] = norm_pair(sum_i16(v.subspan(static_cast<usz>(start2)), n2));
    auto [num2, num2_e] = norm_pair(sum_bands(v, static_cast<i16>(step), 5, 0x40));
    i32 p2 = wmul<i32>(wmul<i32>(num2, den1), 2);
    i32 p1 = wmul<i32>(wmul<i32>(den2, num1), 2);
    i16 s2 = static_cast<i16>(p2 == 0 ? static_cast<u16>(0) : norm_shift_bL2b3(p2));
    i16 s1 = static_cast<i16>(p1 == 0 ? static_cast<u16>(0) : norm_shift_bL2b3(p1));
    i16 m2 = static_cast<i16>(static_cast<i32>(static_cast<u32>(p2) << (static_cast<u32>(static_cast<i32>(s2)) & 0x1f)) >> 16);
    i16 m1 = static_cast<i16>(static_cast<i32>(static_cast<u32>(p1) << (static_cast<u32>(static_cast<i32>(s1)) & 0x1f)) >> 16);
    i16 e2 = wadd<i16>(wsub<i16>(num2_e, s2), den1_e);
    i16 e1 = wadd<i16>(wsub<i16>(den2_e, s1), num1_e);
    if (mant_exp_gt(m2, e2, m1, e1)) {
      num = num2;
      num_exp = num2_e;
      den = den2;
      den_exp = den2_e;
      return;
    }
  }
  num = num1;
  num_exp = num1_e;
  den = den1;
  den_exp = den1_e;
}

static i32 shift32_bL2b4(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<u16>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(static_cast<u16>(wneg<i16>(s))) & 0x1f);
}
static i64 allshl_bL2b4(i64 v, u8 count) { return allshl_al2b3(v, count); }  // identical code
static u16 shl16_bL2b4(u16 v, u32 c) {
  c &= 0x1f;
  return c >= 16 ? static_cast<u16>(0) : static_cast<u16>(v << c);
}
static i32 sar32(i32 v, u32 c) { return v >> (c & 0x1f); }

void pack_tone_frame(i16 tone, CS16 amp, CS16 cfg, usz& cursor, S16 dst) {
  emit_bits(cursor, dst, 6, 0x3f);
  i16 level;
  if (tone < 0x80) {
    level = amp[1];
  } else {
    i32 s = wadd<i32>((static_cast<i32>(amp[1]) << 16) >> 1, (static_cast<i32>(amp[0]) << 16) >> 1);
    i16 m = static_cast<i16>(s >> 16);
    level = m > 0x6fff ? static_cast<i16>(0x6fff) : (m < -0x5000 ? static_cast<i16>(-0x5000) : m);
  }
  i32 q = wmul<i32>(wadd<i32>(level, 0x800), 0x220) >> 17;
  i16 idx = static_cast<i16>(q);
  if (idx > 0x7f) idx = 0x7f;
  else if (idx < 0) idx = 0;
  emit_bits(cursor, dst, 6, static_cast<i16>(idx >> 1));
  i16 n = static_cast<i16>((static_cast<i32>(cfg[2]) - 0xd) >> 3);
  while (n > 0) {
    emit_bits(cursor, dst, 8, tone);
    n = wsub<i16>(n, 1);
  }
  emit_bits(cursor, dst, 1, static_cast<i16>(idx & 1));
}

u16 quantise_voicing_bL2b4(i16 enable, u32& mask0, u32& mask1, CS16 w_a, CS16 w_b, CS16 w_c, std::span<const u32> book, i32 nbits) {
  u32 sh = static_cast<u32>(wsub<i32>(7, nbits));
  if (enable == 1) {
    u16 idx = static_cast<u16>(voicing_codebook_search(w_a, w_b, w_c, book, static_cast<u8>(nbits), 1));
    i32 slot = static_cast<i16>(shl16_bL2b4(idx, sh));
    expand_dibits(mask0, mask1, book[static_cast<usz>(slot)]);
    return idx;
  }
  mask1 = 0;
  mask0 = 0;
  return static_cast<u16>(static_cast<i16>(sar32(0x40, sh)));
}

u16 band_pair_at_common_exp(i16& out_a, i16& out_b, std::span<const i32> energy, i16 exp, i16 idx_a, i16 idx_b) {
  i16 ea = exp;
  i32 va = accumulate_exp(energy.subspan(static_cast<usz>(idx_a)), 5, ea);
  i16 eb = exp;
  i32 vb = accumulate_exp(energy.subspan(static_cast<usz>(idx_b)), 5, eb);
  i32 ua = static_cast<u16>(ea);
  i32 ub = static_cast<u16>(eb);
  i32 common = ea > eb ? ua : ub;
  out_a = static_cast<i16>(shift32_bL2b4(va, static_cast<i16>(ua - common)) >> 16);
  out_b = static_cast<i16>(shift32_bL2b4(vb, static_cast<i16>(ub - common)) >> 16);
  return static_cast<u16>(common);
}

void subband_analysis(S16 hist, i16& hist_exp, S16 out, S16 work, i16 exp, std::span<i32> acc, i16 acc_exp, CS16 input, i16 frames,
                      CS16 taper, CS16 interp_coef) {
  i32 last = static_cast<i32>(frames) - 1;
  i32 marker = last > 0 ? 0 : static_cast<u16>(static_cast<i32>(frames) - 1);
  if (static_cast<u16>(exp) == static_cast<u16>(hist_exp)) copy16_batch3(work, hist, 0x50);
  else scale_block_exp(work, hist, 0x50, static_cast<u16>(hist_exp), exp);
  i16 pow_exp = static_cast<i16>(static_cast<u16>(wmul<i16>(exp, 2)));
  if (frames < 1) {
    hist_exp = exp;
    return;
  }
  i16 spec[32] = {};
  i32 power[16] = {};
  i32 col = 0;
  usz wp = 0x50;
  i32 hop = 0;
  i32 hop16 = 0;
  usz in_off = 0;
  for (;;) {
    for (int rep = 0; rep < 2; rep++) {
      apply_sym_window(S16(spec), input.subspan(in_off), 0x20, 0x10, taper, 0x20);
      dft32_real(S16(spec));
      i16 dc = spec[0];
      cplx_mag_sq_x2(std::span<i32>(power), CS16(spec), 0x10);
      in_off += 4;
      acc_add_shifted(std::span<const i32>(power), pow_exp, acc, acc_exp, 0x10);
      work[wp] = dc > 0 ? dc : static_cast<i16>(0);
      for (usz j = 0; j < 15; j++) work[wp + 1 + j] = static_cast<i16>(sqrt_to_exp(power[1 + j], pow_exp, exp));
      wp += 0x10;
    }
    if (static_cast<i16>(hop16) == static_cast<i16>(marker)) {
      interp_filter(out, interp_coef, CS16(work), static_cast<i16>(hop16 - col + 1), static_cast<i16>(col));
      if (hop == last) {
        usz k = static_cast<usz>((hop - static_cast<i32>(static_cast<i16>(col)) + 1) * 0x20);
        copy16_batch3(hist, work.subspan(k), 0x50);
      } else {
        std::copy(work.begin() + 0x20, work.begin() + 0x20 + 0x50, work.begin());
        wp -= 0x20;
      }
      col += 1;
      marker += 1;
      if (static_cast<i32>(static_cast<i16>(marker)) > last) marker = static_cast<u16>(static_cast<i32>(frames) - 1);
    }
    hop += 1;
    hop16 += 1;
    if (static_cast<i16>(hop16) >= frames) break;
  }
  hist_exp = exp;
}

void band_voicing_measure_bL2b4(S16 measure, std::span<const i32> sig, i32 sig_exp, std::span<const i32> err, i16 err_exp, i16 spacing,
                                u16 max_bin) {
  i16 e = 0;
  i16 m = div_norm(0x2000'0000, -3, spacing, -3, e);
  i32 half = shift32_bL2b4(static_cast<i32>(m) << 16, wsub<i16>(e, 0xf));
  i32 step = wmul<i32>(half, 2);
  u16 start = 0;
  i32 edge = 0;
  for (usz b = 0; b < 8; b++) {
    edge = wadd<i32>(edge, step);
    i16 rounded = static_cast<i16>(wadd<i32>(edge, 0x8000) >> 16);
    i32 stop = rounded > static_cast<i16>(max_bin) ? static_cast<i32>(max_bin) : static_cast<i32>(static_cast<u16>(rounded));
    i32 base = static_cast<i32>(static_cast<i16>(start)) * 4;
    u16 span = static_cast<u16>(stop - static_cast<i32>(start));
    i64 acc = 0;
    if (span < 0x8000) {
      usz o = static_cast<usz>(base / 4);
      for (usz k = 0; k <= span; k++) acc = wadd<i64>(acc, static_cast<i64>(err[o + k]));
    }
    i16 n1 = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
    i64 s1 = n1 >= 0 ? allshl_bL2b4(acc, static_cast<u8>(n1)) : allshr(acc, static_cast<u8>(wneg<i32>(n1)));
    i16 num_exp = static_cast<i16>(static_cast<u16>(err_exp - static_cast<i32>(static_cast<u16>(n1))));
    i16 num = static_cast<i16>(static_cast<i32>(s1) >> 16);
    acc = 0;
    if (span < 0x8000) {
      usz o = static_cast<usz>(base / 4);
      for (usz k = 0; k <= span; k++) acc = wadd<i64>(acc, static_cast<i64>(sig[o + k]));
    }
    i16 n2 = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
    i64 s2 = n2 >= 0 ? allshl_bL2b4(acc, static_cast<u8>(n2)) : allshr(acc, static_cast<u8>(wneg<i32>(n2)));
    i16 den = static_cast<i16>(static_cast<i32>(s2) >> 16);
    i16 den_exp = static_cast<i16>(sig_exp - static_cast<i32>(static_cast<u16>(n2)));
    i16 v = static_cast<i16>(one_minus_ratio(num, num_exp, den, den_exp));
    if (v < measure[b]) measure[b] = v;
    start = static_cast<u16>(wadd<i32>(wsub<i32>(edge, half), 0x8000) >> 16);
  }
}

using namespace enc_tbl;

static u16 norm_shift_abs(u32 x) {
  assert(x != 0);
  u32 bsr = 31 - static_cast<u32>(__builtin_clz(x));
  return static_cast<u16>(wsub<u32>(0x1eu, bsr));
}
static i32 shift_i32(i32 v, i16 sh) {
  if (sh >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f));
  if (sh < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(static_cast<i32>(wneg<i16>(sh))) & 0x1f);
}
static void wr32_bL3b1(S16 st, usz w, u32 v) {
  st[w] = static_cast<i16>(static_cast<u16>(v));
  st[w + 1] = static_cast<i16>(static_cast<u16>(v >> 16));
}

void enc_init(S16 st, i16 mode, i16 reset) {
  if (reset != 0) {
    fill16(st, 0, 0xb70);
    fill16(st.subspan(0x8b8), -15, 7);
    wr32_bL3b1(st, 0x318, 0x147b'2aabu);
    st[0x31a] = 0x147b;
    for (usz i = 0; i < 5; i++) {
      st[0x31d + i] = 0x1000;
      st[0x322 + i] = 0x7fff;
    }
    for (usz i = 0; i < 5; i++) {
      fill16(st.subspan(0x34f + 8 * i), 1, 8);
      st[0x377 + i] = -30;
      fill16(st.subspan(0x327 + 8 * i), 3, 8);
    }
    for (usz i = 0; i < 3; i++) {
      st[0x385 + i] = -30;
      fill16(st.subspan(0x388 + 8 * i), 0x7fff, 8);
    }
    wr32_bL3b1(st, 0x4c5, 0xa001'a401u);
    wr32_bL3b1(st, 0x4c7, 0x9801'9c01u);
    wr32_bL3b1(st, 0x4c9, 0x9001'9401u);
    fill16(st.subspan(0x4cb), static_cast<i16>(0x8c01u), 10);
    fill16(st.subspan(0x4d5), 5, 0x10);
    fill16(st.subspan(0x4e5), 0x7fff, 0x10);
    fill16(st.subspan(0x4f5), 5, 0x10);
    fill16(st.subspan(0x505), 300, 0x10);
    wr32_bL3b1(st, 0x4c3, 0xd301'012cu);
    st[0x4c0] = static_cast<i16>(0xe201u);
    st[0x53b] = -16;
    wr32_bL3b1(st, 0x4bc, 0x000f'a000u);
    wr32_bL3b1(st, 0x539, 0x4e20'e200u);
    fill16(st.subspan(0x53c), 0, 0x10);
    fill16(st.subspan(0x54c), -30, 0x10);
    wr32_bL3b1(st, 0x55c, 0);
    wr32_bL3b1(st, 0x4c2, 0x012c'0000u);
    st[0x917] = 0x2000;
    st[0x9ba] = 0x20;
    gain_for_rate(st[0x9b9], 0x20, -10);
    st[0x9b8] = 0;
    wr32_bL3b1(st, 0x9bf, 0xb001'b001u);
    wr32_bL3b1(st, 0x9c1, 0xb001'b001u);
    wr32_bL3b1(st, 0x9c3, 0xb001'b001u);
    set_agc_defaults(st, -22, 20, -10);
  }
  A13 alloc(st.data() + 0x9c5, 13);
  init_bit_alloc_and_agc(mode, alloc, st.subspan(0x55e, 0x9c5 - 0x55e));
}

i16 analysis_fft(std::optional<std::span<i32>> mag, CS16 input, CS16 win, i32 exp_in, i16 count, i16 out_off, S16 fft_buf, i16 log2n,
                 i16 extra) {
  i16 scratch[256] = {};
  i16 exp = 0;
  copy16_batch3(S16(scratch), input, count);
  normalise_to_exp_inplace(S16(scratch), static_cast<u16>(count), exp);
  i16 n = static_cast<i16>(static_cast<u16>(1u << (static_cast<u16>(log2n) & 0x1f)));
  apply_sym_window(fft_buf, CS16(scratch), n, out_off, win, count);
  i16 e = static_cast<i16>(rfft_forward(fft_buf, wadd<i32>(exp_in, static_cast<u16>(exp)), log2n, extra));
  if (!mag) return e;
  cplx_mag_sq_x2(*mag, fft_buf, wadd<i16>(static_cast<i16>(n >> 1), extra));
  return wmul<i16>(e, 2);
}

void imbe_frame_deinterleave(S16 chan, i16 nharm, CS16 hbits, CS16 alloc) { interleave_imbe_frame(chan, nharm, hbits, alloc, false); }

u16 log2_vector(S16 dst, CS16 src, i16 exp, i16 n) {
  i16 target = static_cast<i16>(wadd<i32>(exp, 3) >> 2);
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) dst[i] = static_cast<i16>(log2_at_exp(static_cast<u32>(rd32(src, 2 * i)), exp, target));
  }
  normalise_to_exp_inplace(dst, static_cast<u16>(n), target);
  return static_cast<u16>(target);
}

static i16 tone_smooth(i16 prev, i16 x) {
  i32 a = wmul<i32>(prev, 0xdffe);
  i32 b = wshl<i32>(wadd<i32>(x, 4), 13);
  return static_cast<i16>(wadd<i32>(a, b) >> 16);
}
static i16 clamp_level(i16 v) {
  if (v > 0x7800) return 0x7800;
  if (v >= -0x1400) return v;
  return -0x1400;
}

i32 detect_signalling_tone(i16& peak_out, S16 result, S16 st, std::span<const i32> spec, u16 exp, i16 offset, i16 ref_level, i32 ref_exp,
                           i16 enable) {
  i16 cls = st[4];
  if (cls != 0x00ff && (cls < 5 || cls > 0x7a)) return 0;
  i16 peak = wadd<i16>(argmax_i32(spec.subspan(5), 0x76), 5);
  std::span<const i32> win = spec.subspan(static_cast<usz>(static_cast<i32>(peak) - 2), 5);
  i16 pexp = static_cast<i16>(exp);
  i32 acc = accumulate_exp(win, 5, pexp);
  i16 mant = static_cast<i16>(acc >> 16);
  i16 lvl_peak = -0x1400;
  if (pexp >= -0x3b && mant != 0) {
    i16 l = static_cast<i16>(log2_at_q(mant, wadd<i16>(pexp, 0x1e), 5));
    lvl_peak = clamp_level(wadd<i16>(l, offset));
  }
  i32 d = wsub<i32>(static_cast<u16>(pexp), ref_exp);
  i32 scaled = shift_i32(static_cast<i32>(mant) << 16, static_cast<i16>(d));
  i32 diff = wsub<i32>(wshl<i32>(ref_level, 16), scaled);
  u16 sh;
  if (diff < 0) {
    if (diff == INT32_MIN) {
      diff = 0x7fff'ffff;
      sh = norm_shift_abs(0x7fff'ffffu);
    } else {
      diff = wneg<i32>(diff);
      sh = norm_shift_abs(static_cast<u32>(diff));
    }
  } else if (diff == 0) {
    sh = 0;
  } else {
    sh = norm_shift_abs(static_cast<u32>(diff));
  }
  i16 fexp = static_cast<i16>(static_cast<u16>(wsub<i32>(ref_exp, sh)));
  i16 fmant = static_cast<i16>(static_cast<i32>(static_cast<u32>(diff) << (static_cast<u32>(sh) & 0x1f)) >> 16);
  i16 lvl_floor = -0x1400;
  if (fexp >= -0x3b && fmant != 0) {
    i16 l = static_cast<i16>(log2_at_q(fmant, wadd<i16>(fexp, 0x1e), 5));
    lvl_floor = clamp_level(wadd<i16>(l, offset));
  }
  cls = st[4];
  i16 lp = lvl_peak, lf = lvl_floor;
  if (cls >= 5) {
    i16 dd = wsub<i16>(peak, cls);
    bool near_ = dd == -0x8000 ? false : (std::abs(static_cast<i32>(dd)) < 2);
    if (near_) {
      lp = tone_smooth(st[0xb], lp);
      lf = tone_smooth(st[0xc], lf);
    }
  }
  i16 delta = wsub<i16>(lp, lf);
  if (enable != 0 && cls == 0x00ff && peak >= 0x58 && delta < 0x2a85) return 0;
  if (delta < 0x1a93) return 0;
  if (peak <= 0x12 && delta < 0x2138) return 0;
  if (peak <= 0xa && delta < 0x2689) return 0;
  i16 shape[6] = {};
  i16 sexp = static_cast<i16>(exp);
  block_normalise_to_q15(S16(shape), win, 5, sexp);
  if (!spread_test(mant, static_cast<u16>(pexp), CS16(shape), static_cast<u16>(sexp), 5)) return 0;
  peak_out = peak;
  result[0] = static_cast<i16>(0xb000u);
  result[1] = static_cast<i16>(log_amp_clamped(mant, static_cast<u16>(pexp), offset));
  st[0xc] = lf;
  st[0xb] = lp;
  return 1;
}

i16 band_voicing_decision(S16 st, i16 step, CS16 spec, CS16 band_exp) {
  for (usz k = 0; k < 3; k++) {
    usz d = 0x20 - 8 * k;
    std::copy(st.begin() + (d - 0x10), st.begin() + (d - 8), st.begin() + d);
    usz d2 = 0x48 - 8 * k;
    std::copy(st.begin() + (d2 - 0x10), st.begin() + (d2 - 8), st.begin() + d2);
    usz p = 0x54 - k;
    st[p] = st[p - 2];
  }
  i32 iv = static_cast<i32>(step) << 16;
  i32 start1 = static_cast<u16>(wadd<i32>(iv >> 10, 0x10000) >> 16);
  i32 start2 = static_cast<u16>(wadd<i32>(iv >> 11, 0x10000) >> 16);
  i16 mask = 0;
  i16 exps[8] = {};
  for (usz i = 0; i < 8; i++) {
    i16 num = 0, num_e = 0, den = 0, den_e = 0;
    pick_pitch_by_band_ratio(num, num_e, den, den_e, spec.subspan(i * 0x20), step, start1, start2);
    st[0x28 + i] = den;
    exps[i] = wadd<i16>(band_exp[i], den_e);
    i16 r = static_cast<i16>(one_minus_ratio(num, num_e, den, den_e));
    mask = wmul<i16>(mask, 2);
    st[i] = r;
    if (r < 0x199a) mask = wadd<i16>(mask, 1);
  }
  i16 mants[8];
  std::copy_n(st.begin() + 0x28, 8, mants);
  i16 e = static_cast<i16>(align_block_exp(st.subspan(0x28, 8), CS16(mants), CS16(exps), 8));
  st[0x50] = e;
  i16 orv = 0;
  for (usz i = 0; i < 8; i++) orv = static_cast<i16>(orv | st[0x28 + i]);
  if (orv == 0) {
    for (usz i = 0; i < 8; i++) st[0x28 + i] = 1;
    st[0x50] = -30;
  }
  return mask;
}

static i16 high16_bL3b2(i32 x) { return static_cast<i16>(x >> 16); }
static i32 top_bit_bL3b2(u32 v) { return v == 0 ? 31 : 31 - __builtin_clz(v); }
static i32 shift32_bL3b2(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f);
}
static i64 allshl_bL3b2(i64 v, u8 count) { return allshl_al2b3(v, count); }  // identical code
static std::pair<i16, i16> norm_shift_mant(i64 acc) {
  i16 s = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
  i64 r = s < 0 ? allshr(acc, static_cast<u8>(static_cast<u32>(wneg<i32>(s)) & 0xff))
                : allshl_bL3b2(acc, static_cast<u8>(static_cast<u32>(static_cast<i32>(s)) & 0xff));
  return {high16_bL3b2(static_cast<i32>(r)), s};
}

void boost_three_band_gains(S16 gains, CS16 lo, CS16 level, CS16 level_exp, i16 count) {
  i16 bias = wmul<i16>(static_cast<i16>(static_cast<i8>(count)), 0x100);
  static const i16 HI[3] = {0x480, 0x400, 0x380};
  static const i16 FLOOR[3] = {-0x4800, -0x5000, -0x5400};
  static const i16 RATE[3] = {0x42b8, 0x32b8, 0x22b8};
  for (usz b = 0; b < 3; b++) boost_gain_if_above_floor(gains[b], lo[b], level[2 * b], level_exp[2 * b], HI[b], FLOOR[b], RATE[b], bias);
}

u16 rotate_spectrum_by_phasor(S16 buf, CS16 phasor, i32 phasor_exp, i32 spec_exp, i16 n) {
  i16 exp = static_cast<i16>(static_cast<u16>(wadd<i32>(wadd<i32>(spec_exp, phasor_exp), 1)));
  i32 pr = phasor[0];
  i32 pi_ = phasor[1];
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) {
      i32 re = buf[2 * i];
      i32 im = buf[2 * i + 1];
      i32 a = wsub<i32>(wmul<i32>(pr, re), wmul<i32>(pi_, im));
      i32 b = wadd<i32>(wmul<i32>(re, pi_), wmul<i32>(pr, im));
      buf[2 * i] = high16_bL3b2(a);
      buf[2 * i + 1] = high16_bL3b2(b);
    }
  }
  usz count = wmul<u16>(static_cast<u16>(n), 2);
  i16 sbuf[0x102] = {};
  CS16 src = snap(S16(sbuf), buf.first(count));
  normalise_to_exp(buf, src, static_cast<u16>(count), exp);
  return static_cast<u16>(exp);
}

i32 minimum_phase_phasor(S16 buf, i32 exp_in, i32 log2n, i16 n, CS16 cos_tbl) {
  i16 half = static_cast<i16>(static_cast<u16>(n / 2));
  u16 keep = wadd<u16>(static_cast<u16>(half), 1);
  i32 r = inverse_real_fft(buf, exp_in, log2n, 1);
  buf[0] = static_cast<i16>(buf[0] >> 1);
  u32 e32 = static_cast<u32>(r) & 0xffff;
  buf[static_cast<usz>(half)] = static_cast<i16>(buf[static_cast<usz>(half)] >> 1);
  i16 sbuf[0x102] = {};
  CS16 src = snap(S16(sbuf), buf.first(keep));
  scale_vec(buf, src, static_cast<i16>(keep), 0x70f7);
  e32 = wsub<u32>(e32, 2u);
  i16 e16 = static_cast<i16>(static_cast<u16>(e32));
  src = snap(S16(sbuf), buf.first(keep));
  normalise_to_exp(buf, src, keep, e16);
  e32 = (e32 & 0xffff'0000u) | static_cast<u32>(static_cast<u16>(e16));
  fill16(buf.subspan(static_cast<usz>(static_cast<i16>(keep))), 0, static_cast<i16>(wsub<i32>(n, keep)));
  i16 f = static_cast<i16>(static_cast<u16>(rfft_forward(buf, static_cast<i32>(e32), static_cast<i16>(log2n), 1)));
  i16 pbuf[0x102] = {};
  CS16 phase = snap(S16(pbuf), CS16(buf).subspan(1, 2 * static_cast<usz>(half)));
  cos_pairs(buf, phase, f, static_cast<u16>(half), cos_tbl);
  return 0;
}

i32 subband_filterbank(S16 out, S16 x, i32 exp, CS16 window, i32 n, CS16 tbl) {
  i16 ns = static_cast<i16>(n);
  u16 halfn = static_cast<u16>(ns >> 1);
  i16 e = static_cast<i16>(static_cast<u16>(exp));
  {
    i16 sbuf[0x102] = {};
    CS16 src = snap(S16(sbuf), CS16(x).first(2 * static_cast<usz>(n)));
    normalise_to_exp(x, src, wmul<u16>(static_cast<u16>(n), 2), e);
  }
  if (static_cast<i16>(halfn) > 0) {
    for (usz i = 0; i < halfn; i++) {
      usz ni = static_cast<usz>(static_cast<isz>(ns));
      i64 a = static_cast<i64>(x[i]) * static_cast<i64>(window[i]) * 2;
      i64 b = static_cast<i64>(x[ni - 1 - i]) * static_cast<i64>(window[ni - 1 - i]) * 2;
      i16 lo = static_cast<i16>((a - b) >> 17);
      i64 c = static_cast<i64>(window[ni - 1 - i]) * static_cast<i64>(x[ni + i]) * 2;
      i64 d = static_cast<i64>(x[2 * ni - 1 - i]) * static_cast<i64>(window[i]) * 2;
      i16 hi = static_cast<i16>((c + d) >> 17);
      x[i] = lo;
      x[ni - 1 - i] = hi;
    }
  }
  {
    i16 sbuf[0x102] = {};
    CS16 src = snap(S16(sbuf), CS16(x).first(static_cast<usz>(n)));
    normalise_to_exp(x, src, static_cast<u16>(n), e);
  }
  i32 nn = static_cast<i16>(n);
  u32 v = static_cast<u32>(nn) << 16;
  u32 sh;
  if (v == 0) {
    sh = 0;
  } else {
    u32 m = static_cast<i32>(v) < 0 ? ~v : v;
    sh = wsub<u32>(0x1eu, static_cast<u32>(top_bit_bL3b2(m))) & 0xffff;
  }
  i16 recip_exp = 0;
  i16 recip = div_norm(0x4000'0000, 1, static_cast<i16>(static_cast<i32>(v << (sh & 0x1f)) >> 16),
                       static_cast<i16>(static_cast<u16>(wsub<u32>(0xfu, sh))), recip_exp);
  i16 bin_exp[64] = {};
  usz nbins = 0;
  if (ns > 0) {
    i16 inc = 0x29;
    i16 k = 0;
    i16 cur = 0x14;
    bool done = false;
    while (!done) {
      for (i16 pos = 0; pos < 4; pos++) {
        i16 next;
        {
          i16 t = wadd<i16>(cur, 0x29);
          next = t > 0x9f ? wsub<i16>(cur, 0x77) : t;
        }
        i16 step = wadd<i16>(inc, static_cast<i16>(2 * pos));
        i16 a = cur;
        i64 acc = 0;
        if (static_cast<i16>(halfn) > 0) {
          for (usz j = 0; j < halfn; j++) {
            i16 a2;
            {
              i16 t = wadd<i16>(a, 0x78);
              a2 = t > 0x9f ? wsub<i16>(a, 0x28) : t;
            }
            i16 nxt;
            {
              i16 t = wadd<i16>(a2, step);
              nxt = t > 0x9f ? wsub<i16>(t, 0xa0) : t;
            }
            i64 asc = static_cast<i64>(x[j]) * static_cast<i64>(tbl[static_cast<usz>(a)]) * 2;
            i64 desc = static_cast<i64>(x[static_cast<usz>(ns) - 1 - j]) * static_cast<i64>(tbl[static_cast<usz>(a2)]) * 2;
            i64 term;
            switch (pos) {
              case 0: term = asc - desc; break;
              case 1: term = asc + desc; break;
              case 2: term = desc - asc; break;
              default: term = -(asc + desc); break;
            }
            acc = wadd<i64>(acc, term);
            a = nxt;
          }
        }
        auto [mant, s] = norm_shift_mant(acc);
        out[nbins] = high16_bL3b2(wmul<i32>(wmul<i32>(mant, recip), 2));
        bin_exp[nbins] = wneg<i16>(s);
        nbins += 1;
        cur = next;
        if (ns <= wadd<i16>(wadd<i16>(k, pos), 1)) {
          done = true;
          break;
        }
      }
      if (done) break;
      inc = wadd<i16>(inc, 8);
      k = wadd<i16>(k, 4);
      if (k >= ns) break;
    }
  }
  u32 max_exp = 0xffff'ffe0u;
  if (ns > 0) {
    for (usz i = 0; i < static_cast<u16>(ns); i++) {
      if (out[i] != 0 && static_cast<i16>(max_exp) < bin_exp[i]) max_exp = static_cast<u16>(bin_exp[i]);
    }
    if (static_cast<i16>(max_exp) > -0x20) {
      for (usz i = 0; i < static_cast<u16>(ns); i++) {
        i32 vv = static_cast<i32>(out[i]) << 16;
        i16 s = wsub<i16>(bin_exp[i], static_cast<i16>(static_cast<u16>(max_exp)));
        out[i] = high16_bL3b2(shift32_bL3b2(vv, s));
      }
      return wadd<i32>(wadd<i32>(wadd<i32>(e, 1), recip_exp), static_cast<i32>(max_exp));
    }
  }
  return -0x20;
}

u16 subband_analysis_frame(S16 state, S16 out, std::span<i32> acc, i16& rows, CS16 pcm, i16 origin, i16 advance, CS16 taper,
                           CS16 interp_coef) {
  constexpr usz PHASE = 0x6dc / 2;
  constexpr usz PEAK = 0x620 / 2;
  constexpr usz HIST_EXP = 0x62c / 2;
  constexpr usz COUNT = 0x630 / 2;
  constexpr usz HIST = 0x63c / 2;
  state[PHASE] = wadd<i16>(state[PHASE], advance);
  i16 hops_i16 = static_cast<i16>(state[PHASE] >> 3);
  u32 hops = static_cast<u16>(hops_i16);
  i16 rem = wadd<i16>(state[PHASE], static_cast<i16>(wmul<u16>(static_cast<u16>(hops_i16), 0xfff8)));
  u16 nwords = static_cast<u16>(wadd<u32>(wmul<u32>(hops, 8u), 0x1cu));
  state[PHASE] = rem;
  i16 inbuf[116] = {};
  i32 start = static_cast<i32>(origin) - static_cast<i32>(rem) - static_cast<i32>(static_cast<i16>(nwords));
  copy16_batch3(S16(inbuf), pcm.subspan(static_cast<usz>(start)), static_cast<i16>(nwords));
  for (usz i = 0; i < 6; i++) {
    i32 v = static_cast<i32>(state[COUNT + i]) - static_cast<i32>(hops_i16);
    state[COUNT - 1 + i] = v > 0 ? static_cast<i16>(v) : static_cast<i16>(0);
  }
  state[0x63a / 2] = static_cast<i16>(static_cast<u16>(wsub<u32>(0x3au, hops)));
  for (usz i = 0; i < 6; i++) {
    i32 hi = state[PEAK + i + 8];
    i32 lo = state[PEAK + i + 7];
    state[PEAK + i] = (hi - lo > 0) ? state[PEAK + i + 1] : static_cast<i16>(-0x8000);
  }
  rows = static_cast<i16>(static_cast<u16>(wmul<u32>(hops, 2u)));
  std::fill_n(acc.begin(), 0x10, 0);
  i16 exp = 0;
  {
    i16 sbuf[0x102] = {};
    CS16 src = snap(S16(sbuf), CS16(inbuf).first(nwords));
    normalise_to_exp(S16(inbuf), src, nwords, exp);
  }
  i16 acc_exp = static_cast<i16>(static_cast<i32>(static_cast<u16>(exp)) * 2 + 7);
  i16 work[112] = {};
  i16 hist_exp = state[HIST_EXP];
  subband_analysis(state.subspan(HIST), hist_exp, out, S16(work), exp, acc, acc_exp, CS16(inbuf), hops_i16, taper, interp_coef);
  state[HIST_EXP] = hist_exp;
  return static_cast<u16>(hops);
}

static i32 prev_voicing(CS16 pblk) {
  return static_cast<i32>((static_cast<u32>(static_cast<u16>(pblk[5])) << 16) | static_cast<u32>(static_cast<u16>(pblk[4])));
}
static void put_u32(S16 dst, usz word, u32 v) {
  dst[word] = static_cast<i16>(static_cast<u16>(v));
  dst[word + 1] = static_cast<i16>(static_cast<u16>(v >> 16));
}

u16 encode_frame_params(S16 bits, usz& cursor, S16 cur, S16 prev, CS16 pblk, i16 rate, CS16 peaks, i16 tone_flag, i16 frame_class,
                        S16 cfg, CS16 weights, CS16 tbl, CS16 parts) {
  i16 prev_total = cfg[2];
  {
    A13 t(cfg.data(), 13);
    if (t[0] == 0) imbe_bit_alloc(t);
    else ambe_bit_alloc(t, 0);
  }
  fill16(bits, 0, prev_total);
  cursor = 0;
  prev[0] = frame_class;
  cur[0] = frame_class;
  cur[1] = 0xff;
  if (frame_class == 3) {
    pitch_config_init(cur, rate, peaks[0], peaks[1], 1, tbl);
    if (tone_flag != 0 && rate == 0x20) cur[6] = 0x563b;
  }
  u32 val = static_cast<u16>(lag_to_pitch_index(cur[6]));
  set_pitch_params(cur, static_cast<i16>(static_cast<u16>(val)));
  i16 t = triads_capped(cur[2]);
  u32 nbits = t > 8 ? 8u : static_cast<u32>(static_cast<u16>(t));
  u32 code;
  if (frame_class == 3 && cur[1] != 0xff) {
    u32 mask;
    if (tone_flag == 0 || rate > 0x7a) {
      u16 m0 = two_group_mask(cur, static_cast<i16>(static_cast<u16>(nbits)), 0, tbl, parts);
      u16 m1 = two_group_mask(cur, static_cast<i16>(static_cast<u16>(nbits)), 1, tbl, parts);
      mask = static_cast<u32>(m0 | m1);
    } else {
      u32 sh = nbits & 0x1f;
      mask = static_cast<u32>(wsub<u16>(sh >= 16 ? static_cast<u16>(0) : static_cast<u16>(1u << sh), 1));
    }
    code = wadd<u32>(mask, 0x38u);
    u32 dib = static_cast<u16>(gather_dibits(prev_voicing(pblk)));
    u32 pat = voicing_pattern(static_cast<i16>(static_cast<u16>(code)), dib, static_cast<i16>(static_cast<u16>(nbits)));
    u32 a = 0, b = 0;
    expand_dibits(a, b, pat);
    put_u32(prev, 4, a);
    put_u32(cur, 4, b);
  } else {
    u32 a = 0, b = 0;
    code = quantise_voicing_bL2b1(a, b, prev_voicing(pblk), weights, weights.subspan(0x10), weights.subspan(0x20), static_cast<i32>(nbits));
    put_u32(prev, 4, a);
    put_u32(cur, 4, b);
  }
  i16 nb;
  if (static_cast<i16>(static_cast<u16>(code)) <= 0x38) {
    val = static_cast<i16>(static_cast<u16>(code)) == 0x38 ? 0x19u : (wadd<u32>(code, 0x80u) & 0xffff);
    code = 0;
    set_pitch_params(cur, static_cast<i16>(static_cast<u16>(val)));
    nb = triads_capped(cur[2]);
  } else {
    code = wsub<u32>(code, 0x38u);
    nb = triads_capped(cur[2]);
    if (nb > 8) code = extract_bit_pattern(static_cast<i16>(static_cast<u16>(code)), static_cast<u16>(nb), cur[2]);
  }
  emit_bits(cursor, bits, cfg[5], static_cast<i16>(static_cast<u16>(val)));
  emit_bits(cursor, bits, nb, static_cast<i16>(static_cast<u16>(code)));
  return static_cast<u16>(code);
}

static i16 high16_bL3b3(i32 x) { return static_cast<i16>(x >> 16); }
static i32 top_bit_bL3b3(u32 v) { return v == 0 ? 31 : 31 - __builtin_clz(v); }
static u16 norm_shift_bL3b3(i32 x) {
  if (x == 0) return 0;
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  return static_cast<u16>(0x1e - top_bit_bL3b3(m));
}
static i32 shift32_bL3b3(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f);
}
static i32 abs_sat(i32 x) {
  if (x < 0) return x == INT32_MIN ? INT32_MAX : wneg<i32>(x);
  return x;
}

void slot_analysis(S16 out, S16 weights, S16 clamped, S16 levels, i32 half, i32 base, const SlotCtx& ctx, CS16 cos_tbl) {
  i32 slot_u = static_cast<i32>(wadd<i32>(base, half) & 0xffff);
  i16 slot = static_cast<i16>(slot_u);
  u16 tilt = 0;
  slot_band_weights(weights, clamped, ctx.bands, ctx.mags, ctx.active_mask, slot_u, (slot > 1 ? 1 : 0) + 1, tilt, cos_tbl);
  usz si = static_cast<usz>(slot);
  log_band_levels(levels, ctx.energies.subspan(si * 8), ctx.energy_exp[si], 0);
  i32 p = pitch_select(ctx.pitch_cand, ctx.pitch_err, ctx.pitch_target, slot, static_cast<i16>(wsub<i32>(1, half)));
  out[6] = static_cast<i16>(p);
}

void slot_harmonic_levels(S16 out, i16 step, i16 n, i32 flags, S16 state, i16 offset, CS16 spec, CS16 exps, i16 slot) {
  usz si = static_cast<usz>(static_cast<isz>(static_cast<i32>(slot)));
  harmonic_log_mags(out, step, static_cast<u16>(n), spec.subspan(si * ROW_W), exps[si], offset);
  bias_clamp(out, n);
  if (flags & 1) smooth_peak(out, state[0x40], step);
}

i16 window_complex_block(S16 dst, CS16 src, i32 exp, CS16 win, i32 exp_add, u16 n) {
  i16 e = static_cast<i16>(static_cast<u16>(wadd<i32>(exp_add, exp) & 0xffff));
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 w = win[i];
      dst[2 * i] = high16_bL3b3(wmul<i32>(wmul<i32>(src[2 * i], w), 2));
      dst[2 * i + 1] = high16_bL3b3(wmul<i32>(wmul<i32>(src[2 * i + 1], w), 2));
    }
  }
  i16 sbuf[0x104] = {};
  CS16 scratch = snap(S16(sbuf), CS16(dst).first(2 * static_cast<usz>(n)));
  normalise_to_exp(dst, scratch, wmul<u16>(n, 2), e);
  return e;
}

u16 tone_detect(i16& id, S16 mag, S16 state, std::span<const i32> energy, i16 exp, i16 offset, i16 ref_level, i32 ref_exp, CS16 tbl) {
  i16 latched = state[4];
  i16 band;
  if (latched < 0xa0 || latched >= 0xa4) {
    i16 best = 0;
    pick_best_band(best, 4, energy, tbl, 3);
    band = best;
  } else {
    band = wsub<i16>(latched, 0xa0);
  }
  usz ti = (static_cast<usz>(static_cast<isz>(band)) & 0xffff) * 2;
  i32 e_in = static_cast<u16>(exp);
  i16 e_lo = static_cast<i16>(e_in);
  i16 e_hi = static_cast<i16>(e_in);
  i32 lo = tbl[ti];
  i32 hi = tbl[ti + 1];
  i32 a = accumulate_exp(energy.subspan(static_cast<usz>(lo)), 3, e_lo);
  i32 bb = accumulate_exp(energy.subspan(static_cast<usz>(hi)), 3, e_hi);
  i32 ea = static_cast<u16>(e_lo), eb = static_cast<u16>(e_hi);
  i32 emax = e_lo <= e_hi ? eb : ea;
  i32 ma = static_cast<i16>(shift32_bL3b3(a, static_cast<i16>(ea - emax)) >> 16);
  i32 mb = static_cast<i16>(shift32_bL3b3(bb, static_cast<i16>(eb - emax)) >> 16);
  if (((ma << 16) >> 3) > (mb << 15)) return 0;
  if (((mb << 16) >> 3) > (ma << 15)) return 0;
  i16 e_band = static_cast<i16>(e_in);
  i16 cnt = static_cast<i16>(wadd<u16>(wsub<u16>(static_cast<u16>(tbl[ti + 1]), static_cast<u16>(tbl[ti])), 5));
  i32 c = accumulate_exp(energy.subspan(static_cast<usz>(lo - 1)), cnt, e_band);
  i16 mant = static_cast<i16>(static_cast<u16>(c >> 16));
  i16 ident = static_cast<i16>(wadd<i32>(band, 0xa0));
  i16 m1 = static_cast<i16>(log_amp_clamped(mant, static_cast<u16>(e_band), offset));
  i32 sh = wsub<i32>(static_cast<u16>(e_band), ref_exp);
  i32 scaled = shift32_bL3b3(static_cast<i32>(mant) << 16, static_cast<i16>(sh));
  i32 d = abs_sat(wsub<i32>(static_cast<i32>(ref_level) << 16, scaled));
  u16 n = d == 0 ? static_cast<u16>(0) : norm_shift_bL3b3(d);
  i16 m2 = static_cast<i16>(
      log_amp_clamped(static_cast<i16>(static_cast<i32>(static_cast<u32>(d) << (static_cast<u32>(n) & 0x1f)) >> 16), wsub<i32>(ref_exp, n), offset));
  if (ident == state[4]) {
    m1 = high16_bL3b3(wadd<i32>(wmul<i32>(state[0xb], 0xdffe), wshl<i32>(wadd<i32>(m1, 4), 13)));
    m2 = high16_bL3b3(wadd<i32>(wmul<i32>(state[0xc], 0xdffe), wshl<i32>(wadd<i32>(m2, 4), 13)));
  }
  if (wsub<i32>(wshl<i32>(wsub<i32>(m1, 0x23e0), 16), wshl<i32>(m2, 16)) < 0) return 0;
  if (hi - lo < 4) {
    i16 e_sp = static_cast<i16>(e_in);
    i32 mid = static_cast<i16>((lo + 2 + hi) / 2);
    i16 buf[6] = {};
    block_normalise_to_q15(S16(buf), energy.subspan(static_cast<usz>(mid - 2)), 5, e_sp);
    if (spread_test(mant, static_cast<u16>(e_band), CS16(buf), static_cast<u16>(e_sp), 5)) return 0;
  }
  id = ident;
  i16 v = high16_bL3b3(wsub<i32>(wshl<i32>(m1, 16), 0x03ff'0001));
  mag[0] = v;
  mag[1] = v;
  state[0xb] = m1;
  state[0xc] = m2;
  return 1;
}

static u16 norm_shift_bsr(i32 x) {
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  i32 bsr = m == 0 ? 0 : 31 - __builtin_clz(m);
  return static_cast<u16>(0x1e - bsr);
}

static bool closer_halved(i16 sv, i16 prev) {
  i32 p = static_cast<i32>(prev) << 16;
  i32 a = abs_sat(wsub<i32>((static_cast<i32>(sv) << 16) >> 1, p));
  i32 b = abs_sat(wsub<i32>(static_cast<i32>(sv) << 16, p));
  return a < b;
}

u16 pitch_halve_decision(i16& ratio, i32 pitch, CS16 tail, std::span<const i32> spec, i16 prev_a, i16 level, i16 voiced_run, i16 prev_b,
                         i16 prev_c, i16 prev_d, i16 conf, i16 enable) {
  i16 sv = static_cast<i16>(pitch);
  if (sv > 0x199a) {
    i16 gate = static_cast<i16>(pitch_gate_score(spec, sv, -3));
    if (gate >= 0) {
      ratio = gate;
      return static_cast<u16>(pitch);
    }
  }
  i32 s1 = sum_bands(tail, sv, 6, 0x40);
  u16 sh1 = s1 == 0 ? static_cast<u16>(0) : norm_shift_bsr(s1);
  i32 v1 = static_cast<i32>(static_cast<u32>(s1) << (static_cast<u32>(sh1) & 0x1f));
  u16 u2 = wneg<u16>(sh1);
  i16 m1 = static_cast<i16>(static_cast<u16>(v1 >> 16));
  i16 score = static_cast<i16>(unvoiced_ratio(m1, static_cast<i16>(u2), tail, sv, 6));
  if (sv <= 0xccc) {
    ratio = score;
    return static_cast<u16>(pitch);
  }
  i32 s2 = sum_bands(tail, sv, 5, 0x40);
  u16 sh2 = s2 == 0 ? static_cast<u16>(0) : norm_shift_bsr(s2);
  i32 v2 = static_cast<i32>(static_cast<u32>(s2) << (static_cast<u32>(sh2) & 0x1f));
  u16 u3 = wneg<u16>(sh2);
  i16 m2 = static_cast<i16>(static_cast<u16>(v2 >> 16));
  i32 iv8 = static_cast<i32>(u3) - static_cast<i32>(u2) - 1;
  i32 half = (static_cast<i32>(m1) << 16) >> 1;  // E_cand / 2
  auto decide = [&]() -> bool {
    if (static_cast<i32>(static_cast<i16>(u3)) <= static_cast<i32>(wadd<i16>(static_cast<i16>(u2), 1)) &&
        shl_sat(wmul<i32>(m2, 0x8a60), static_cast<i16>(iv8)) > half) {
      bool fall_through = true;
      if (voiced_run > 2) {
        i16 c1 = high16_bL3b3(wsub<i32>(static_cast<i32>(sv) << 16, wmul<i32>(prev_d, 0xe666)));
        if (c1 >= 0) {
          i16 c2 = high16_bL3b3(wsub<i32>(wmul<i32>(sv, 0xe8ba), static_cast<i32>(prev_d) << 16));
          if (c2 <= 0) fall_through = false;  // straight to the main test
        }
      }
      if (fall_through) {
        i16 d1 = high16_bL3b3(wsub<i32>(static_cast<i32>(sv) << 16, wmul<i32>(prev_b, 0xd99a)));
        if (d1 < 0) return true;
        i16 d2 = high16_bL3b3(wsub<i32>(wmul<i32>(sv, 0xde9c), static_cast<i32>(prev_b) << 16));
        if (d2 > 0) return true;
      }
    }
    if (shl_sat(wmul<i32>(m2, 0xe8ba), static_cast<i16>(iv8)) > half) {
      if ((conf < score || conf < 0x199a) && enable != 0) {
        if (closer_halved(sv, prev_c)) return true;
      } else if (level < 0xccd) {
        if (shl_sat(wmul<i32>(m2, 0xb6dc), static_cast<i16>(iv8)) > half) return true;
      } else {
        i32 t = shl_sat(static_cast<i32>(m2) << 16, static_cast<i16>(iv8));
        i32 thr = static_cast<i32>(high16_bL3b3(wsub<i32>(0x599a'0000, wmul<i32>(level, 0x2666)))) * static_cast<i32>(m1) * 2;
        if (t > thr && closer_halved(sv, prev_a)) return true;
      }
    }
    if (voiced_run < 3) return false;
    if (shl_sat(static_cast<i32>(m2) << 16, static_cast<i16>(iv8)) < half) return false;
    return closer_halved(sv, prev_c);
  };
  if (decide()) {
    ratio = static_cast<i16>(unvoiced_ratio(m2, static_cast<i16>(u3), tail, sv, 5));
    return static_cast<u16>(wadd<i32>((static_cast<i32>(sv) << 16) >> 1, 0x8000) >> 16);
  }
  ratio = score;
  return static_cast<u16>(pitch);
}

void frame_tracker_step(S16 state, std::span<const i32> bands, i32 exp, i16 den) {
  for (usz k = 0; k < 5; k++) state[0x80 - k] = state[0x7f - k];
  usz blk = 0x91;
  usz sc = 0x9b;
  for (int rep = 0; rep < 2; rep++) {
    copy16_batch3(state.subspan(blk), CS16(state).subspan(blk - 8, 8), 8);
    state[sc] = state[sc - 1];
    blk -= 8;
    sc -= 1;
  }
  state[0x25] = static_cast<i16>(log_mean_energy(bands, exp, den));
  i16 gain;
  if (state[0x24] < -0x1400) {
    i16 d = static_cast<i16>(wadd<i32>(state[0x24], 0x1400));
    i16 s = high16_bL3b3(wmul<i32>(d, 0x550a));
    i16 arg = s < -0x352 ? static_cast<i16>(-0x352) : s;
    i16 p = static_cast<i16>(pow2_scaled(arg, 7, 0));
    gain = static_cast<i16>(static_cast<u16>(high16_bL3b3(wmul<i32>(p, 0x8b34))));
  } else {
    gain = 0x459a;
  }
  i16 peak = 0;
  i16 passthru = state[0x27];
  u16 level_u = band_voicing_tracker(state, bands, static_cast<i16>(exp), den, passthru, peak, gain);
  i16 level_s = static_cast<i16>(level_u);
  smooth_level(state, level_s);
  i16 level = level_s < peak ? peak : level_s;
  u16 pen = static_cast<u16>(level_penalty(state, level));
  state[0x7b] = level;
  if (pen == 0) update_smoothed_level(state);
  if (state[0x9c] < 8) state[0x9c] = wadd<i16>(state[0x9c], 1);
  state[0x22] = static_cast<i16>(wmul<i16>(state[0x22], 2) | static_cast<i16>(pen));
}

static i32 shift32_bL3b4(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<u16>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(static_cast<u16>(wneg<i16>(s))) & 0x1f);
}
static i16 shl16_bL3b4(i16 v, i32 count) {
  u32 c = static_cast<u32>(count) & 0x1f;
  return c >= 16 ? static_cast<i16>(0) : static_cast<i16>(static_cast<u16>(static_cast<u16>(v) << c));
}
static i64 allshl_bL3b4(i64 v, u8 count) { return allshl_al2b3(v, count); }  // identical code

i16 windowed_power_spectrum(std::optional<std::span<i32>> mag, CS16 input, i32 exp_sig, CS16 win, i32 exp_win, i16 win_count, i16 out_off,
                            S16 fft, i16 log2n, i16 extra) {
  i16 n = shl16_bL3b4(1, log2n);
  apply_sym_window(fft, input, n, out_off, win, win_count);
  i32 exp = static_cast<u16>(rfft_forward(fft, wadd<i32>(exp_sig, exp_win), log2n, extra));
  if (!mag) return static_cast<i16>(exp);
  i16 count = wadd<i16>(static_cast<i16>(n >> 1), extra);
  cplx_mag_sq_x2(*mag, fft, count);
  return static_cast<i16>(wmul<i32>(exp, 2));
}

i16 quantise_pitch(i16& w0, i16& l, i16 mode, i32 levels_shift, i16 allow_high) {
  if (mode == 1) {
    i16 n = shl16_bL3b4(0x3c, wsub<i32>(levels_shift, 6));
    i16 two_n = static_cast<i16>(wmul<u16>(static_cast<u16>(n), 2));
    i32 lg = log2_q(w0, -4);
    i32 neg = lg == INT32_MIN ? 0x7fff'ffff : -lg;
    i16 hi = static_cast<i16>(static_cast<i32>(static_cast<u32>(neg) << 12) >> 16);
    i16 d = wsub<i16>(hi, 0x44fd);
    i16 idx = static_cast<i16>(div32(wmul<i32>(d, two_n), 0x54a4));
    if (idx < 0) idx = 0;
    else if (idx >= n) idx = static_cast<i16>(wsub<u16>(static_cast<u16>(n), 1));
    i16 cell = static_cast<i16>(wadd<u16>(wmul<u16>(static_cast<u16>(idx), 2), 1));
    i16 recon = static_cast<i16>(div32(wmul<i32>(cell, 0x2a52), two_n));
    i32 log_pitch = wsub<i32>(-0x44fd'0000, static_cast<i32>(recon) << 16) >> 16;
    fundamental_from_log2(w0, l, static_cast<i16>(log_pitch), allow_high);
    return idx;
  }
  if (mode == 2) {
    i16 n = shl16_bL3b4(0x3e, wsub<i32>(levels_shift, 6));
    w0 = 0x4027;
    l = 0x000f;
    return n;
  }
  return 0;
}

i16 cplx_mul_conj_norm(S16 dst, CS16 a, i16 exp_a, CS16 b, i16 exp_b, i16 n) {
  i16 exp = wadd<i16>(wadd<i16>(exp_b, 1), exp_a);
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) {
      i32 b_re = b[2 * i], b_im = b[2 * i + 1], a_re = a[2 * i], a_im = a[2 * i + 1];
      i32 re = wadd<i32>(wmul<i32>(a_re, b_re), wmul<i32>(a_im, b_im)) >> 16;
      i32 im = wsub<i32>(wmul<i32>(a_im, b_re), wmul<i32>(a_re, b_im)) >> 16;
      dst[2 * i] = static_cast<i16>(re);
      dst[2 * i + 1] = static_cast<i16>(im);
    }
  }
  u16 count = wmul<u16>(static_cast<u16>(n), 2);
  i16 sbuf[0x102] = {};
  CS16 src = snap(S16(sbuf), CS16(dst).first(count));
  normalise_to_exp(dst, src, count, exp);
  return exp;
}

u32 dual_peak_gate(i16 gate_a, i16 gate_b, i16 ref_exp, std::span<const i32> table, i16 exp, i16 idx_a, i16 lin_a, i16 idx_b, i16 lin_b,
                   i16 thresh) {
  i16 buf_a[6] = {};
  i16 exp_a = exp;
  block_normalise_to_q15(S16(buf_a), table.subspan(static_cast<usz>(idx_a), 5), 5, exp_a);
  i16 buf_b[6] = {};
  i16 exp_b = exp;
  block_normalise_to_q15(S16(buf_b), table.subspan(static_cast<usz>(idx_b), 5), 5, exp_b);
  if (peak_test(CS16(buf_a), exp_a, 5, gate_a, ref_exp, lin_a, idx_a, thresh) &&
      peak_test(CS16(buf_b), exp_b, 5, gate_b, ref_exp, lin_b, idx_b, thresh)) {
    return 1;
  }
  return 0;
}

static i64 accumulate(std::span<const i32> v, usz base, u16 span) {
  if (static_cast<i16>(span) < 0) return 0;
  i64 acc = 0;
  for (usz k = 0; k <= span; k++) acc = wadd<i64>(acc, static_cast<i64>(v[base + k]));
  return acc;
}

static void band_voicing_measure_bL3b4(S16 measure, std::span<const i32> sig, i32 sig_exp, std::span<const i32> err, i16 err_exp, i16 spacing,
                                       u16 max_idx) {
  i16 e = 0;
  i16 m = div_norm(0x2000'0000, -3, spacing, -3, e);
  i32 half = shift32_bL3b4(static_cast<i32>(m) << 16, wsub<i16>(e, 0xf));
  i32 inc = wmul<i32>(half, 2);
  u16 start = 0;
  i32 edge = 0;
  for (usz slot = 0; slot < 8 && slot < measure.size(); slot++) {
    edge = wadd<i32>(edge, inc);
    i16 rounded = static_cast<i16>(wadd<i32>(edge, 0x8000) >> 16);
    u32 stop = rounded > static_cast<i16>(max_idx) ? static_cast<u32>(max_idx) : static_cast<u32>(static_cast<u16>(rounded));
    u16 span = static_cast<u16>(wsub<u32>(stop, start));
    usz base = static_cast<usz>(static_cast<isz>(static_cast<i16>(start)));
    i64 num = accumulate(err, base, span);
    i16 n1 = static_cast<i16>(norm64(static_cast<u32>(num), static_cast<u32>(num >> 32)));
    i64 s1 = n1 >= 0 ? allshl_bL3b4(num, static_cast<u8>(n1)) : allshr(num, static_cast<u8>(wneg<i16>(n1)));
    i16 num_m = static_cast<i16>(static_cast<i32>(s1) >> 16);
    i16 num_e = static_cast<i16>(static_cast<u16>(wsub<i32>(err_exp, static_cast<u16>(n1))));
    i64 den = accumulate(sig, base, span);
    i16 n2 = static_cast<i16>(norm64(static_cast<u32>(den), static_cast<u32>(den >> 32)));
    i64 s2 = n2 >= 0 ? allshl_bL3b4(den, static_cast<u8>(n2)) : allshr(den, static_cast<u8>(wneg<i16>(n2)));
    i16 den_m = static_cast<i16>(static_cast<i32>(s2) >> 16);
    i16 den_e = static_cast<i16>(wsub<i32>(sig_exp, static_cast<u16>(n2)));
    i16 v = static_cast<i16>(one_minus_ratio(num_m, num_e, den_m, den_e));
    if (v < measure[slot]) measure[slot] = v;
    start = static_cast<u16>(wadd<i32>(wsub<i32>(edge, half), 0x8000) >> 16);
  }
}

void voicing_measure(S16 measure, i16 step, std::span<const i32> spec) {
  i32 full[56] = {};
  i32 half[56] = {};
  i16 exps[55] = {};
  std::span<i32> full_s(full), half_s(half);
  i16 n = band_energy(full_s.subspan(1), S16(exps), step, spec);
  i32 sbuf[55] = {};
  std::span<const i32> src = snap32(std::span<i32>(sbuf), full_s.subspan(1));
  i32 exp_full = static_cast<u16>(align_to_max_exp(full_s.subspan(1), src, CS16(exps), static_cast<u16>(n)));
  band_energy_sum(half_s.subspan(1), S16(exps), step, static_cast<u16>(n), spec);
  src = snap32(std::span<i32>(sbuf), half_s.subspan(1));
  i16 exp_half = static_cast<i16>(static_cast<u16>(align_to_max_exp(half_s.subspan(1), src, CS16(exps), static_cast<u16>(n))));
  band_voicing_measure_bL3b4(measure, std::span<const i32>(full), exp_full, std::span<const i32>(half), exp_half, step, static_cast<u16>(n));
}

static i32 shift32_bL4b1(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<u16>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(static_cast<u16>(wneg<i16>(s))) & 0x1f);
}
static i16 high16_bL4b1(i32 v) { return static_cast<i16>(static_cast<u32>(v) >> 16); }
static i64 shift64(i64 v, i16 s) {
  u32 c = static_cast<u32>(std::abs(static_cast<i32>(s))) & 0x3f;
  if (s < 0) return c >= 64 ? (v >> 63) : (v >> c);
  return c >= 64 ? 0 : static_cast<i64>(static_cast<u64>(v) << c);
}
static u16 norm30(i32 x) {
  if (x == 0) return 0;
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  i32 i = 31;
  if (m != 0) {
    while ((m >> i) == 0) i -= 1;
  }
  return static_cast<u16>(0x1e - i);
}

u32 pitch_track_frame(S16 track, S16 par, CS16 corr, std::span<const i32> spec, i32 unused_p5, i16 first, i32 unused_p7,
                      i16& flag_out) {
  (void)unused_p5;
  (void)unused_p7;
  for (int k = 7; k >= 5; k--) {
    par[static_cast<usz>(k - 3)] = par[static_cast<usz>(k - 5)];
    par[static_cast<usz>(k + 2)] = par[static_cast<usz>(k)];
  }
  if (first == 0) track[6] = 0;
  i16 lo = 4;
  if (track[6] > 2) {
    if (track[2] < 0x2400 || par[2] < 0x2400) {
      if (track[2] > 0x1fff && par[2] > 0x1fff) lo = 5;
    } else {
      lo = 6;
    }
  }
  i16 pitch = refine_pitch_by_harmonics(corr, lo, 0x20);
  if (pitch < 0x3e01) {
    if (pitch < 0x666) pitch = 0x666;
  } else {
    pitch = 0x3e00;
  }
  i16 alt = track[6] < 1 ? track[2] : par[2];
  i16 ratio = par[5];
  u16 ret = pitch_halve_decision(ratio, pitch, corr, spec, track[4], track[5], track[6], track[2], alt, par[2], par[7], first);
  par[5] = ratio;
  par[0] = static_cast<i16>(ret);
  flag_out = 0;
  level_track_update(track, par[0], par[5], par[2], par[7]);
  i16 cur_v = par[5];
  i16 prev_v = par[7];
  if (cur_v > 0x3333 && prev_v < 0x3333) {
    par[6] = prev_v;
    par[1] = par[2];
    return ret;
  }
  if ((prev_v < 0x3334 || cur_v > 0x3332) && (prev_v < 0x199b || (prev_v >> 1) <= cur_v)) {
    par[1] = static_cast<i16>(wadd<i32>(par[2], par[0]) >> 1);
    par[6] = static_cast<i16>(wadd<i32>(cur_v, prev_v) >> 1);
    return (static_cast<u32>(static_cast<u16>(prev_v >> 15)) << 16) | ret;
  }
  par[1] = par[0];
  par[6] = cur_v;
  return ret;
}

void pack_pitch_and_voicing(S16 payload, usz& cursor, i16 rate, S16 cur, S16 prev, i32 unused_p6, i16 tone, CS16 tone_amp, i16 cls,
                            A13 cfg, CS16 weights, std::span<const u32> book, i16 allow_high) {
  (void)unused_p6;
  if (cfg[0] == 0) {
    imbe_bit_alloc(cfg);
  } else {
    i16 sub = static_cast<i16>(wsub<u32>(0x31u, AMBE_RATE_TABLE[static_cast<usz>(rate)]));
    ambe_bit_alloc(cfg, sub);
  }
  fill16(payload, 0, 0x31);
  cursor = 0;
  prev[0] = cls;
  cur[0] = cls;
  if (cls == 3) {
    cur[1] = tone;
    pack_tone_frame(tone, tone_amp, CS16(cfg.data(), 13), cursor, payload);
    return;
  }
  cur[1] = 0xff;
  u32 mask_prev = (static_cast<u32>(static_cast<u16>(prev[5])) << 16) | static_cast<u32>(static_cast<u16>(prev[4]));
  u32 mask_cur = (static_cast<u32>(static_cast<u16>(cur[5])) << 16) | static_cast<u32>(static_cast<u16>(cur[4]));
  u16 vidx = quantise_voicing_bL2b4(cls, mask_prev, mask_cur, weights.first(0x10), weights.subspan(0x10, 0x10), weights.subspan(0x20, 0x10),
                                    book, cfg[4]);
  prev[4] = static_cast<i16>(static_cast<u16>(mask_prev));
  prev[5] = static_cast<i16>(static_cast<u16>(mask_prev >> 16));
  cur[4] = static_cast<i16>(static_cast<u16>(mask_cur));
  cur[5] = static_cast<i16>(static_cast<u16>(mask_cur >> 16));
  i16 code = rate;
  i16 fa = prev[7], fb = cur[7];
  u32 bits = static_cast<u32>(static_cast<i32>(cfg[5]));
  i32 split = split_voiced_freq_code(code, prev, cur, fa, fb, bits, 1);
  if (split == 0) {
    i16 w0 = cur[6];
    i16 l = cur[2];
    code = quantise_pitch(w0, l, cls, cfg[5], allow_high);
    cur[6] = w0;
    cur[2] = l;
  } else {
    prev[6] = 0x1079;
    cur[6] = 0x1079;
    prev[2] = 0x38;
    cur[2] = 0x38;
  }
  emit_bits(cursor, payload, cfg[5], code);
  emit_bits(cursor, payload, cfg[4], static_cast<i16>(vidx));
}

std::optional<SinkCall> slot_amplitudes_and_dispatch(S16 par, S16 state, i16 slot, i16 base, const SlotContext& ctx, CS16 tmpl, CS16 spec,
                                                     CS16 exps) {
  u16 abs_ = wadd<u16>(static_cast<u16>(base), static_cast<u16>(slot));
  if (slot == 0) {
    if (par[0] == 2) {
      i16 n = par[2];
      copy16_batch3(par.subspan(8), tmpl.subspan(1), n);
    } else {
      u32 flags = (static_cast<u32>(static_cast<u16>(par[5])) << 16) | static_cast<u32>(static_cast<u16>(par[4]));
      i16 step = par[6], n = par[2];
      slot_harmonic_levels(par.subspan(8), step, n, static_cast<i32>(flags), state, ctx.level_bias, spec, exps, static_cast<i16>(abs_));
    }
  }
  if (ctx.sink_enable != 0 && par[0] != 3) {
    SinkCall c;
    c.args[0] = ctx.sink_arg0;
    c.args[1] = ctx.sink_d;
    c.args[2] = 0;  // filled in by the caller: the address of `par`
    c.args[3] = ctx.sink_e;
    c.args[4] = ctx.sink_c;
    c.args[5] = ctx.sink_a;
    c.args[6] = ctx.sink_b;
    c.args[7] = static_cast<u16>(slot);
    c.args[8] = abs_;
    c.args[9] = ctx.sink_arg9;
    return c;
  }
  return std::nullopt;
}

i16 subband_power_spectrum(std::span<i32> mag, CS16 hist, CS16 fresh, i16 new_len, CS16 edges, CS16 exps, i32 target, S16 work, CS16 win) {
  i16 keep = static_cast<i16>(wsub<u32>(0x3au, static_cast<u16>(new_len)));
  usz start = static_cast<usz>(wsub<i32>(new_len, 9));
  copy16_batch3(work, hist.subspan(start), keep);
  copy16_batch3(work.subspan(static_cast<u16>(keep)), fresh, new_len);
  for (usz i = 0; i < 6; i++) {
    usz at = static_cast<usz>(static_cast<isz>(edges[i]));
    u16 n = static_cast<u16>(wsub<i16>(edges[i + 1], edges[i]));
    i16 sbuf[0x3a] = {};
    CS16 src = snap(S16(sbuf), CS16(work).subspan(at, n));
    rebase_block_exp(work.subspan(at), src, n, exps[i], target);
  }
  usz at = static_cast<usz>(static_cast<isz>(edges[6]));
  u16 n = static_cast<u16>(wsub<u32>(0x3au, static_cast<u16>(edges[6])));
  i16 sbuf[0x3a] = {};
  CS16 src = snap(S16(sbuf), CS16(work).subspan(at, n));
  rebase_block_exp(work.subspan(at), src, n, exps[6], target);
  i16 ibuf[0x3a] = {};
  CS16 input = snap(S16(ibuf), CS16(work).first(0x3a));
  return windowed_power_spectrum(mag, input, target, win, 0, 0x3a, 0x1d, work, 6, 0);
}

u16 align_phase_and_score_bands(CS16 win, CS16 meas, i16 meas_exp, S16 reference, i16 ref_exp, i16 slope, i16 slope_exp, S16 out) {
  i16 z[2] = {};
  u16 corr_exp = static_cast<u16>(complex_correlate(S16(z), win, meas, meas_exp, reference, ref_exp, 0x7f));
  i16 ang_exp = 0;
  i32 ang = atan2_bfp(z[1], static_cast<i16>(corr_exp), z[0], static_cast<i16>(corr_exp), ang_exp);
  i32 scaled = wmul<i32>(wmul<i32>(static_cast<i16>(ang), slope), 2);
  i16 sh = wsub<i16>(wadd<i16>(ang_exp, slope_exp), 7);
  if (sh < 0) {
    if (sh < -0x1e) out[0] = static_cast<i16>(scaled >> 31);
    else out[0] = high16_bL4b1(scaled >> (static_cast<u32>(static_cast<u16>(wneg<i16>(sh))) & 0x1f));
  } else {
    out[0] = high16_bL4b1(static_cast<i32>(static_cast<u32>(scaled) << (static_cast<u32>(static_cast<u16>(sh)) & 0x1f)));
  }
  if (z[0] < -0x7ffe) z[0] = -0x7fff;
  if (z[1] < -0x7ffe) z[1] = -0x7fff;
  i32 x = z[0];
  i32 y = z[1];
  i64 acc = wadd<i64>(static_cast<i64>(wmul<i32>(wmul<i32>(y, y), 2)), static_cast<i64>(wmul<i32>(wmul<i32>(x, x), 2)));
  u16 n = static_cast<u16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
  i32 shifted = static_cast<i32>(shift64(acc, static_cast<i16>(n)));
  u32 phasor_exp;
  if (shifted < 1) {
    phasor_exp = 0;
  } else {
    i16 e = static_cast<i16>(wsub<u16>(wmul<u16>(corr_exp, 2), n));
    i32 root = sqrt_norm(shifted, e);
    i16 de = 0;
    i16 inv = div_norm(0x4000'0000, 1, high16_bL4b1(root), e, de);
    z[0] = high16_bL4b1(wmul<i32>(wmul<i32>(inv, x), 2));
    z[1] = high16_bL4b1(wmul<i32>(wmul<i32>(inv, y), 2));
    phasor_exp = wadd<u16>(corr_exp, static_cast<u16>(de));
  }
  u16 rot = rotate_spectrum_by_phasor(reference, CS16(z), static_cast<i32>(phasor_exp), ref_exp, 0x7f);
  return band_spectral_match(meas, meas_exp, reference, static_cast<i16>(rot), out);
}

u32 detect_dual_tone(i16& class_out, S16 amps, u16 expected, std::span<const i32> energy, i16 exp, i16 amp_offset, i16 threshold, i32 floor_,
                     const ToneTables& t) {
  bool lo_ok = false;
  bool hi_ok = false;
  u32 best = 0;
  i16 i_lo = 0, j_lo = 0;
  argmax_block_sum(best, i_lo, 4, energy, t.bin_lo_a, 5);
  argmax_block_sum(best, j_lo, 4, energy, t.bin_lo_b, 5);
  i16 a1 = 0, b1 = 0;
  u16 e1 = band_pair_at_common_exp(a1, b1, energy, exp, t.bin_lo_a[static_cast<usz>(i_lo)], t.bin_lo_b[static_cast<usz>(j_lo)]);
  u16 c1 = wadd<u16>(static_cast<u16>(t.class_[static_cast<usz>((static_cast<i32>(i_lo) + static_cast<i32>(j_lo) * 4) & 0xf)]), 0x80);
  if (c1 == expected ||
      static_cast<i16>(dual_peak_gate(a1, b1, static_cast<i16>(e1), energy, exp, t.bin_lo_a[static_cast<usz>(i_lo)], t.lin_lo_a[static_cast<usz>(i_lo)],
                                      t.bin_lo_b[static_cast<usz>(j_lo)], t.lin_lo_b[static_cast<usz>(j_lo)], 0x51ec)) == 1) {
    lo_ok = true;
  }
  i16 i_hi = 0, j_hi = 0;
  argmax_block_sum(best, i_hi, 4, energy, t.bin_hi_a, 5);
  argmax_block_sum(best, j_hi, 4, energy, t.bin_hi_b, 5);
  i16 a2 = 0, b2 = 0;
  u16 e2 = band_pair_at_common_exp(a2, b2, energy, exp, t.bin_hi_a[static_cast<usz>(i_hi)], t.bin_hi_b[static_cast<usz>(j_hi)]);
  u16 c2 = wadd<u16>(static_cast<u16>(t.class_[static_cast<usz>((static_cast<i32>(i_hi) + static_cast<i32>(j_hi) * 4) & 0xf)]), 0x90);
  if (c2 == expected ||
      static_cast<i16>(dual_peak_gate(a2, b2, static_cast<i16>(e2), energy, exp, t.bin_hi_a[static_cast<usz>(i_hi)], t.lin_hi_a[static_cast<usz>(i_hi)],
                                      t.bin_hi_b[static_cast<usz>(j_hi)], t.lin_hi_b[static_cast<usz>(j_hi)], 0x4189)) == 1) {
    hi_ok = true;
  }
  i16 a = a1, b = b1;
  u16 e = e1, c = c1;
  bool adopt_high = false;
  if (!lo_ok) {
    if (!hi_ok) return 0;
    adopt_high = true;
  } else if (hi_ok) {
    i32 s2 = wadd<i32>((static_cast<i32>(a2) << 16) >> 1, (static_cast<i32>(b2) << 16) >> 1);
    u16 n2 = norm30(s2);
    i32 x2 = shift32_bL4b1(s2, static_cast<i16>(n2)) >> 16;
    u16 e2m = wsub<u16>(e2, n2);
    i32 s1 = wadd<i32>((static_cast<i32>(a1) << 16) >> 1, (static_cast<i32>(b1) << 16) >> 1);
    u16 n1 = norm30(s1);
    i32 x1 = shift32_bL4b1(s1, static_cast<i16>(n1)) >> 16;
    if (mant_exp_gt(static_cast<i16>(x2), static_cast<i16>(e2m), static_cast<i16>(x1), static_cast<i16>(wsub<u16>(e1, n1)))) adopt_high = true;
  }
  if (adopt_high) {
    a = a2;
    b = b2;
    e = e2;
    c = c2;
  }
  i32 ia = static_cast<i32>(a) << 16;
  i32 ib = static_cast<i32>(b) << 16;
  if ((ia >> 4) <= wmul<i32>(b, 0xa000) && (ib >> 4) <= wmul<i32>(a, 0xa000)) {
    i32 sh = wadd<i32>(wsub<i32>(e, floor_), 1);
    i32 mean = wadd<i32>(ib >> 1, ia >> 1);
    i32 level = shift32_bL4b1(mean, static_cast<i16>(sh));
    if (wmul<i32>(threshold, 0xf810) <= level) {
      class_out = static_cast<i16>(c);
      amps[0] = static_cast<i16>(log_amp_clamped(a, static_cast<i32>(e), amp_offset));
      amps[1] = static_cast<i16>(log_amp_clamped(b, static_cast<i32>(e), amp_offset));
      return 1;
    }
  }
  return 0;
}

u16 classify_frame_tone(S16 result, std::span<const i32> energy, i16 exp, i16 offset, S16 st, i16 enable, const ToneTables& dual, CS16 bands) {
  i16 e = exp;
  i32 acc = accumulate_exp(energy, 0x80, e);
  i16 ref_level = static_cast<i16>(static_cast<u16>(acc >> 16));
  if (ref_level == 0 || e < -0x1a) return 0xff;
  i32 ref_exp = static_cast<u16>(e);
  i16 cls = 0xff;
  if (static_cast<u16>(detect_dual_tone(cls, result, static_cast<u16>(st[4]), energy, exp, offset, ref_level, ref_exp, dual)) != 1 &&
      static_cast<u16>(detect_signalling_tone(cls, result, st, energy, static_cast<u16>(exp), offset, ref_level, ref_exp, enable)) != 1) {
    tone_detect(cls, result, st, energy, exp, offset, ref_level, ref_exp, bands);
  }
  return static_cast<u16>(cls);
}

static i32 tishift(i32 x, i32 n) {
  if (static_cast<i16>(n) < 0) {
    if (static_cast<i16>(n) > -0x1f) return x >> (static_cast<u32>(wneg<i32>(n)) & 0x1f);
    return x >> 31;
  }
  return static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(n) & 0x1f));
}

void spectral_envelope_frame(S16 st, S16 meas, i32 meas_exp, CS16 power, i32 power_exp, CS16 amp, CS16 lvl, i16 lvl_exp, CS16 cos_tbl) {
  power_exp = wadd<i32>(power_exp, 0x1e);
  meas_exp = wadd<i32>(meas_exp, 0xf);
  st[2] = st[1];
  st[5] = st[4];
  st[8] = st[7];
  i16 row[8];
  std::copy_n(st.begin() + 0x14, 8, row);
  copy16_batch3(st.subspan(0x1c), CS16(row), 8);
  st[0x26] = st[0x25];
  st[1] = st[0];
  st[4] = st[3];
  st[7] = st[6];
  std::copy_n(st.begin() + 0x0c, 8, row);
  copy16_batch3(st.subspan(0x14), CS16(row), 8);
  st[0x25] = st[0x24];
  i16 phas[0x102] = {};
  i16 win[0x102] = {};
  i16 sbuf[0x102] = {};
  u32 e0 = log_spectrum_pairs(S16(phas), power, power_exp, 0x81);
  i32 e = minimum_phase_phasor(S16(phas), static_cast<u16>(e0), 8, 0x100, cos_tbl);
  u16 wexp = log2_vector(S16(win), power, static_cast<i16>(power_exp), 0x81);
  CS16 src = snap(S16(sbuf), CS16(phas));
  i16 ref_exp = window_complex_block(S16(phas), src, static_cast<u16>(e), CS16(win), wexp, 0x81);
  i16 min_ = 0x7fff;
  for (usz i = 0; i < 0x81; i++) {
    if (win[i] != 0 && win[i] < min_) min_ = win[i];
  }
  u16 shift;
  if (wsub<i16>(min_, 1) == 0) {
    shift = 0;
  } else {
    u32 x = static_cast<u32>(wshl<i32>(wsub<i16>(min_, 1), 16));
    u32 mag = static_cast<i32>(x) < 0 ? ~x : x;
    u32 i = 31;
    while ((mag >> i) == 0) i -= 1;
    shift = wsub<u16>(0x0f, static_cast<u16>(wsub<u32>(0x1eu, i)));
  }
  src = snap(S16(sbuf), CS16(win));
  i32 win_exp = reciprocal_vec(S16(win), src, wexp, static_cast<i16>(shift), 0x81);
  src = snap(S16(sbuf), CS16(phas));
  ref_exp = cplx_mul_conj_norm(S16(phas), src, ref_exp, src.subspan(2), ref_exp, 0x80);
  src = snap(S16(sbuf), CS16(meas).first(0x102));
  i16 m = window_complex_block(meas, src, meas_exp, CS16(win), static_cast<u16>(win_exp), 0x81);
  src = snap(S16(sbuf), CS16(meas).first(0x102));
  i16 meas_exp2 = cplx_mul_conj_norm(meas, src, m, src.subspan(2), m, 0x80);
  voiced_band_mask(S16(win), amp, lvl, lvl_exp);
  fill16(S16(win), 0, 0xc);
  u16 cur_exp = align_phase_and_score_bands(CS16(win), CS16(phas), ref_exp, meas, meas_exp2, static_cast<i16>(0xae84u), 6, st);
  st[0x24] = static_cast<i16>(cur_exp);
  if (st[1] > 0 && st[0] < 0 && st[9] >= st[10] && (st[9] != st[10] || st[6] >= st[7])) {
    i16 prev_exp = st[0x25];
    i32 top = static_cast<i16>(cur_exp) > prev_exp ? static_cast<i32>(static_cast<i16>(cur_exp)) : static_cast<i32>(prev_exp);
    i32 base = static_cast<u16>(wadd<i32>(top, 1));
    i32 dc = wsub<i32>(static_cast<i32>(static_cast<i16>(cur_exp)), base);
    i32 dp = wsub<i32>(static_cast<u16>(prev_exp), base);
    i32 dd = wsub<i32>(static_cast<i32>(static_cast<i16>(cur_exp)), static_cast<u16>(prev_exp));
    for (usz k = 0; k < 8; k++) {
      i32 cur = static_cast<i32>(st[0x0c + k]) << 16;
      i32 a = tishift(cur, dc);
      i32 b = tishift(static_cast<i32>(st[0x14 + k]) << 16, dp);
      if (wsub<i32>(a, b) < 0) st[0x14 + k] = static_cast<i16>(tishift(cur, dd) >> 16);
    }
  }
}

void pitch_voicing_frame(S16 st, CS16 corr, CS16 spec, CS16 band_exp, S16 aux, std::span<const i32> energy, i16 energy_exp, i16& flag) {
  i16 first = st[0x48d];
  u32 v = pitch_track_frame(st.subspan(0x30a, 0x311 - 0x30a), st.subspan(0x311), corr, energy, 0, first, 0, flag);
  st[0x48f] = st[0x48e];
  st[0x48e] = st[0x48d];
  st[0x48d] = band_voicing_decision(st.subspan(0x31b), static_cast<i16>(static_cast<u16>(v)), spec, band_exp);
  if (st[0x311] > 0x3759) st[0x311] = static_cast<i16>(st[0x311] >> 1);
  if (st[0x311] > 0x1000 || flag != 0) {
    i16 step = st[0x311];
    voicing_measure(st.subspan(0x31b), step, energy);
    if (flag != 0 && st[0x322] < 0x1999) st[0x322] = 0x1999;
  }
  subband_energy_update(energy, energy_exp, st, aux, flag);
  i16 n = wadd<i16>(static_cast<i16>(popcount_low(static_cast<u32>(static_cast<i32>(st[0x48f])), 8)),
                    static_cast<i16>(popcount_low(static_cast<u32>(static_cast<i32>(st[0x48d])), 8)));
  if ((static_cast<u16>(aux[0x22]) & 0x1c) == 0) n = 0;
  bool voiced = st[0x316] <= 0x2000 || st[0x318] <= 0x2000;
  if (voiced && n >= 4) {
    i16 level = static_cast<i16>(wsub<i32>(wmul<i32>(aux[0x24], 0x550a), 0x0b00'0000) >> 16);
    if (compare_scaled(level, 7, aux[0x29], aux[0x39]) && compare_scaled(aux[0x2b], aux[0x3b], aux[0x29], aux[0x39]) && st[0x31c] < 0x199a) {
      i16 t = static_cast<i16>(wmul<i32>(wmul<i32>(st[0x31b], st[0x31c]), 2) >> 16);
      st[0x31b] = static_cast<i16>(shl_sat(wmul<i32>(t, 0xa000), 3) >> 16);
    }
  } else {
    boost_three_band_gains(st.subspan(0x31b), CS16(aux).subspan(0x91), CS16(aux).subspan(0x29), CS16(aux).subspan(0x39), n);
    i16 l = static_cast<i16>(wsub<i32>(wmul<i32>(aux[0x24], 0x550a), 0x0d00'0000) >> 16);
    if (compare_scaled(l, 7, aux[0x29], aux[0x39]) && aux[0x7d] < 0x1c00 && compare_scaled(aux[0x2b], aux[0x3b], aux[0x29], aux[0x39]) &&
        (static_cast<u8>(static_cast<u16>(st[0x48d])) & 0x7f) == 0 && (static_cast<u8>(static_cast<u16>(st[0x48f])) & 0x7f) == 0 &&
        wsub<i32>(aux[0x24], aux[0x9d]) > 0x1600 && st[0x31b] < 0x1c29) {
      st[0x31b] = 0x1c29;
    }
  }
}

static i32 mul32x16_bL5b1(i32 v, i16 s) {
  i32 lo = wmul<i32>(static_cast<u16>(v), s) >> 15;
  i32 hi = wmul<i32>(wmul<i32>(static_cast<i16>(v >> 16), s), 2);
  return wadd<i32>(lo, hi);
}
static i32 get32_bL5b1(CS16 b, usz i) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(b[2 * i])) | (static_cast<u32>(static_cast<i32>(b[2 * i + 1])) << 16));
}
static void put32_bL5b1(S16 b, usz i, i32 v) {
  b[2 * i] = static_cast<i16>(v);
  b[2 * i + 1] = static_cast<i16>(v >> 16);
}

void subband_spectrum_frame(S16 total, S16 spec, S16 band_exp, CS16 hist, CS16 fresh, i16 new_len, CS16 exps, CS16 edges, CS16 state, CS16 win) {
  i32 target = static_cast<u16>(vec_max(exps, 7));
  i16 level = 0, lexp = 0, vflag = 0;
  i16 gate = static_cast<i16>(static_cast<u16>(speech_activity_gate(state, level, lexp, vflag)));
  i32 lexp32 = static_cast<u16>(lexp);
  i32 flag32 = static_cast<u16>(vflag);
  i32 acc[32] = {};
  u16 acc_exp = 0xffe2;  // -30
  i32 seg = 0;
  usz hist_at = 0;
  usz fresh_at = 0;
  i16 work16[80] = {};
  i32 half1[32] = {};
  for (usz band = 0; band < 8; band++) {
    usz base = band * 16;
    u16 seg_exp[2] = {0, 0};
    i16 gain_m = 0;
    u16 gain_e = 0;
    for (usz half = 0; half < 2; half++) {
      i32 mag[32] = {};
      i16 e = subband_power_spectrum(std::span<i32>(mag), hist.subspan(hist_at), fresh.subspan(fresh_at), new_len, edges, exps, target,
                                     S16(work16), win);
      seg_exp[half] = static_cast<u16>(e);
      mag[0] = 0;
      mag[1] = 0;
      u32 uexp = static_cast<u16>(e);
      i32 weighted[32] = {};
      if (gate == 0) {
        std::copy_n(mag, 32, weighted);
      } else {
        i16 scale;
        u16 bexp;
        if (band == 0) {
          gain_e = static_cast<u16>(log_energy_to_gain(gain_m, state, static_cast<i16>(half)));
          scale = gain_m;
          bexp = gain_e;
        } else {
          i16 s = 0;
          bexp = static_cast<u16>(voicing_threshold(s, state, level, lexp32, static_cast<i16>(seg), flag32));
          scale = s;
        }
        for (usz k = 0; k < 32; k++) weighted[k] = mul32x16_bL5b1(mag[k], scale);
        uexp = wadd<u32>(uexp, static_cast<u32>(bexp));
        if (band == 0) {
          for (usz k = 0; k < 32; k++) mag[k] = mul32x16_bL5b1(mag[k], gain_m);
          seg_exp[half] = static_cast<u16>(wadd<i16>(static_cast<i16>(gain_e), static_cast<i16>(seg_exp[half])));
        }
      }
      if (half == 0) {
        for (usz k = 0; k < 32; k++) put32_bL5b1(spec, base + k, mag[k]);
      } else {
        std::copy_n(mag, 32, half1);
      }
      i32 src[32];
      std::copy_n(acc, 32, src);
      acc_exp = static_cast<u16>(block_add_sat(std::span<i32>(acc), std::span<const i32>(src), std::span<const i32>(weighted),
                                               static_cast<i16>(acc_exp), static_cast<i16>(uexp), 0x20));
      seg += 1;
      hist_at += 49;
      fresh_at += 11;
    }
    i32 blk[32];
    for (usz k = 0; k < 32; k++) blk[k] = get32_bL5b1(spec, base + k);
    i32 src[32];
    std::copy_n(blk, 32, src);
    band_exp[band] = block_add_sat(std::span<i32>(blk), std::span<const i32>(src), std::span<const i32>(half1), static_cast<i16>(seg_exp[0]),
                                   static_cast<i16>(seg_exp[1]), 0x20);
    for (usz k = 0; k < 32; k++) put32_bL5b1(spec, base + k, blk[k]);
    i16 out[32] = {};
    block_normalise_to_q15(S16(out), std::span<const i32>(blk), 0x20, band_exp[band]);
    std::copy_n(out, 32, spec.begin() + static_cast<isz>(band * 32));
  }
  i16 e = static_cast<i16>(acc_exp);
  block_normalise_to_q15(total, std::span<const i32>(acc), 0x20, e);
}

void encode_frame_imbe_spectrum(S16 chan, i32& pos, S16 cur, S16 prev, i16 mask, CS16 alloc, CS16 nib_tbl, CS16 gain_book, CS16 weight,
                                CS16 loading, CS16 band_tbl, CS16 cos_tbl) {
  u16 hbits[56] = {};
  unpack_nibbles(SU16(hbits), cur[2], nib_tbl);
  code_log_mags_with_voicing(prev, cur, mask, CSU16(hbits), chan, pos, 1, gain_book, weight, loading, band_tbl, cos_tbl);
  i16 hbits_i[56];
  for (usz i = 0; i < 56; i++) hbits_i[i] = static_cast<i16>(hbits[i]);
  imbe_frame_deinterleave(chan, cur[2], CS16(hbits_i), alloc);
  cur[0] = 1;
  copy_params_batch2(prev, cur, mask);
}

namespace trk_i {
constexpr usz HOLD = 3, LAST = 4, CLASS = 5, AMPS = 7;
}
namespace sp_i {
constexpr usz EXP = 0x60c / 2, MID = 0x622 / 2, BANDS = 0x636 / 2, BANDS_N = 0x53, LVL = 0x686 / 2, LVL_EXP = 0x6d6 / 2, TRK = 0x6e0 / 2,
              TRK_N = 0x27, TRK_V = 0x2c, FB = 0x72e / 2, FB_EXP = 0x90e / 2;
}

void tone_class_half_frame(S16 trk, std::span<const i32> energy, i16 exp, u32 flags, i16 half, i16 enable, const ToneTables& dual, CS16 bands) {
  usz i = static_cast<usz>(static_cast<isz>(static_cast<i32>(half)));
  usz cls_w = trk_i::CLASS + i;
  usz amp_w = trk_i::AMPS + 2 * i;
  if ((flags & 0x1000) == 0) {
    trk[cls_w] = 0xff;
    trk[amp_w] = 0;
    trk[amp_w + 1] = 0;
    return;
  }
  i16 amps[2] = {trk[amp_w], trk[amp_w + 1]};
  u16 cls = classify_frame_tone(S16(amps), energy, exp, 0, trk, enable, dual, bands);
  if (cls != 0xff) {
    trk[cls_w] = static_cast<i16>(cls);
    trk[amp_w] = amps[0];
    trk[amp_w + 1] = amps[1];
    trk[trk_i::HOLD] = 2;
    trk[trk_i::LAST] = trk[cls_w];
    return;
  }
  i16 last = trk[trk_i::LAST];
  if (last >= 0x80 && trk[trk_i::HOLD] > 0) {
    trk[cls_w] = last;
    trk[trk_i::HOLD] = wsub<i16>(trk[trk_i::HOLD], 1);
    trk[amp_w] = static_cast<i16>(0xb001u);
    trk[amp_w + 1] = static_cast<i16>(0xb001u);
    trk[trk_i::LAST] = trk[cls_w];
    return;
  }
  if ((last == 0x13 || last == 0x3a) && trk[trk_i::HOLD] > 0) {
    trk[cls_w] = last;
    trk[trk_i::HOLD] = wsub<i16>(trk[trk_i::HOLD], 1);
    trk[trk_i::LAST] = trk[cls_w];
    return;
  }
  trk[cls_w] = 0xff;
  trk[amp_w] = 0;
  trk[amp_w + 1] = 0;
  trk[trk_i::LAST] = trk[cls_w];
}

void subband_analysis_step(S16 total, S16 spec, S16 band_exp, std::span<i32> acc, S16 sub, i16 active, CS16 pcm, i16 advance, S16 trk, CS16 taper,
                           CS16 interp_coef, CS16 win) {
  i16 fresh[16 * 11] = {};
  i16 rows = 0;
  u16 hops = subband_analysis_frame(sub, S16(fresh), acc, rows, pcm, 0x17f, advance, taper, interp_coef);
  i32 energy_exp = static_cast<u16>(wadd<i16>(wmul<i16>(sub[0x62c / 2], 2), 7));
  frame_tracker_step(trk, std::span<const i32>(acc), energy_exp, rows);
  if (active > 0) {
    subband_spectrum_frame(total, spec, band_exp, CS16(sub).first(0x310), CS16(fresh), static_cast<i16>(hops), CS16(sub).subspan(0x310, 7),
                           CS16(sub).subspan(0x317, 7), CS16(trk), win);
  }
  shift_band_history(sub, CS16(fresh), static_cast<i16>(hops));
}

void analyse_frame_spectrum(S16 sp, i16 active, CS16 ring, S16 work, i16 advance, CS16 cfg, CS16 cos_tbl, CS16 fb_win, CS16 fb_tbl) {
  i32 power_exp = static_cast<u16>(sp[sp_i::EXP]);
  i32 meas_exp = static_cast<u16>(static_cast<i16>(sp[sp_i::EXP] >> 1));
  {
    i16 amp[8], lvl[8];
    std::copy_n(sp.begin() + sp_i::BANDS, 8, amp);
    std::copy_n(sp.begin() + sp_i::LVL, 8, lvl);
    i16 lvl_exp = sp[sp_i::LVL_EXP];
    i16 trk[sp_i::TRK_N];
    std::copy_n(sp.begin() + sp_i::TRK, sp_i::TRK_N, trk);
    spectral_envelope_frame(S16(trk), work, meas_exp, CS16(sp), power_exp, CS16(amp), CS16(lvl), lvl_exp, cos_tbl);
    std::copy_n(trk, sp_i::TRK_N, sp.begin() + sp_i::TRK);
  }
  if (active > 0) {
    i16 bands[sp_i::BANDS_N], mid[8], out[sp_i::TRK_V];
    std::copy_n(sp.begin() + sp_i::BANDS, sp_i::BANDS_N, bands);
    std::copy_n(sp.begin() + sp_i::MID, 8, mid);
    std::copy_n(sp.begin() + sp_i::TRK, sp_i::TRK_V, out);
    frame_band_voicing(S16(bands), S16(mid), CS16(sp), S16(out), cfg[0x180 / 2]);
    std::copy_n(bands, sp_i::BANDS_N, sp.begin() + sp_i::BANDS);
    std::copy_n(mid, 8, sp.begin() + sp_i::MID);
    std::copy_n(out, sp_i::TRK_V, sp.begin() + sp_i::TRK);
  }
  usz row = sp_i::FB + 2 * 0x50;
  usz ew = sp_i::FB_EXP + 4;
  for (int rep = 0; rep < 2; rep++) {
    std::copy(sp.begin() + (row - 0x50), sp.begin() + row, sp.begin() + row);
    sp[ew] = sp[ew - 2];
    sp[ew + 1] = sp[ew - 1];
    row -= 0x50;
    ew -= 2;
  }
  i16 at = wmul<i16>(wsub<i16>(0x72, advance), 2);
  usz out_w = sp_i::FB;
  usz exp_w = sp_i::FB_EXP;
  for (int rep = 0; rep < 2; rep++) {
    copy16_batch3(work, ring.subspan(static_cast<usz>(static_cast<i32>(at))), 0x50);
    i32 e = subband_filterbank(sp.subspan(out_w), work, 0xf, fb_win, 0x28, fb_tbl);
    sp[exp_w] = static_cast<i16>(e);
    at = wadd<i16>(at, 0x28);
    out_w += 0x28;
    exp_w += 1;
  }
}

std::array<std::optional<SinkCall>, 2> pack_frame(S16 bits, i16 rate, S16 blk0, S16 blk1, S16 pblk, i16 unit, CS16 peaks, i16 tone_flag, CS16 tmpl,
                                                  i16 cls, S16 cfg, const FrameCtx& ctx, i16 allow_high, const PackTables& tbl) {
  i16 base = cfg[1];
  isz bi = base;
  blk1[7] = ctx.env[static_cast<usz>(bi + 1)];
  blk0[7] = ctx.env[static_cast<usz>(bi)];
  i16 locals[48] = {};
  S16 ls(locals);
  if (cls == 1) {
    S16 l60 = ls.subspan(0, 8), l50 = ls.subspan(8, 8), l40 = ls.subspan(16, 8), l30 = ls.subspan(24, 8), l20 = ls.subspan(32, 8),
        l10 = ls.subspan(40, 8);
    slot_analysis(blk1, l50, l10, l30, 1, base, ctx.slot, tbl.blob);
    slot_analysis(blk0, l60, l20, l40, 0, base, ctx.slot, tbl.blob);
  }
  usz cursor = 0;
  u16 code;
  if (cfg[0] == 0) {
    code = encode_frame_params(bits, cursor, blk0, blk1, CS16(pblk), unit, peaks, tone_flag, cls, cfg, CS16(locals), tbl.blob, tbl.partitions);
  } else {
    i16 c13[13];
    std::copy_n(cfg.begin(), 13, c13);
    pack_pitch_and_voicing(bits, cursor, rate, blk0, blk1, 0, unit, peaks, cls, A13(c13), CS16(locals), tbl.voicing_book, allow_high);
    std::copy_n(c13, 13, cfg.begin());
    code = 0;
  }
  std::array<std::optional<SinkCall>, 2> sinks{};
  if (cls != 3) {
    if (ctx.amp.sink_enable != 0 && blk1[0] != 3) {
      SinkCall c;
      c.args[0] = ctx.amp.sink_arg0;
      c.args[1] = ctx.amp.sink_d;
      c.args[2] = 0;  // the address of `blk1`
      c.args[3] = ctx.amp.sink_e;
      c.args[4] = ctx.amp.sink_c;
      c.args[5] = ctx.amp.sink_a;
      c.args[6] = ctx.amp.sink_b;
      c.args[7] = 1;
      c.args[8] = wadd<u16>(static_cast<u16>(base), 1);
      c.args[9] = ctx.amp.sink_arg9;
      sinks[0] = c;
    }
    sinks[1] = slot_amplitudes_and_dispatch(blk0, pblk, 0, base, ctx.amp, tmpl, ctx.spec, ctx.exps);
  }
  if (cfg[0] == 0) {
    i32 pos = static_cast<i32>(cursor);
    encode_frame_imbe_spectrum(bits, pos, blk0, pblk, static_cast<i16>(code), CS16(cfg), tbl.nibbles, tbl.gain64, tbl.hoc_weight, tbl.hoc_loading,
                               tbl.blob, tbl.blob);
    return sinks;
  }
  if (cls != 3) {
    i32 pos = static_cast<i32>(cursor);
    code_gain_and_amplitudes(pblk, blk0, CS16(cfg), bits, pos, 1, tbl.gain32, tbl.block_lens, tbl.prba9, tbl.prba7, tbl.coef_books, tbl.blob);
    frame_bit_permute(bits, CS16(cfg), 0);
  }
  copy_params_batch2(pblk, blk0, 0);
  return sinks;
}

void analyse_frame(S16 sub, S16 st, i16 half, CS16 ring, i16 advance, S16 trk, i16& flag, CS16 taper, CS16 interp, CS16 subband_win, CS16 wide_win) {
  i16 band_exp[8] = {};
  i16 spec[0x120] = {};
  i16 total[32] = {};
  i32 acc[16] = {};
  subband_analysis_step(S16(total), S16(spec), S16(band_exp), std::span<i32>(acc), sub, half, ring, advance, trk, taper, interp, subband_win);
  if (half >= 1) {
    i16 fft_buf[258] = {};
    i32 energy[129] = {};
    i16 exp = analysis_fft(std::span<i32>(energy), ring, wide_win, -7, 0xff, 0, S16(fft_buf), 8, 0);
    energy[128] = static_cast<i32>((static_cast<u32>(static_cast<u16>(fft_buf[257])) << 16) | static_cast<u32>(static_cast<u16>(fft_buf[256])));
    pitch_voicing_frame(st, CS16(total), CS16(spec).first(0x100), CS16(band_exp), trk, std::span<const i32>(energy), exp, flag);
  }
}

i32 rd32(CS16 s, usz w) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(s[w])) | (static_cast<u32>(static_cast<u16>(s[w + 1])) << 16));
}
static void wr32_bL8b1(S16 s, usz w, i32 v) {
  s[w] = static_cast<i16>(v);
  s[w + 1] = static_cast<i16>(v >> 16);
}
static void move_within(S16 s, usz first, usz last, usz dst) {
  std::memmove(s.data() + dst, s.data() + first, (last - first) * sizeof(i16));
}

void front_end(S16 st, CS16 pcm, i16 nsamp, u32 flags, i16 half, const FrontEndTables& t) {
  i16 keep = static_cast<i16>(wsub<u16>(0x17f, static_cast<u16>(nsamp)));
  if (keep > 0) {
    usz src = RING + static_cast<usz>(static_cast<isz>(nsamp));
    move_within(st, src, src + static_cast<usz>(keep), RING);
  }
  usz at = RING + static_cast<usz>(keep);
  if (flags & 0x100) {
    if (nsamp > 0) std::copy_n(pcm.begin(), static_cast<usz>(nsamp), st.begin() + static_cast<isz>(at));
  } else {
    i32 mem[6];
    for (usz k = 0; k < 6; k++) mem[k] = rd32(st, 2 * k);
    usz n = static_cast<usz>(std::max<i16>(nsamp, 0));
    dc_blocker(std::span<i32>(mem), st.subspan(at, n), pcm, static_cast<u16>(nsamp));
    for (usz k = 0; k < 6; k++) wr32_bL8b1(st, 2 * k, mem[k]);
  }
  move_within(st, SPEC_ROW1, SPEC_ROW1 + ROW_W, SPEC_ROW2);
  st[EXP0 + 2] = st[EXP0 + 1];
  move_within(st, SPEC, SPEC + ROW_W, SPEC_ROW1);
  st[EXP0 + 1] = st[EXP0];
  i16 work[258] = {};
  {
    i32 mag[0x81] = {};
    i16 e = analysis_fft(std::span<i32>(mag), CS16(st).subspan(RING + RING_ANALYSIS, 199), t.hamming, -7, 199, 0, S16(work), 8, 1);
    for (usz i = 0; i < 0x81; i++) wr32_bL8b1(st, SPEC + 2 * i, mag[i]);
    st[EXP0] = e;
  }
  static_assert(SPEC + SPEC_W == TRK && TRK + TRK_W == SUB && SUB + SUB_W <= RING, "state layout");
  {
    CS16 ring = CS16(st).subspan(RING, RING_W);
    S16 sp = st.subspan(SPEC, SPEC_W);
    S16 trk = st.subspan(TRK, TRK_W);
    S16 sub = st.subspan(SUB, SUB_W);
    i16 dead = 0;
    analyse_frame(sub, sp, half, ring, nsamp, trk, dead, t.taper, t.interp, t.subband_win, t.wide_win);
    analyse_frame_spectrum(sp, half, ring, S16(work), nsamp, CS16(trk), t.cos, t.fb_win, t.fb_tbl);
  }
  {
    i32 energy[0x81];
    for (usz i = 0; i < 0x81; i++) energy[i] = rd32(st, SPEC + 2 * i);
    i16 exp = st[EXP0];
    i16 enable = st[TONE_ENABLE];
    i16 tone[TONE_TRK_W];
    std::copy_n(st.begin() + TONE_TRK, TONE_TRK_W, tone);
    tone_class_half_frame(S16(tone), std::span<const i32>(energy), exp, flags, half, enable, t.tone, CS16(enc_tbl::TONE_BANDS));
    std::copy_n(tone, TONE_TRK_W, st.begin() + TONE_TRK);
  }
}

std::pair<u16, std::array<std::optional<SinkCall>, 2>> frame_tail(S16 st, S16 bits, i16 rate, u32 flags, i16 p4, i16 gain, S16 blk0, S16 blk1,
                                                                  Bases bases, const PackTables& tbl) {
  flags = st[CFG] == 0 ? (flags & 0xffff'f7ffu) : flags;
  i16 sel = get_word1(CS16(st).subspan(CFG));
  i16 trk_flags = st[TRK + 0x44 / 2];
  bool hit = (wshl<i32>(3, wadd<u32>(static_cast<u16>(sel), 2u) & 0x1f) & static_cast<i32>(trk_flags)) != 0;
  {
    i32 timer = rd32(st, TRK + 0x40 / 2);
    rearm_timer(trk_flags, static_cast<i8>(sel), timer);
    wr32_bL8b1(st, TRK + 0x40 / 2, timer);
  }
  u16 code = hit ? disposition::REARMED : static_cast<u16>(0);
  i16 unit = 0;
  i16 peaks[2] = {};
  i16 tmpl[0x10] = {};
  i16 cls;
  i16 level_bias = 0;
  bool tone = take_signalling_entry(unit, S16(peaks), flags, st.subspan(TONE_TRK, TONE_TRK_W));
  if (tone) {
    cls = 3;
    code |= 0x8002;
    i16 src[2] = {peaks[0], peaks[1]};
    vec_add_scalar(S16(peaks), CS16(src), gain, 2);
  } else if (code == 0 && (flags & 0x800) != 0) {
    cls = 2;
    code = static_cast<u16>(band_level_floor(S16(tmpl), st.subspan(TRK, TRK_W)));
    i16 src[0x10];
    std::copy_n(tmpl, 0x10, src);
    vec_add_scalar(S16(tmpl), CS16(src), gain, 0x10);
  } else {
    cls = 1;
    code |= disposition::ACTIVE;
    level_bias = wadd<i16>(level_bias, gain);
  }
  tone_seq_track(unit, st.subspan(SEQ, SEQ_W), 1);
  CS16 snap_ = CS16(st).first(PBLK);
  S16 pblk = st.subspan(PBLK, CFG - PBLK);
  S16 cfg = st.subspan(CFG, ENC_STATE_WORDS - CFG);
  FrameCtx ctx;
  ctx.slot.bands = snap_.subspan(0x64e / 2);
  ctx.slot.mags = snap_.subspan(0x69e / 2);
  ctx.slot.pitch_cand = snap_.subspan(0x63a / 2, (0x644 - 0x63a) / 2);
  ctx.slot.pitch_err = snap_.subspan(0x644 / 2, (0x64e - 0x644) / 2);
  ctx.slot.pitch_target = snap_[0x630 / 2];
  ctx.slot.energies = snap_.subspan(0x710 / 2);
  ctx.slot.energy_exp = snap_.subspan(0x740 / 2);
  ctx.slot.active_mask = snap_[0x97c / 2];
  ctx.amp.spec_off = wadd<u32>(bases.state, 0x18u);
  ctx.amp.exps_off = wadd<u32>(bases.state, 0x624u);
  ctx.amp.sink_a = wadd<u32>(bases.state, 0x746u);
  ctx.amp.sink_b = wadd<u32>(bases.state, 0x926u);
  ctx.amp.sink_arg0 = static_cast<u16>(snap_[0x97c / 2]);
  ctx.amp.sink_enable = static_cast<i16>((flags >> 6) & 1);
  ctx.amp.sink_c = wadd<u32>(bases.state, 0xa2eu);
  ctx.amp.sink_d = wadd<u32>(bases.state, 0x122eu);
  ctx.amp.sink_e = wadd<u32>(bases.state, 0x938u);
  ctx.amp.sink_fn = bases.sink;
  ctx.amp.level_bias = level_bias;
  ctx.amp.sink_arg9 = static_cast<u16>(p4);
  ctx.env = snap_.subspan(0x6f8 / 2);
  ctx.spec = snap_.subspan(SPEC);
  ctx.exps = snap_.subspan(EXP0);
  auto sinks = pack_frame(bits, rate, blk0, blk1, pblk, unit, CS16(peaks), static_cast<i16>(flags & 0x200), CS16(tmpl), cls, cfg, ctx,
                          static_cast<i16>(flags & 0x80), tbl);
  return {code, sinks};
}

std::pair<u16, std::array<std::optional<SinkCall>, 2>> enc_frame(S16 bits, i16 rate, CS16 pcm, i16 nsamp, u32 flags, i16 half, i16 unit, S16 st,
                                                                 S16 blk0, S16 blk1, Bases bases, const FrontEndTables& fe, const PackTables& tbl) {
  assert(st.size() >= ENC_STATE_WORDS);
  if (nsamp > NSAMP_MAX) nsamp = NSAMP_MAX;
  else if (nsamp < NSAMP_MIN) nsamp = NSAMP_MIN;
  front_end(st, pcm, nsamp, flags, half, fe);
  if ((half & 1) == 0) return {0, {}};
  i16 gain;
  if ((flags & F_AGC_BIAS) == 0) {
    gain = 0;
  } else {
    i32 g = level_follow_gain(CS16(st).subspan(LEVEL), CS16(st).subspan(FOLLOW_CTRL), st.subspan(AGC));
    gain = static_cast<i16>(wmul<i32>(g, GAIN_SCALE) >> 16);
  }
  return frame_tail(st, bits, rate, flags, unit, gain, blk0, blk1, bases, tbl);
}

void band_min_of_max(std::span<i32> out, CS16 cur, CS16 prev, CS16 next, i16 flags, i8 a, i8 b) {
  if ((((static_cast<i32>(flags) >> ((static_cast<i32>(a) + static_cast<i32>(b) + 2) & 0x1f)) & 1)) == 0) next = cur;
  if ((((static_cast<i32>(flags) >> ((static_cast<i32>(b) - static_cast<i32>(a) + 2) & 0x1f)) & 1)) == 0) prev = cur;
  for (usz i = 0; i < 8; i++) {
    i16 x = prev[i], y = next[i], c = cur[i];
    i16 m;
    if (y < x) m = c <= x ? c : x;
    else m = c <= y ? c : y;
    out[i] = static_cast<i32>(m) << 16;
  }
}

bool mant_exp_gt(i16 m1, i16 e1, i16 m2, i16 e2) {
  i32 a = static_cast<i32>(m1) << 16;
  i32 b = static_cast<i32>(m2) << 16;
  i16 ea = a != 0 ? e1 : e2;
  i16 eb = b != 0 ? e2 : ea;
  i16 d = wsub<i16>(eb, ea);
  if (d > 0) {
    d = static_cast<i16>(-d);
    if (d > -0x21) {
      if (d > -1) return b < wshl<i32>(a, static_cast<u32>(static_cast<i32>(d)) & 0x1f);
      if (d > -0x1f) return b < (a >> (static_cast<u32>(-static_cast<i32>(d)) & 0x1f));
    }
    return b < static_cast<i32>(m1) / 0x8001 + (static_cast<i32>(m1) >> 31);
  }
  if (d > -0x21) {
    if (d > -1) return wshl<i32>(b, static_cast<u32>(static_cast<i32>(d)) & 0x1f) < a;
    if (d > -0x1f) return (b >> (static_cast<u32>(-static_cast<i32>(d)) & 0x1f)) < a;
  }
  return static_cast<i32>(m2) / 0x8001 + (static_cast<i32>(m2) >> 31) < a;
}

void acc_add_shifted(std::span<const i32> src, i16 e_src, std::span<i32> acc, i16 e_acc, u16 n) {
  if (static_cast<i16>(n) > 0) {
    i16 sh = wsub<i16>(e_src, e_acc);
    for (usz i = 0; i < n; i++) acc[i] = wadd<i32>(acc[i], shift_i32_batch1(src[i], sh));
  }
}

void dft32_real(S16 buf) {
  i16 t[32] = {};
  t[0] = static_cast<i16>(buf[0] >> 1);
  for (usz k = 0; k < 15; k++) {
    i32 hi = buf[31 - k];
    i32 lo = buf[1 + k];
    t[1 + k] = static_cast<i16>((hi + lo) >> 1);
    t[17 + k] = static_cast<i16>((lo - hi) >> 1);
  }
  i16 mid = buf[16];
  t[16] = static_cast<i16>(mid >> 1);
  i32 sum = 0;
  for (usz i = 0; i < 17; i++) sum = wadd<i32>(sum, t[i]);
  buf[0] = static_cast<i16>(sum >> 3);
  buf[1] = 0;
  const i16* TW = enc_tbl::DFT32_TW;
  for (i16 m = 1; m < 0x10; m++) {
    i64 acc_c = static_cast<i64>(wshl<i32>(t[0], 16));
    i64 acc_s = 0;
    i16 ti = m;
    for (usz j = 0; j < 15; j++) {
      usz k = (static_cast<usz>(static_cast<isz>(ti)) & 0x1f) * 2;
      acc_c = wadd<i64>(acc_c, static_cast<i64>(wmul<i32>(wmul<i32>(TW[k], t[1 + j]), 2)));
      acc_s = wadd<i64>(acc_s, static_cast<i64>(wmul<i32>(wmul<i32>(TW[k + 1], t[17 + j]), 2)));
      ti = static_cast<i16>(wadd<i16>(ti, m) & 0x1f);
    }
    acc_c = wadd<i64>(acc_c, static_cast<i64>(wmul<i32>(wmul<i32>(TW[(static_cast<usz>(static_cast<isz>(ti)) & 0x1f) * 2], t[16]), 2)));
    buf[2 * static_cast<usz>(m)] = static_cast<i16>(acc_c >> 0x13);
    buf[2 * static_cast<usz>(m) + 1] = static_cast<i16>(acc_s >> 0x13);
  }
}

bool spread_test(i16 thr, i32 exp0, CS16 v, i32 exp1, i16 n) {
  u32 s1 = 0;
  if (n > 1) {
    i32 w = 1;
    for (usz i = 1; i < static_cast<u16>(n); i++) {
      i32 term = wmul<i32>(v[i], w);
      w += 1;
      s1 = wadd<u32>(s1, static_cast<u32>(wmul<i32>(term, 2)));
    }
  }
  u32 sh1 = s1 == 0 ? 0u : (static_cast<u32>(norm_shift_batch1(s1)) & 0xffff);
  i32 m1 = static_cast<i16>((s1 << (sh1 & 0x1f)) >> 16);
  i32 sq = wmul<i32>(wmul<i32>(m1, m1), 2);
  u32 s2 = 0;
  if (n > 1) {
    i16 w = 1;
    for (;;) {
      usz i = static_cast<usz>(w);
      i16 ww = wmul<i16>(w, w);
      w = wadd<i16>(w, 1);
      s2 = wadd<u32>(s2, static_cast<u32>(wmul<i32>(wmul<i32>(ww, v[i]), 2)));
      if (w >= n) break;
    }
  }
  u32 sh2 = s2 == 0 ? 0u : (static_cast<u32>(norm_shift_batch1(s2)) & 0xffff);
  i32 m2 = static_cast<i16>((s2 << (sh2 & 0x1f)) >> 16);
  u32 e2 = wadd<u32>(wsub<u32>(static_cast<u32>(exp1), sh2), 0xfu) & 0xffff;
  i32 align = static_cast<i32>(wsub<u32>(wsub<u32>(wmul<u32>(wadd<u32>(wsub<u32>(static_cast<u32>(exp1), sh1), 0xfu) & 0xffff, 2u), e2), static_cast<u32>(exp0)));
  i32 t = thr;
  i32 lhs_sq = shift_i32_batch1(sq, static_cast<i16>(align));
  i32 rhs_exp = static_cast<i32>(wsub<u32>(static_cast<u32>(exp0), e2));
  i32 rhs = shift_i32_batch1(wmul<i32>(wmul<i32>(static_cast<i16>(static_cast<u32>(wmul<i32>(t, 0x999a)) >> 16), t), 2), static_cast<i16>(rhs_exp));
  return wsub<i32>(wmul<i32>(wmul<i32>(m2, t), 2), lhs_sq) < rhs;
}

void argmax_block_sum(u32& best, i16& best_idx, i16 ncand, std::span<const i32> table, CS16 offs, u16 win) {
  i16 idx = 0;
  u32 top = 0;
  i16 out_idx = 0;
  u32 ret = 0;
  i16 k = 0;
  while (k < ncand) {
    i64 acc = 0;
    if (static_cast<i16>(win) > 0) {
      isz base = offs[static_cast<usz>(k)];
      for (isz j = 0; j < static_cast<isz>(win); j++) {
        acc = wadd<i64>(acc, static_cast<i64>(table[static_cast<usz>(base + j)]));
        idx = out_idx;
      }
    }
    u32 v = static_cast<u32>(static_cast<u64>(acc) >> 3);
    if (static_cast<i32>(top) <= static_cast<i32>(v)) {
      top = v;
      idx = k;
      out_idx = k;
    }
    k = static_cast<i16>(k + 1);
    ret = top;
  }
  best = ret;
  best_idx = idx;
}

void level_track_update(S16 st, i16 level, i16 a, i16 ref_level, i16 c) {
  i16 smooth = st[2];
  i16 leak = st[3];
  i16 pub_level = st[4];
  if (a < 0xccd) {
    smooth = static_cast<i16>(static_cast<u32>(wadd<i32>(wmul<i32>(level, 0x4ccc), wmul<i32>(smooth, 0xb334))) >> 16);
  }
  i32 d = wsub<i32>(wshl<i32>(level, 16), wshl<i32>(ref_level, 16));
  i32 mag;
  if (d < 0) mag = static_cast<u32>(d) == 0x8000'0000u ? 0x7fff'ffff : wneg<i32>(d);
  else mag = d;
  i32 thr = wmul<i32>(level, 0x999a) >> 2;
  i16 hang = mag < thr ? wadd<i16>(st[6], 1) : static_cast<i16>(0);
  i16 mant, exp, decayed;
  if (a < 0x199a && c < 0x199a && mag < thr) {
    i16 sh = d == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_shift_batch1(static_cast<u32>(d)));
    mant = static_cast<i16>((static_cast<u32>(d) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f)) >> 16);
    exp = static_cast<i16>(-sh);
    decayed = 0x7fff;
    leak = static_cast<i16>(static_cast<u32>(wadd<i32>(wadd<i32>(wmul<i32>(level, 0xa3d8) >> 5, wmul<i32>(leak, 0xfae2)), 0x8000)) >> 16);
    pub_level = level;
  } else {
    i32 u = wmul<i32>(st[0], 0xcccc);
    i16 sh = u == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_shift_batch1(static_cast<u32>(u)));
    exp = wsub<i16>(st[1], sh);
    mant = static_cast<i16>((static_cast<u32>(u) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f)) >> 16);
    decayed = static_cast<i16>(static_cast<u32>(wmul<i32>(st[5], 0xe666)) >> 16);
  }
  i32 m = mant;
  i32 scaled;
  if (exp < 0) {
    if (exp < -0x1e) scaled = m / 0x8e3a + (m >> 31);
    else scaled = wmul<i32>(m, 0xe666) >> (static_cast<u32>(wneg<i32>(exp)) & 0x1f);
  } else {
    scaled = wshl<i32>(wmul<i32>(m, 0xe666), static_cast<u32>(static_cast<i32>(exp)) & 0x1f);
  }
  i32 acc = sat_add_batch1(wmul<i32>(pub_level, 0xe666), scaled);
  acc = sat_add_batch1(wmul<i32>(leak, 0xcccc) >> 3, acc);
  i16 out = static_cast<i16>(acc >> 16);
  if (out < 0x3e01) {
    if (out < 0x666) out = 0x666;
  } else {
    out = 0x3e00;
  }
  st[4] = out;
  st[5] = decayed;
  st[1] = exp;
  st[2] = smooth;
  st[0] = mant;
  st[3] = leak;
  st[6] = hang;
}

i16 vec_max(CS16 v, i16 n) {
  i16 m = v[0];
  if (n > 1) {
    for (usz i = 1; i < static_cast<u16>(n); i++) {
      if (m < v[i]) m = v[i];
    }
  }
  return m;
}
i16 get_word1(CS16 p) { return p[1]; }
void vec_add_scalar(S16 dst, CS16 src, i16 val, u16 n) {
  if (val != 0 && static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) dst[i] = wadd<i16>(src[i], val);
  }
}

i16 argmax(CS16 a, i16 n) {
  i16 idx = 1;
  i16 max = a[0];
  i16 prev = 0;
  i16 best = 0;
  if (n > 1) {
    usz p = 0;
    for (;;) {
      p += 1;
      best = prev;
      if (max < a[p]) {
        max = a[p];
        best = idx;
      }
      idx = static_cast<i16>(idx + 1);
      prev = best;
      if (!(idx < n)) break;
    }
  }
  return best;
}

bool compare_scaled(i16 p1, i16 p2, i16 p3, i16 p4) {
  i32 iv1 = p1;
  i32 iv2 = wmul<i32>(iv1, 0x10000);
  i16 s5 = p4;
  if (iv2 != 0) s5 = p2;
  i32 iv3 = p3;
  i32 iv4 = wmul<i32>(iv3, 0x10000);
  i16 d = p4;
  if (iv4 == 0) d = s5;
  d = wsub<i16>(d, s5);
  if (d > 0) {
    d = wneg<i16>(d);
    if (d > -0x21) {
      if (d > -1) return wshl<i32>(iv2, static_cast<u32>(static_cast<i32>(d)) & 0x1f) < iv4;
      if (d > -0x1f) return (iv2 >> (static_cast<u32>(-static_cast<i32>(d)) & 0x1f)) < iv4;
    }
    return (iv1 / 0x8001 + (iv1 >> 0x1f)) < iv4;
  }
  if (d > -0x21) {
    if (d > -1) return iv2 < wshl<i32>(iv4, static_cast<u32>(static_cast<i32>(d)) & 0x1f);
    if (d > -0x1f) return iv2 < (iv4 >> (static_cast<u32>(-static_cast<i32>(d)) & 0x1f));
  }
  return iv2 < (iv3 / 0x8001 + (iv3 >> 0x1f));
}

void cos_pairs(S16 out, CS16 src, i16 p3, u16 p4, CS16 table) {
  i16 s6;
  u32 mask;
  if (p3 < 0) {
    s6 = 0;
    mask = 0x7fff'ffffu;
  } else if (p3 != 0) {
    s6 = p3;
    mask = wsub<u32>(1u << (static_cast<u32>(0x1f - static_cast<i32>(static_cast<i8>(p3))) & 0x1f), 1u);
  } else {
    s6 = 0;
    mask = 0x7fff'ffffu;
  }
  if (static_cast<i16>(p4) > 0) {
    i16 sh = static_cast<i16>(p3 - s6);
    usz oi = 0, si = 0;
    for (u16 r = 0; r < p4; r++) {
      i32 iv1 = static_cast<i32>(src[si]) << 16;
      u32 uv2 = shr_or_shl(iv1, sh) & mask;
      i16 s3;
      if (s6 < 0) {
        if (s6 < -0x1e) s3 = static_cast<i16>(static_cast<i32>(uv2) >> 0x1f);
        else s3 = static_cast<i16>(static_cast<u32>(static_cast<i32>(uv2) >> (static_cast<u32>(-static_cast<i32>(s6)) & 0x1f)) >> 16);
      } else {
        s3 = static_cast<i16>(wshl<u32>(uv2, static_cast<u32>(static_cast<i32>(s6)) & 0x1f) >> 16);
      }
      i32 u2 = (static_cast<i32>(s3) << 16) >> 6;
      u32 uv5;
      if (u2 < 0) uv5 = static_cast<u32>(u2) == 0x8000'0000u ? 0x7fff'ffffu : wneg<u32>(static_cast<u32>(u2));
      else uv5 = static_cast<u32>(u2);
      usz idx0 = static_cast<usz>(static_cast<isz>(static_cast<i16>(uv5 >> 16)));
      usz idx1 = static_cast<usz>(wadd<u32>((static_cast<u32>(static_cast<i32>(uv5) >> 16)) & 0xffff, 1u) & 0x1ff);
      out[oi] = static_cast<i16>((static_cast<i32>(table[idx0]) * (0x10000 - static_cast<i32>(uv5 & 0xffff)) +
                                  static_cast<i32>(table[idx1]) * static_cast<i32>(uv5 & 0xffff)) >> 16);
      u2 = wadd<i32>(u2, 0x180'0000);
      u32 uv2b;
      if (u2 < 0) uv2b = static_cast<u32>(u2) == 0x8000'0000u ? 0x7fff'ffffu : wneg<u32>(static_cast<u32>(u2));
      else uv2b = static_cast<u32>(u2);
      usz idx2 = static_cast<usz>((static_cast<u32>(static_cast<i32>(uv2b) >> 16)) & 0x1ff);
      out[oi + 1] = static_cast<i16>((static_cast<i32>(table[idx2]) * (0x10000 - static_cast<i32>(uv2b & 0xffff)) +
                                      static_cast<i32>(table[(idx2 + 1) & 0x1ff]) * static_cast<i32>(uv2b & 0xffff)) >> 16);
      oi += 2;
      si += 2;
    }
  }
}

void pick_best_band(i16& out, i16 p2, std::span<const i32> energy, CS16 pairs, i16 p5) {
  u32 uv6 = 0xffff'ffffu;
  u32 local_20 = 0xffff'ffffu;
  i16 s9 = 0, s4 = 0, s2 = 0;
  if (p2 > 0) {
    i32 iv10 = p5;
    usz pi_ = 0;
    for (;;) {
      i32 iv5 = pairs[pi_];
      u32 local_8 = 0;
      s2 = static_cast<i16>(static_cast<u32>(wadd<i32>(wmul<i32>(pairs[pi_ + 1], 0x10000), wmul<i32>(iv5, -0x10000))) >> 16);
      i32 iv8 = iv10 + s2;
      u32 cnt = static_cast<u32>(wmul<i32>(iv8, 0x10000) >> 16) & 0xffff;
      i32 istk = 0;
      if (static_cast<i16>(iv8) > 0) {
        usz k = static_cast<usz>(iv5);
        for (;;) {
          u32 val = static_cast<u32>(energy[k]);
          k += 1;
          u32 s = local_8 + val;
          u32 carry = s < local_8 ? 1u : 0u;
          local_8 = s;
          istk = wadd<i32>(wadd<i32>(istk, static_cast<i32>(val) >> 0x1f), static_cast<i32>(carry));
          cnt -= 1;
          uv6 = local_20;
          if (cnt == 0) break;
        }
      }
      u32 acc = (local_8 >> 3) | (static_cast<u32>(istk) << 0x1d);
      if (p5 < s2) {
        iv8 = wadd<i32>(wmul<i32>(s2, 0x10000), wmul<i32>(iv10, -0x10000));
        u32 cnt2 = static_cast<u32>(iv8 >> 16) & 0xffff;
        local_8 = 0;
        istk = 0;
        if (static_cast<i16>(static_cast<u32>(iv8) >> 16) > 0) {
          usz k = static_cast<usz>(iv10 + iv5);
          for (;;) {
            u32 val = static_cast<u32>(energy[k]);
            k += 1;
            u32 s = local_8 + val;
            u32 carry = s < local_8 ? 1u : 0u;
            local_8 = s;
            istk = wadd<i32>(wadd<i32>(istk, static_cast<i32>(val) >> 0x1f), static_cast<i32>(carry));
            cnt2 -= 1;
            if (cnt2 == 0) break;
          }
        }
        acc = wsub<u32>(acc, (local_8 >> 3) | (static_cast<u32>(istk) << 0x1d));
        uv6 = local_20;
      }
      if (static_cast<i32>(uv6) <= static_cast<i32>(acc)) {
        uv6 = acc;
        local_20 = acc;
        s9 = s4;
      }
      s4 = static_cast<i16>(s4 + 1);
      pi_ += 2;
      s2 = s9;
      if (!(s4 < p2)) break;
    }
  }
  out = s2;
}

void smooth_peak(S16 arr, i16& state, i16 p3) {
  i16 s2 = state;
  if (p3 < 0x3334) {
    i16 s7 = 0;
    if (p3 < 0x199b) s7 = static_cast<i16>((p3 < 0x1112 ? 1 : 0) + 1);
    i32 iv = wadd<i32>(wmul<i32>(wmul<i32>(static_cast<i32>(s7) + 1, p3), 0x10000), wmul<i32>(s2, -0xcccc));
    for (;;) {
      i16 s4 = static_cast<i16>(static_cast<u32>(iv) >> 16);
      if (s4 < 1) break;
      u32 nxt = static_cast<u32>(wmul<i32>(arr[static_cast<usz>(s7) + 1], 0x10000));
      u32 prod = static_cast<u32>(wmul<i32>(p3, -0x3c00));
      u32 uv5 = wadd<u32>(prod, nxt);
      u32 sat = static_cast<i32>((uv5 ^ prod) & (uv5 ^ nxt)) < 0 ? wadd<u32>(static_cast<i32>(nxt) < 0 ? 1u : 0u, 0x7fff'ffffu) : uv5;
      i16 s3 = static_cast<i16>(sat >> 16);
      if (arr[static_cast<usz>(s7)] < s3) arr[static_cast<usz>(s7)] = s3;
      s7 = static_cast<i16>(s7 - 1);
      iv = wadd<i32>(wmul<i32>(s4, 0x10000), wmul<i32>(p3, -0x10000));
      if (!(s7 >= 0)) break;
    }
  }
  state = static_cast<i16>(static_cast<u32>(wadd<i32>(wmul<i32>(p3, 0xcccc) >> 3, wmul<i32>(s2, 0xe666))) >> 16);
}

bool peak_test(CS16 a, i16 p2, i16 p3, i16 p4, i16 p5, i16 p6, i16 p7, i16 p8) {
  bool result = false;
  if (p4 > 0) {
    i32 acc = 0;
    if (p3 > 1) {
      i32 mult = 1;
      usz idx = 0;
      u16 cnt = static_cast<u16>(p3 - 1);
      for (;;) {
        idx += 1;
        acc = wadd<i32>(acc, wmul<i32>(wmul<i32>(a[idx], mult), 2));
        mult += 1;
        cnt = wsub<u16>(cnt, 1);
        if (cnt == 0) break;
      }
    }
    i16 shift = static_cast<i16>(p2 - p5);
    if (shift < 0) {
      if (shift < -0x1e) acc >>= 0x1f;
      else acc >>= static_cast<u32>(-static_cast<i32>(shift)) & 0x1f;
    } else {
      acc = wshl<i32>(acc, static_cast<u32>(static_cast<i32>(shift)) & 0x1f);
    }
    i32 iv2 = wmul<i32>(wmul<i32>(p6, p4), 2);
    i32 uv5 = iv2 >> 9;
    acc = wsub<i32>(wadd<i32>(acc, wmul<i32>(wmul<i32>(p7, p4), 2)), uv5);
    if (acc < 0) acc = acc == INT32_MIN ? 0x7fff'ffff : -acc;
    i32 t1 = wmul<i32>(static_cast<i32>(static_cast<u32>(uv5) & 0xffff), p8) >> 0xf;
    i32 t2 = wmul<i32>(wmul<i32>(static_cast<i16>(iv2 >> 0x19), p8), 2);
    result = acc < (wadd<i32>(t1, t2) >> 5);
  }
  return result;
}

i16 block_add_sat(std::span<i32> dst, std::span<const i32> a, std::span<const i32> b, i16 p3, i16 p5, u16 n) {
  i16 s1 = p3 <= p5 ? p5 : p3;
  if (static_cast<i16>(n) > 0) {
    i16 sh_a = static_cast<i16>(p3 - s1);
    i16 sh_b = static_cast<i16>(p5 - s1);
    for (usz i = 0; i < n; i++) {
      u32 ua = shr_or_shl(a[i], sh_a);
      u32 ub = shr_or_shl(b[i], sh_b);
      u32 sum = wadd<u32>(ua, ub);
      u32 res = static_cast<i32>((sum ^ ub) & (sum ^ ua)) < 0 ? wadd<u32>(static_cast<i32>(ua) < 0 ? 1u : 0u, 0x7fff'ffffu) : sum;
      dst[i] = static_cast<i32>(res);
    }
  }
  return s1;
}

bool find_active_entry(std::span<const u8> state, i16& out_tag, S16 out_pair) {
  auto rd = [&](usz off) { return static_cast<i16>(static_cast<u16>(state[off]) | (static_cast<u16>(state[off + 1]) << 8)); };
  out_tag = 0xff;
  for (usz i = 0; i < 2; i++) {
    i16 t = rd(10 + i * 2);
    if (t != 0xff) {
      out_tag = t;
      out_pair[0] = rd(0xe + i * 4);
      out_pair[1] = rd(0x10 + i * 4);
      return out_tag != 0xff;
    }
  }
  return out_tag != 0xff;
}

u32 log2_transform(u32 param_1, i16& exp, CS16 tbl) {
  if (param_1 == 0) {
    exp = 0;
    return 0;
  }
  u32 uv = static_cast<i32>(param_1) < 0 ? ~param_1 : param_1;
  i32 i1 = 0x1f;
  if (uv != 0) {
    while ((uv >> i1) == 0) i1 -= 1;
  }
  i32 iv3 = static_cast<i16>((param_1 << (static_cast<u32>(0x1e - i1) & 0x1f)) >> 0x10);
  u32 m1 = wadd<u32>(static_cast<u32>(wmul<i32>(iv3, -0x1f1a)), 0x8000u) & 0xffff'0000u;
  i32 h1 = static_cast<i16>(static_cast<u16>(wadd<u32>(m1, 0x3f26'0000u) >> 16));
  u32 m2 = wadd<u32>(static_cast<u32>(wmul<i32>(wmul<i32>(h1, iv3), 2)), 0x8000u) & 0xffff'0000u;
  i32 h2 = static_cast<i16>(static_cast<u16>(wadd<u32>(m2, 0x9292'0000u) >> 16));
  u32 m3 = wadd<u32>(static_cast<u32>(wmul<i32>(wmul<i32>(h2, iv3), 2)), 0x8000u) & 0xffff'0000u;
  i32 h3 = static_cast<i16>(static_cast<u16>(wadd<u32>(m3, 0x7b83'0000u) >> 16));
  i32 t = wadd<i32>(wmul<i32>(wmul<i32>(h3, iv3), 2), 0x8000);
  u32 uv2 = wadd<u32>(static_cast<u32>(t >> 1) & 0xffff'8000u, 0x2128'8000u);
  u32 uv4_norm;
  if (uv2 == 0) {
    uv4_norm = 0;
  } else {
    u32 u = static_cast<i32>(uv2) < 0 ? ~uv2 : uv2;
    i32 i3 = 0x1f;
    if (u != 0) {
      while ((u >> i3) == 0) i3 -= 1;
    }
    uv4_norm = wsub<u32>(0x1eu, static_cast<u32>(i3)) & 0xffff;
  }
  uv2 = wshl<u32>(uv2, uv4_norm & 0x1f);
  u32 uv4 = wadd<u32>(static_cast<u16>(wsub<i16>(exp, static_cast<i16>(0x1e - i1))), wsub<u32>(1u, uv4_norm));
  i16 s5 = static_cast<i16>(uv4);
  uv4 &= 3;
  if (uv4 != 0) {
    uv2 = static_cast<u32>(wadd<i32>(wmul<i32>(wmul<i32>(tbl[uv4], static_cast<i16>(uv2 >> 16)), 2), 0x8000)) & 0xffff'0000u;
    s5 = static_cast<i16>(s5 + 1);
  }
  exp = static_cast<i16>((s5 + 3) >> 2);
  return uv2;
}

i32 popcount_bits(u32 x, u16 n) {
  i32 count = 0;
  if (static_cast<i16>(n) > 0) {
    u16 u = static_cast<u16>(x);
    for (u16 r = 0; r < n; r++) {
      u = static_cast<u16>(static_cast<i16>(u) >> 1);
      count += static_cast<i32>(x & 1);
      x = u;
    }
  }
  return count;
}

u32 thresh5(i16 p1, i16 p2, i16 p3, i16 p4, i16 p5) {
  if (p3 < 0x199a && wmul<i32>(p3, 0x10000) < ((static_cast<i32>(p2) << 16) >> 1) && wmul<i32>(p3, 0x10000) < ((static_cast<i32>(p1) << 16) >> 1)) {
    i32 v = wsub<i32>(wmul<i32>(p4, 0x10000), (static_cast<i32>(p5) << 16) >> 1);
    if (v < 0) v = v == INT32_MIN ? 0x7fff'ffff : wneg<i32>(v);
    if (v < ((static_cast<i32>(p5) << 16) >> 3)) return 1;
  }
  return 0;
}

static i32 abs_sat32(i32 x) {
  if (x < 0) return x == INT32_MIN ? INT32_MAX : -x;
  return x;
}
static i16 shift_to_i16(i32 v, i16 s) {
  if (s < 0) {
    if (s < -0x1e) return static_cast<i16>(v >> 31);
    return high16_batch3(v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f));
  }
  return high16_batch3(static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f)));
}

u32 band_edges(S16 out, i16 scale, i32 exp, i16 w0, i16 floor_) {
  i32 v = wadd<i32>(wmul<i32>(w0, 0x10000), 0x8000);
  u32 e = v == 0 ? 0u : (wsub<u32>(0x1eu, static_cast<u32>(top_bit_batch3(v < 0 ? ~static_cast<u32>(v) : static_cast<u32>(v)))) & 0xffff);
  i32 norm = static_cast<i32>(static_cast<u32>(v) << (e & 0x1f));
  u32 q = wsub<u32>(0xfu, e) & 0xffff;
  i16 sh = static_cast<i16>(wadd<i32>(wsub<i32>(exp, 0xf), static_cast<i32>(q)));
  i32 k = scale;
  i16 centre = shift_to_i16(wmul<i32>(wmul<i32>(high16_batch3(norm), k), 2), sh);
  out[1] = centre;
  if (centre >= floor_) return 1;
  i16 step = static_cast<i16>(wneg<u32>(q));
  norm = wadd<i32>(norm, wmul<i32>(shift32_al2b2(0x4000'0000, step), -2));
  out[0] = wadd<i16>(shift_to_i16(wmul<i32>(wmul<i32>(high16_batch3(norm), k), 2), sh), 1);
  norm = wadd<i32>(norm, shift32_al2b2(0x2000'0000, step));
  out[2] = wadd<i16>(shift_to_i16(wmul<i32>(wmul<i32>(high16_batch3(norm), k), 2), sh), 1);
  i32 top = wadd<i32>(shift32_al2b2(0x4000'0000, step), norm);
  i16 hi = shift_to_i16(wmul<i32>(wmul<i32>(high16_batch3(top), k), 2), sh);
  out[3] = hi;
  if (out[2] <= hi) return 0;
  return 1;
}

void interp_mid(S16 w, S16 v) {
  i16 w0 = w[0];
  i16 w2 = w[2];
  if (w0 > 0x3333 && w2 < 0x3333) {
    v[1] = v[2];
    w[1] = w[2];
    return;
  }
  if (w0 > 0x1999 && w2 < (w0 >> 1)) {
    v[1] = v[2];
    w[1] = w[2];
    return;
  }
  if ((w2 > 0x3333 && w0 < 0x3333) || (w2 > 0x1999 && w0 < (w2 >> 1))) {
    v[1] = v[0];
    w[1] = w[0];
    return;
  }
  w[1] = high16_batch3(wadd<i32>(wmul<i32>(w2, 0x10000) >> 1, wmul<i32>(w0, 0x10000) >> 1));
  i32 next = wmul<i32>(v[2], 0x10000);
  i32 half_prev = wmul<i32>(v[0], 0x10000) >> 1;
  i32 half_next = next >> 1;
  v[1] = high16_batch3(wadd<i32>(half_next, half_prev));
  if (w[0] < 0xccc) {
    i32 d = abs_sat32(wsub<i32>(wmul<i32>(v[0], 0x10000), half_next));
    if (wadd<i32>(d, wmul<i32>(v[0], -0x3332)) < 0) {
      v[1] = high16_batch3(wadd<i32>(next >> 2, half_prev));
      return;
    }
    i32 d2 = abs_sat32(wsub<i32>(next, half_prev));
    if (wadd<i32>(d2, wmul<i32>(v[0], -0x1998)) < 0) v[1] = high16_batch3(wadd<i32>(next, half_prev));
  }
}

void bias_clamp(S16 v, i16 n) {
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i32 a = wmul<i32>(v[i], 0x10000);
      i32 s = sat_add32(a, 0x00aa'0000);
      i16 r = high16_batch3(s);
      if (r > 0x77ff) r = 0x77ff;
      v[i] = r;
    }
  }
}

void spectrum_harmonic_log_mags(S16 out, CS16 st, i16 step, i16 n, i16 offset) {
  harmonic_log_mags(out, step, static_cast<u16>(n), st.subspan(SPEC), st[EXP0], offset);
  bias_clamp(out, n);
}

void rearm_timer(i16 flags, i8 sel, i32& timer) {
  i32 mask = wshl<i32>(3, static_cast<u32>(static_cast<i32>(sel) + 2) & 0x1f);
  if ((mask & static_cast<i32>(flags)) != 0) timer = 0xfa000;
}

i32 div32_sat15(i32 num, i16 den) {
  i32 sign = static_cast<i32>(den) ^ num;
  i32 n = num < 0 ? (num == INT32_MIN ? INT32_MAX : -num) : num;
  i32 d = den < 0 ? (den == INT16_MIN ? static_cast<i32>(INT16_MAX) : static_cast<i32>(-den)) : static_cast<i32>(den);
  i32 q;
  if (n < d) q = 0;
  else if ((n >> 16) < d) q = (n / d) >> 1;
  else q = 0x7fff;
  return sign < 0 ? -q : q;
}

i32 sum_i16(CS16 x, i16 n) {
  i32 acc = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) acc = wadd<i32>(acc, x[i]);
  }
  return acc;
}

i32 i32_block_exp(std::span<const i32> x, i16 n) {
  i32 max = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i32 v = x[i];
      if (v < 0) v = v == INT32_MIN ? INT32_MAX : wneg<i32>(v);
      if (max < v) max = v;
    }
  }
  if (max == 0) return 0;
  return -static_cast<i32>(static_cast<u16>(norm_shift_batch4(max)));
}

void voiced_scale8(std::span<i32> dst, CS16 src, CS16 aux, i16 vmask, i16 a, i16 b) {
  u32 sh1 = static_cast<u16>(wadd<i16>(wadd<i16>(a, 2), b)) & 0x1f;
  u32 sh2 = static_cast<u16>(wadd<i16>(b, 2)) & 0x1f;
  i32 m = vmask;
  bool gated = ((m >> sh1) & 1) != 0 && ((m >> sh2) & 1) != 0;
  for (usz i = 0; i < 8; i++) {
    i32 s = src[i];
    i32 v;
    if (!gated) {
      v = s << 16;
    } else {
      i16 t = aux[i];
      if (t < 0x1127) v = wmul<i32>(s, 0xab86);
      else if (t > 0x1999) v = s << 16;
      else v = wmul<i32>(wmul<i32>(wmul<i16>(t, 5), s), 2);
    }
    dst[i] = v;
  }
}

void cplx_mag_sq_x2(std::span<i32> dst, CS16 src, i16 n) {
  if (n <= 0) return;
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    i32 re = src[2 * i];
    i32 im = src[2 * i + 1];
    dst[i] = wmul<i32>(wadd<i32>(wmul<i32>(re, re), wmul<i32>(im, im)), 2);
  }
}

void interp_filter(S16 out, CS16 coef, CS16 src, i16 n, i16 off) {
  i32 c[4] = {coef[0], coef[1], coef[2], coef[3]};
  static const std::pair<isz, usz> taps[7] = {{-64, 0}, {-48, 1}, {-32, 2}, {-16, 3}, {32, 0}, {16, 1}, {0, 2}};
  isz p = off;
  for (isz row = 0; row < 16; row++) {
    if (n > 0) {
      for (isz col = 0; col < n; col++) {
        isz b = 64 + row + 32 * col;
        i64 sum = 0;
        for (const auto& [d, ci] : taps) {
          i32 prod = wmul<i32>(wmul<i32>(src[static_cast<usz>(b + d)], c[ci]), 2);
          sum += static_cast<i64>(prod);
        }
        out[static_cast<usz>(p)] = static_cast<i16>((sum + 0x10000) >> 17);
        p++;
      }
    }
    p += static_cast<isz>(wsub<i16>(0xb, n));
  }
}

i16 argmax_i32(std::span<const i32> x, i16 n) {
  i16 best = 0;
  if (n > 1) {
    i32 cur = x[0];
    for (i16 i = 1; i < n; i = static_cast<i16>(i + 1)) {
      i32 v = x[static_cast<usz>(i)];
      if (cur < v) {
        cur = v;
        best = i;
      }
    }
  }
  return best;
}

void band_floor(CS16 state, CS16 src, S16 cur, S16 alt, CS16 levels, i16 mode) {
  constexpr i32 Q = static_cast<i32>(0x6666'0000u);
  bool gate = state[7] < 0xccc;
  for (usz i = 0; i < 8; i++) {
    if (mode != 0 && i >= 5) {
      i16 v = static_cast<i16>(shl_sat(Q, wsub<i16>(-3, levels[2])) >> 16);
      if (alt[i] < v) alt[i] = v;
      v = static_cast<i16>(shl_sat(Q, wsub<i16>(-3, levels[1])) >> 16);
      if (cur[i] < v) cur[i] = v;
    } else {
      i16 v0 = static_cast<i16>(shl_sat(Q, wsub<i16>(-3, levels[0])) >> 16);
      bool ok = gate && src[i] > v0;
      i16 v1 = static_cast<i16>(shl_sat(Q, wsub<i16>(-3, levels[1])) >> 16);
      if (cur[i] < v1 && ok) cur[i] = v1;
    }
  }
}

i16 update_pitch_track(i16& val, u16& exp, i16& conf, i16& age, i16 add, i16 add_exp, i16 passthru, i16 k) {
  i32 m = wmul<i32>(wmul<i32>(val, k), 2);
  u16 n = m == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m));
  i32 e = wadd<i32>(exp, wsub<i32>(0xf, n));
  i16 oe = 0;
  i16 r = add_norm(static_cast<i16>(static_cast<i32>(static_cast<u32>(m) << (static_cast<u32>(n) & 0x1f)) >> 16), static_cast<i16>(e), add, add_exp, oe);
  i32 m2 = (static_cast<i32>(k) + 1) << 16;
  u16 n2 = m2 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m2));
  i16 oe2 = oe;
  i16 q = div_norm(static_cast<i32>(r) << 16, oe, static_cast<i16>(static_cast<i32>(static_cast<u32>(m2) << (static_cast<u32>(n2) & 0x1f)) >> 16),
                   static_cast<i16>(wsub<i32>(0xf, n2)), oe2);
  val = q;
  exp = static_cast<u16>(oe2);
  conf = 0x7fff;
  age = 5;
  return passthru;
}

void update_ratio_accum(i16 x, S16 st) {
  i32 p = x;
  i32 m1 = wmul<i32>(wmul<i32>(st[2], p), 2);
  u16 n1 = m1 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m1));
  i16 ex = wadd<i16>(wsub<i16>(st[3], static_cast<i16>(n1)), 0xf);
  i16 v = static_cast<i16>(static_cast<i32>(static_cast<u32>(m1) << (static_cast<u32>(n1) & 0x1f)) >> 16);
  i32 m2 = wmul<i32>(wmul<i32>(st[4], v), 2);
  u16 n2 = m2 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m2));
  i16 e2 = wadd<i16>(wsub<i16>(st[5], static_cast<i16>(n2)), ex);
  i16 mant2 = static_cast<i16>(static_cast<i32>(static_cast<u32>(m2) << (static_cast<u32>(n2) & 0x1f)) >> 16);
  i16 e7 = st[7];
  i16 r = add_norm(st[6], st[7], mant2, e2, e7);
  st[7] = e7;
  st[6] = r;
  i32 m3 = wmul<i32>(wmul<i32>(v, p), 2);
  u16 n3 = m3 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m3));
  i16 e3 = static_cast<i16>(wadd<i32>(wsub<i32>(static_cast<u16>(ex), n3), 0xf));
  i16 mant3 = static_cast<i16>(static_cast<i32>(static_cast<u32>(m3) << (static_cast<u32>(n3) & 0x1f)) >> 16);
  i16 e9 = st[9];
  i16 den = add_norm(st[8], st[9], mant3, e3, e9);
  st[9] = e9;
  st[8] = den;
  if (den > 0) {
    i16 e1 = st[1];
    i16 q = div_norm(static_cast<i32>(st[6]) << 16, st[7], den, st[9], e1);
    st[1] = e1;
    st[0] = q;
  }
}

static i16 mul_round(i16 a, i16 b) { return static_cast<i16>(wadd<i32>(wmul<i32>(wmul<i32>(a, b), 2), 0x8000) >> 16); }

void apply_sym_window(S16 out, CS16 input, i16 len, i16 out_off, CS16 win, i16 count) {
  isz h = static_cast<i16>(static_cast<u16>(count >> 1));
  isz fwd, bwd;
  if (out_off == 0) {
    fwd = 0;
    bwd = static_cast<isz>(len) - 1;
  } else {
    fwd = out_off;
    bwd = static_cast<isz>(out_off) - 1;
  }
  isz in_fwd = h;
  if (count & 1) {
    out[static_cast<usz>(fwd)] = mul_round(input[static_cast<usz>(h)], win[static_cast<usz>(h)]);
    in_fwd = h + 1;
    fwd += 1;
  }
  if (h > 0) {
    for (isz j = 0; j < h; j++) {
      i16 c = win[static_cast<usz>(h - 1 - j)];
      out[static_cast<usz>(fwd)] = mul_round(input[static_cast<usz>(in_fwd + j)], c);
      fwd += 1;
      out[static_cast<usz>(bwd)] = mul_round(input[static_cast<usz>(h - 1 - j)], c);
      bwd -= 1;
    }
  }
  fill16(out.subspan(static_cast<usz>(fwd)), 0, wsub<i16>(len, count));
}

i32 pitch_select(CS16 cand, CS16 err, i16 target, i16 idx, i16 alt) {
  auto choose = [&]() -> i16 {
    if (idx >= 1) {
      usz i = static_cast<usz>(idx);
      if (thresh5(err[i + 1], err[i], err[i - 1], cand[i], cand[i - 1]) != 0) {
        return static_cast<i16>((static_cast<i32>(cand[i]) << 17) >> 16);
      }
    }
    usz i = static_cast<usz>(idx);
    if (err[i] > 0x3333) {
      i16 j = alt;
      if (err[static_cast<usz>(alt)] >= 0x3333) {
        j = 0;
        for (i16 k = 1; k < 5; k++) {
          if (err[static_cast<usz>(k)] < err[static_cast<usz>(j)]) j = k;
        }
      }
      if (err[static_cast<usz>(j)] >= 0x3333) {
        i16 best = 0;
        for (i16 k = 1; k < 5; k++) {
          i32 dk = std::abs(wsub<i32>(cand[static_cast<usz>(k)], target));
          i32 db = std::abs(wsub<i32>(cand[static_cast<usz>(best)], target));
          if (dk < db) best = k;
        }
        return cand[static_cast<usz>(best)];
      }
      return cand[static_cast<usz>(j)];
    }
    return cand[i];
  };
  i16 chosen = choose();
  return shl_sat(static_cast<i32>(chosen) << 16, 1) >> 16;
}

void smooth_level(S16 state, i16 level) {
  i16 cur = state[0x25];
  i16 prev = state[0x24];
  i32 m1, ebase, m2;
  if (cur > prev) {
    if (level > 0x4000 && cur > -0x2800) {
      m1 = wmul<i32>(cur, 0xcccc);
      ebase = 4;
      m2 = wmul<i32>(prev, 0xe666);
    } else {
      m1 = wmul<i32>(cur, 0xa3d8);
      ebase = 1;
      m2 = wmul<i32>(prev, 0xfd70);
    }
  } else if (level > 0x2000) {
    m1 = wmul<i32>(cur, 0xa3d8);
    ebase = 1;
    m2 = wmul<i32>(prev, 0xfd70);
  } else if (prev <= -0x2800) {
    state[0x24] = prev;
    return;
  } else {
    m1 = wmul<i32>(cur, 0x8312);
    ebase = -2;
    m2 = wmul<i32>(prev, 0xffbe);
  }
  u16 n1 = m1 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m1));
  i16 e1 = static_cast<i16>(wsub<i32>(ebase, n1));
  i16 a = static_cast<i16>(static_cast<i32>(static_cast<u32>(m1) << (static_cast<u32>(n1) & 0x1f)) >> 16);
  u16 n2 = m2 == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(m2));
  i16 b = static_cast<i16>(static_cast<i32>(static_cast<u32>(m2) << (static_cast<u32>(n2) & 0x1f)) >> 16);
  i16 e = 0;
  i16 r = add_norm(a, e1, b, static_cast<i16>(wsub<i32>(7, n2)), e);
  state[0x24] = static_cast<i16>(shift32_al3b2(static_cast<i32>(r) << 16, wsub<i16>(e, 7)) >> 16);
}

u32 align_to_max_exp(std::span<i32> dst, std::span<const i32> src, CS16 exps, u16 n) {
  i16 mx = vec_max(exps, static_cast<i16>(n));
  i32 last = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 v = shift32_al3b2(src[i], wsub<i16>(exps[i], mx));
      dst[i] = v;
      last = v;
    }
  }
  return (static_cast<u32>(last) & 0xffff'0000u) | static_cast<u32>(static_cast<u16>(mx));
}

static i64 shift64_l1_batch1(i64 v, i16 c);

i16 band_energy(std::span<i32> mags, S16 exps, i16 step, std::span<const i32> spec) {
  i32 inc = (static_cast<i32>(step) << 9) >> 3;
  i32 a = wadd<i32>((static_cast<i32>(step) << 9) >> 4, 0x8000);
  i32 b = wadd<i32>(a, inc);
  i16 k = 0;
  while (static_cast<i16>(b >> 16) < 0x80) {
    i16 i0 = static_cast<i16>(static_cast<u16>(static_cast<u32>(a) >> 16));
    u16 i1 = static_cast<u16>(static_cast<u32>(b) >> 16);
    i16 cnt = static_cast<i16>(wsub<u16>(i1, static_cast<u16>(i0)));
    i32 f0 = static_cast<i32>((static_cast<u32>(a) & 0xffff) >> 1);
    i32 v = spec[static_cast<usz>(i0)];
    i32 p = wadd<i32>(wmul<i32>(static_cast<i32>(static_cast<u32>(v) & 0xffff), f0) >> 15, wmul<i32>(wmul<i32>(static_cast<i16>(v >> 16), f0), 2));
    p = p == INT32_MIN ? INT32_MAX : wneg<i32>(p);
    i64 acc = static_cast<i64>(wadd<i32>(v, p));
    i32 f1 = static_cast<i32>((static_cast<u32>(b) >> 1) & 0x7fff);
    i32 v2 = spec[static_cast<usz>(i0) + static_cast<usz>(static_cast<u16>(cnt))];
    i32 q = wadd<i32>(wmul<i32>(static_cast<i32>(static_cast<u32>(v2) & 0xffff), f1) >> 15, wmul<i32>(wmul<i32>(static_cast<i16>(v2 >> 16), f1), 2));
    acc = wadd<i64>(acc, static_cast<i64>(q));
    if (cnt > 1) {
      for (usz j = 1; j < static_cast<usz>(cnt); j++) acc = wadd<i64>(acc, static_cast<i64>(spec[static_cast<usz>(i0) + j]));
    }
    i16 s = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
    i64 sh = shift64_l1_batch1(acc, s);
    exps[static_cast<usz>(k)] = wneg<i16>(s);
    mags[static_cast<usz>(k)] = static_cast<i32>(sh);
    k = wadd<i16>(k, 1);
    a = b;
    b = wadd<i32>(b, inc);
  }
  return k;
}

u32 complex_correlate(S16 out, CS16 w, CS16 a, i16 exp_a, CS16 b, i16 exp_b, u16 n) {
  i64 acc1 = 0, acc2 = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i16 ar = a[2 * i], ai = a[2 * i + 1];
      i16 br = b[2 * i], bi = b[2 * i + 1];
      i32 wi = w[i];
      i32 x = static_cast<i16>((wmul<i32>(ar, wi) << 2) >> 16);
      i32 y = static_cast<i16>((wmul<i32>(ai, wi) << 2) >> 16);
      i32 t1 = wmul<i32>(wmul<i32>(bi, y), 2);
      i32 t2 = wmul<i32>(wmul<i32>(br, x), 2);
      acc1 = wadd<i64>(acc1, wadd<i64>(static_cast<i64>(t1), static_cast<i64>(t2)));
      i32 u1 = wmul<i32>(wmul<i32>(br, y), 2);
      i32 u2 = wmul<i32>(wmul<i32>(bi, x), 2);
      acc2 = wadd<i64>(acc2, wsub<i64>(static_cast<i64>(u1), static_cast<i64>(u2)));
    }
  }
  i16 s1 = static_cast<i16>(norm64(static_cast<u32>(acc1), static_cast<u32>(acc1 >> 32)));
  i64 sh1 = shift64_l1_batch1(acc1, s1);
  i16 e1 = static_cast<i16>(wadd<i32>(wsub<i32>(exp_a, static_cast<u16>(s1)), exp_b));
  i16 s2 = static_cast<i16>(norm64(static_cast<u32>(acc2), static_cast<u32>(acc2 >> 32)));
  i64 sh2 = shift64_l1_batch1(acc2, s2);
  i16 e2 = static_cast<i16>(wadd<i32>(wsub<i32>(exp_a, static_cast<u16>(s2)), exp_b));
  i16 mx = e1 > e2 ? e1 : e2;
  i64 r1 = shift64_l1_batch1(sh1, wsub<i16>(e1, mx));
  i64 r2 = shift64_l1_batch1(sh2, wsub<i16>(e2, mx));
  out[0] = static_cast<i16>(static_cast<i32>(r1) >> 16);
  out[1] = static_cast<i16>(static_cast<i32>(r2) >> 16);
  return (static_cast<u32>(static_cast<i16>(static_cast<i32>(r2) >> 16)) & 0xffff'0000u) | static_cast<u32>(static_cast<u16>(mx));
}

void shift_band_history(S16 buf, CS16 src, i16 n) {
  i16 keep = wsub<i16>(0x31, n);
  usz row = 0, s = 0;
  for (int r = 0; r < 16; r++) {
    if (keep > 0) {
      usz k = static_cast<usz>(keep);
      usz off = static_cast<usz>(n);
      move_within(buf, row + off, row + off + k, row);
    }
    copy16_batch3(buf.subspan(row + (0x31 - static_cast<usz>(n))), src.subspan(s), n);
    row += 0x31;
    s += 0xb;
  }
}

static i64 allshl_l1_batch1(i64 v, u8 count) { return allshl_al2b3(v, count); }  // identical code
static i64 shift64_l1_batch1(i64 v, i16 c) {
  if (c < 0) return allshr(v, static_cast<u8>(wneg<i16>(c)));
  return allshl_l1_batch1(v, static_cast<u8>(c));
}

static i32 shift_i32_l1_batch2(i32 v, i16 sh) {
  if (sh < 0) {
    if (sh < -0x1e) return v >> 31;
    return v >> (static_cast<u32>(wneg<i32>(sh)) & 0x1f);
  }
  return wshl<i32>(v, static_cast<u32>(static_cast<i32>(sh)) & 0x1f);
}

i16 band_energy_walk(i16& exp_out, std::span<const i32> table, i32 num, i16 den, i16 sh, i32& start_out) {
  u16 q = static_cast<u16>(div32(num, den));
  i32 width = shift_i32_l1_batch2(static_cast<i32>(den) << 9, sh);
  i32 half = width >> 1;
  i32 quarter = width >> 2;
  i32 acc32 = wadd<i32>(quarter == INT32_MIN ? INT32_MAX : wneg<i32>(quarter), 0x10000);
  if (static_cast<i16>(q) > 0) acc32 = wadd<i32>(acc32, wmul<i32>(static_cast<i32>(q), width));
  u16 pos = static_cast<u16>(high16_al2b1(acc32));
  start_out = acc32;
  acc32 = wadd<i32>(acc32, half);
  u16 next = static_cast<u16>(high16_al2b1(acc32));
  u16 span = wsub<u16>(next, pos);
  u32 lo = 0;
  i32 hi = 0;
  while (static_cast<i16>(next) <= 0x80) {
    if (static_cast<i16>(span) > 0) {
      usz base = static_cast<usz>(static_cast<isz>(static_cast<i16>(pos)));
      for (usz k = 0; k < span; k++) acc64(lo, hi, static_cast<u32>(table[base + k]));
    }
    i32 mid = wadd<i32>(acc32, half);
    acc32 = wadd<i32>(mid, half);
    pos = static_cast<u16>(high16_al2b1(mid));
    next = static_cast<u16>(high16_al2b1(acc32));
    span = wsub<u16>(next, pos);
  }
  if (static_cast<i16>(pos) < 0x80) {
    u16 cnt = wsub<u16>(0x80, pos);
    if (static_cast<i16>(cnt) > 0) {
      usz base = static_cast<usz>(static_cast<isz>(static_cast<i16>(pos)));
      for (usz k = 0; k < cnt; k++) acc64(lo, hi, static_cast<u32>(table[base + k]));
    }
  }
  auto [s, low] = norm64_align(lo, static_cast<u32>(hi));
  exp_out = wneg<i16>(s);
  return high16_al2b1(low);
}

u32 voicing_codebook_search(CS16 w_a, CS16 w_b, CS16 w_c, std::span<const u32> book, u8 nbits, i16 mode) {
  i16 t_unv[16] = {}, t_mid[16] = {}, t_voi[16] = {};
  for (usz k = 0; k < 16; k++) {
    usz j = 15 - k;
    i32 a = w_a[j];
    i32 b = w_b[j];
    i32 a2 = high16_al2b1(wmul<i32>(wmul<i32>(a, a), 2));
    i32 na = static_cast<i16>(0x7fff - a);
    i32 na2 = high16_al2b1(wmul<i32>(wmul<i32>(na, na), 2));
    i32 b2 = high16_al2b1(wmul<i32>(wmul<i32>(b, b), 2));
    i32 ia2 = static_cast<i16>(0x7fff - a2);
    i32 nb = static_cast<i16>(0x7fff - b);
    i32 nb2 = high16_al2b1(wmul<i32>(wmul<i32>(nb, nb), 2));
    i32 ina2 = static_cast<i16>(0x7fff - na2);
    t_unv[k] = high16_al2b1(wadd<i32>(wmul<i32>(wmul<i32>(ina2, b2), 2) >> 1, wshl<i32>(na2, 16)));
    t_mid[k] = high16_al2b1(wadd<i32>(wmul<i32>(wmul<i32>(ia2, b2), 2) >> 1, wshl<i32>(a2, 16)));
    t_voi[k] = high16_al2b1(wadd<i32>(wmul<i32>(wmul<i32>(nb2, ia2), 2) >> 1, wshl<i32>(a2, 16)));
  }
  i16 limit, stride;
  u16 wrap;
  if (mode == 0) {
    limit = wadd<i16>(static_cast<i16>(static_cast<u16>(1u << (nbits & 0x1f))), 0x38);
    stride = 1;
    wrap = static_cast<u16>(((static_cast<i32>(limit) - 0x38) >> 1) + 0x38);
  } else {
    i16 nb = static_cast<i16>(static_cast<i8>(nbits));
    stride = static_cast<i16>(static_cast<u16>(1u << (static_cast<u32>(static_cast<i32>(wsub<i16>(7, nb))) & 0x1f)));
    limit = static_cast<i16>(static_cast<u16>(1u << (static_cast<u32>(static_cast<i32>(nb)) & 0x1f)));
    wrap = 0;
  }
  i16 best_exp = 0x7fff, best_mant = 0x7fff;
  u32 best_idx = 0;
  if (limit > 0) {
    u16 idx = 0;
    for (;;) {
      if (mode == 0 && static_cast<i16>(idx) == 0x39) idx = wrap;
      i32 code = static_cast<i32>(book[static_cast<usz>(static_cast<i32>(static_cast<i16>(idx)) * static_cast<i32>(stride))]);
      u32 lo = 0;
      i32 hi = 0;
      for (usz k = 0; k < 16; k++) {
        i16 s;
        switch (static_cast<u16>(code) & 3) {
          case 1: s = t_unv[k]; break;
          case 2: s = t_voi[k]; break;
          default: s = t_mid[k]; break;
        }
        i32 prod = wmul<i32>(wmul<i32>(w_c[15 - k], s), 2);
        acc64(lo, hi, static_cast<u32>(prod));
        code >>= 2;
      }
      auto [s, low] = norm64_align(lo, static_cast<u32>(hi));
      i16 e = wneg<i16>(s);
      i16 m = high16_al2b1(low);
      if (m == 0) return idx;
      if (e <= best_exp && (e != best_exp || m < best_mant)) {
        best_idx = idx;
        best_mant = m;
        best_exp = e;
      }
      idx = wadd<u16>(idx, 1);
      if (static_cast<i16>(idx) >= limit) break;
    }
  }
  return best_idx;
}

u16 blend_toward(i16& val, u16& exp, i16& out_a, i16& out_b, i16 x, u16 x_exp, u16 ret) {
  i16 ca = 0x6666;
  i16 cb = 0x7333;
  u16 ea = wsub<u16>(x_exp, 3);
  i16 v = val;
  u16 ev = exp;
  if (compare_scaled(x, static_cast<i16>(x_exp), v, static_cast<i16>(ev)) && compare_scaled(x, static_cast<i16>(x_exp), -0x7000, 5)) {
    ca = 0x4000;
    cb = 0x4000;
    ea = x_exp;
  }
  i32 pa = wmul<i32>(wmul<i32>(ca, x), 2);
  u32 na = norm_exp_l1_batch2(static_cast<u32>(pa));
  i32 pb = wmul<i32>(wmul<i32>(v, cb), 2);
  u32 nb = norm_exp_l1_batch2(static_cast<u32>(pb));
  i16 e = 0;
  i16 m = add_norm(high16_al2b1(static_cast<i32>(static_cast<u32>(pa) << (na & 0x1f))), static_cast<i16>(wsub<u16>(ea, static_cast<u16>(na))),
                   high16_al2b1(static_cast<i32>(static_cast<u32>(pb) << (nb & 0x1f))), static_cast<i16>(wsub<u16>(ev, static_cast<u16>(nb))), e);
  val = m;
  exp = static_cast<u16>(e);
  out_a = 0x7fff;
  out_b = 5;
  return ret;
}

u32 log_scaled_product(i16& mant, i16& exp, i16& exp2, i16 n, i16 a, i16 a_exp, i16 b, i16 tail) {
  i32 p = wmul<i32>(wmul<i32>(a, b), 2);
  i16 s = p == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_shift_l1_batch2(static_cast<u32>(p)));
  exp = wadd<i16>(wsub<i16>(a_exp, s), tail);
  mant = high16_al2b1(static_cast<i32>(static_cast<u32>(p) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f)));
  i32 q = wshl<i32>(0x20 - static_cast<i32>(n), 0x19);
  u32 nq = norm_exp_l1_batch2(static_cast<u32>(q));
  i16 m = add_norm(mant, exp, high16_al2b1(static_cast<i32>(static_cast<u32>(q) << (nq & 0x1f))), static_cast<i16>(-0x1e - static_cast<i32>(nq)), exp2);
  if (m == 0) {
    exp2 = 0xf;
    return 0xffff'8000u;
  }
  i32 l = log2_q(m, exp2);
  i16 sl = l == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_shift_l1_batch2(static_cast<u32>(l)));
  exp2 = wsub<i16>(0xf, sl);
  return static_cast<u16>(high16_al2b1(static_cast<i32>(static_cast<u32>(l) << (static_cast<u32>(static_cast<i32>(sl)) & 0x1f))));
}

i16 sq_diff_norm(i16& exp_out, CS16 x, i16 x_exp, CS16 y, i16 y_exp, i16 n) {
  CS16 a, b;
  i16 e_lo, e_hi;
  if (x_exp < y_exp) { a = y; b = x; e_lo = x_exp; e_hi = y_exp; }
  else { a = x; b = y; e_lo = y_exp; e_hi = x_exp; }
  u32 lo = 0;
  i32 hi = 0;
  i32 cnt = wmul<i32>(n, 2);
  if (cnt > 0) {
    i16 sh = wsub<i16>(wsub<i16>(e_lo, e_hi), 1);
    for (usz i = 0; i < static_cast<usz>(cnt); i++) {
      i32 bv = shift_i32_l1_batch2(static_cast<i32>(b[i]) << 16, sh);
      i32 d = high16_al2b1(wsub<i32>((static_cast<i32>(a[i]) << 16) >> 1, bv));
      acc64(lo, hi, static_cast<u32>(wmul<i32>(wmul<i32>(d, d), 2)));
    }
  }
  auto [s, low] = norm64_align(lo, static_cast<u32>(hi));
  exp_out = static_cast<i16>(static_cast<i32>(e_hi) * 2 + 2 - static_cast<i32>(s));
  return high16_al2b1(low);
}

void log_weight8(S16 out, std::span<const i32> src, u32 unused_, CS16 shifts) {
  (void)unused_;
  for (usz i = 0; i < 8; i++) {
    i16 hi = high16_al2b1(shl_sat(src[i], shifts[i]));
    if (hi == 0) {
      out[i] = 0x7fff;
    } else {
      i32 h = log2_q(hi, 0) >> 1;
      i32 neg = h == INT32_MIN ? INT32_MAX : wneg<i32>(h);
      i32 d = wsub<i32>(neg, 0xa934);
      if (d < 0) out[i] = 0;
      else if (d <= 0xfffe) out[i] = high16_al2b1(wshl<i32>(wadd<i32>(d, 1), 0xf));
      else out[i] = 0x7fff;
    }
  }
}

static void set_agc_params(S16 st, i16 p2, i16 p3, i16 p4, i16 p5, i16 p6, i16 p7, i16 p8, i16 p9, i16 p10, i16 p11) {
  constexpr usz W = 0x16a2 / 2;
  i16 v = wmul<i16>(wadd<i16>(p2, 4), 0x100);
  st[W] = v;
  st[W + 1] = v;
  st[W + 2] = wmul<i16>(p3, 0x100);
  st[W + 3] = wmul<i16>(p4, 0x100);
  st[W + 4] = wmul<i16>(p5, 0x100);
  st[W + 5] = p6;
  st[W + 6] = p7;
  st[W + 7] = p8;
  st[W + 8] = p9;
  st[W + 9] = wmul<i16>(p10, 0x100);
  st[W + 10] = wmul<i16>(p11, 0x100);
}
void set_agc_defaults(S16 st, i16 a, i16 b, i16 c) { set_agc_params(st, a, b, c, -0x28, 0x4000, 0xccc, 0x333, 0xa3, 10, 0x19); }

void update_smoothed_level(S16 st) {
  i32 a = wmul<i32>(st[0x25], 0xcccc);
  u32 na = norm_exp_l1_batch2(static_cast<u32>(a));
  i32 b = wmul<i32>(st[0x28], 0xe666);
  u32 nb = norm_exp_l1_batch2(static_cast<u32>(b));
  i16 e = 0;
  i16 m = add_norm(high16_al2b1(static_cast<i32>(static_cast<u32>(a) << (na & 0x1f))), static_cast<i16>(wsub<u32>(4u, na) & 0xffff),
                   high16_al2b1(static_cast<i32>(static_cast<u32>(b) << (nb & 0x1f))), static_cast<i16>(wsub<u32>(7u, nb)), e);
  i16 sh = wsub<i16>(e, 7);
  if (sh >= 0) st[0x28] = high16_al2b1(wshl<i32>(static_cast<i32>(m) << 16, static_cast<u32>(static_cast<i32>(sh)) & 0x1f));
  else if (sh < -0x1e) st[0x28] = static_cast<i16>(m >> 0xf);
  else st[0x28] = high16_al2b1((static_cast<i32>(m) << 16) >> (static_cast<u32>(static_cast<u8>(wneg<i16>(sh))) & 0x1f));
}

void band_energy_sum(std::span<i32> out, S16 exps, i16 step, u16 n, std::span<const i32> table) {
  if (static_cast<i16>(n) <= 0) return;
  i32 inc = (static_cast<i32>(step) << 9) >> 4;
  u32 cur = static_cast<u32>(wadd<i32>((static_cast<i32>(step) << 9) >> 5, 0x8000));
  for (usz k = 0; k < n; k++) {
    cur = wadd<u32>(cur, static_cast<u32>(inc));
    i32 i0 = static_cast<i16>(cur >> 16);
    u32 nxt = wadd<u32>((static_cast<u32>(i0) << 16) | (cur & 0xffff), static_cast<u32>(inc));
    u32 top = static_cast<u32>(static_cast<i32>(nxt) >> 16) & 0xffff;
    if (high16_al2b1(static_cast<i32>(nxt)) > 0x7f) top = 0x7f;
    u16 span = static_cast<u16>(wsub<u32>(top, static_cast<u32>(static_cast<i32>(cur) >> 16) & 0xffff));
    u32 v0 = static_cast<u32>(table[static_cast<usz>(i0)]);
    i32 f0 = static_cast<i16>(static_cast<i32>(cur & 0xffff) >> 1);
    i32 t = mul32x16(v0, f0);
    t = t == INT32_MIN ? INT32_MAX : wneg<i32>(t);
    u32 head = static_cast<u32>(wadd<i32>(t, static_cast<i32>(v0)));
    i32 f1 = static_cast<i16>(static_cast<u16>(static_cast<u16>(static_cast<i32>(nxt) >> 1) & 0x7fff));
    isz edge = static_cast<isz>(i0) + static_cast<isz>(static_cast<i16>(span));
    u32 tail = static_cast<u32>(mul32x16(static_cast<u32>(table[static_cast<usz>(edge)]), f1));
    u32 lo = wadd<u32>(head, tail);
    i32 hi = wadd<i32>(wadd<i32>(static_cast<i32>(head) >> 31, static_cast<i32>(tail) >> 31),
                       static_cast<i32>((static_cast<u64>(head) + static_cast<u64>(tail)) >> 32));
    if (static_cast<i16>(span) > 1) {
      for (usz j = 1; j < span; j++) acc64(lo, hi, static_cast<u32>(table[static_cast<usz>(i0) + j]));
    }
    auto [s, low] = norm64_align(lo, static_cast<u32>(hi));
    exps[k] = wneg<i16>(s);
    out[k] = low;
    cur = (static_cast<u32>(static_cast<i32>(static_cast<i16>(top))) << 16) | (nxt & 0xffff);
  }
}

void subband_energy_update(std::span<const i32> energy, i16 exp_in, CS16 st_a, S16 st_b, i16 scale_flag) {
  constexpr usz MB = 0x150 / 2;
  constexpr usz EB = MB + 0x10;
  constexpr usz WM = 0x13c / 2;
  constexpr usz WE = 0x13e / 2;
  constexpr usz FLAG = 0x180 / 2;
  constexpr usz VOTES = 0x182 / 2;
  i32 groups[16] = {};
  usz p = 0;
  i16 cnt = 5;
  for (auto& g : groups) {
    i64 v = 0;
    if (cnt != 0) {
      for (i16 r = 0; r < cnt; r++) {
        v = wadd<i64>(v, static_cast<i64>(energy[p]));
        p += 1;
      }
    }
    cnt = 8;
    g = mul32x16(static_cast<u32>(v >> 2), 0x4800);
  }
  i16 s1;
  i32 lo1;
  {
    u32 lo = 0;
    i32 hi = 0;
    for (const auto& g : groups) acc64(lo, hi, static_cast<u32>(g));
    std::tie(s1, lo1) = norm64_align(lo, static_cast<u32>(hi));
  }
  i16 e1 = static_cast<i16>(static_cast<i32>(exp_in) - static_cast<i32>(s1) + 3);
  if (scale_flag == 0) {
    for (usz i = 0; i < 4; i++) {
      i16 sv = st_a[0x63e / 2 + i];
      if (sv < 0x199a) {
        i32 m = wmul<i16>(sv, 5);
        groups[8 + 2 * i] = mul32x16(static_cast<u32>(groups[8 + 2 * i]), m);
        groups[8 + 2 * i + 1] = mul32x16(static_cast<u32>(groups[8 + 2 * i + 1]), m);
      }
    }
  }
  i16 s2;
  i32 lo2;
  {
    u32 lo = 0;
    i32 hi = 0;
    for (usz i = 8; i < 16; i++) acc64(lo, hi, static_cast<u32>(groups[i]));
    std::tie(s2, lo2) = norm64_align(lo, static_cast<u32>(hi));
  }
  i16 e2 = static_cast<i16>(static_cast<i32>(exp_in) - static_cast<i32>(s2) + 3);
  i16 m2 = high16_al2b1(lo2);
  for (usz i = 0; i < 8; i++) {
    u32 v = static_cast<u32>(groups[8 + i]);
    u32 n = norm_exp_l1_batch2(v);
    u32 vs = v << (n & 0x1f);
    i16 a = high16_al2b1(wmul<i32>(st_b[MB + i], 0xfeb8));
    i16 b = high16_al2b1(mul32x16(vs, 0xa3));
    i16 e = st_b[EB + i];
    i16 r = add_norm(a, e, b, static_cast<i16>(static_cast<i32>(exp_in) - static_cast<i32>(n) + 3), e);
    st_b[EB + i] = e;
    st_b[MB + i] = r;
  }
  i16 acc_m = 0, acc_e = 0;
  for (usz i = 0; i < 8; i++) acc_m = add_norm(acc_m, acc_e, st_b[MB + i], st_b[EB + i], acc_e);
  i16 a = high16_al2b1(wmul<i32>(st_b[WM], 0xfeb8));
  i16 b = high16_al2b1(mul32x16(static_cast<u32>(lo1), 0xa3));
  i16 we = st_b[WE];
  i16 nw = add_norm(a, we, b, e1, we);
  st_b[WE] = we;
  st_b[VOTES] = wmul<i16>(st_b[VOTES], 2);
  st_b[WM] = nw;
  if (mant_exp_gt(m2, e2, high16_al2b1(wmul<i32>(acc_m, 0x7ae)), acc_e) && mant_exp_gt(m2, e2, 2000, -0x10)) {
    st_b[VOTES] = static_cast<i16>(static_cast<u16>(st_b[VOTES]) | 1);
  }
  if ((static_cast<u8>(static_cast<u16>(st_b[VOTES])) & 0xf) == 0) {
    st_b[FLAG] = 0;
    return;
  }
  if (mant_exp_gt(acc_m, acc_e, 0x7a12, -10)) {
    st_b[FLAG] = 1;
    return;
  }
  if (mant_exp_gt(acc_m, acc_e, 0x9c4, -13)) {
    if (mant_exp_gt(acc_m, acc_e, high16_al2b1(wmul<i32>(st_b[WM], 0xcccc)), st_b[WE])) st_b[FLAG] = 1;
    return;
  }
  if (mant_exp_gt(acc_m, acc_e, 0xc35, -10)) {
    if (mant_exp_gt(acc_m, acc_e, high16_al2b1(wmul<i32>(st_b[WM], 0x8ccc)), st_b[WE])) st_b[FLAG] = 1;
    return;
  }
  st_b[FLAG] = 0;
}

void dc_blocker(std::span<i32> st, S16 out, CS16 src, u16 n) {
  i32 tmp[84] = {};
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) tmp[i] = (static_cast<i32>(src[i]) << 16) >> 2;
  }
  if (static_cast<i16>(n) > 0) {
    i32 y1p = st[1];
    i32 y1p2 = st[4];
    i32 y_p = st[3];
    i32 y1b = st[2];
    i32 y_p2 = st[5];
    for (usz i = 0; i < n && i < 84; i++) {
      i32 a = y1b;
      i32 b = y_p;
      i32 x = tmp[i];
      i32 prod = mul32x16(static_cast<u32>(y1p), 0x78ab);
      i32 xprev = st[0];
      i64 v = static_cast<i64>(prod) - static_cast<i64>(xprev) + static_cast<i64>(x);
      i32 y1 = sat64(v);
      st[0] = x;
      i64 v2 = v + static_cast<i64>(y1p2) - 2 * static_cast<i64>(a) + 2 * static_cast<i64>(mul32x16(static_cast<u32>(b), 0x7c20)) -
               static_cast<i64>(mul32x16(static_cast<u32>(y_p2), 0x78af));
      y1p2 = a;
      i32 y = sat64(v2);
      tmp[i] = y;
      y1b = y1;
      y1p = y1;
      y_p = y;
      y_p2 = b;
    }
    st[2] = y1p;
    st[3] = y_p;
    st[1] = y1p;
    st[4] = y1p2;
    st[5] = y_p2;
  }
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 v = shl_sat(mul32x16(static_cast<u32>(tmp[i]), 0x78aa), 2);
      i32 r = wadd<i32>(v, 0x8000);
      if (((r ^ v) & r) < 0) r = static_cast<i32>(wadd<u32>(0x7fff'ffffu, v < 0 ? 1u : 0u));
      out[i] = high16_al2b1(r);
    }
  }
}

std::pair<i16, i32> norm64_align(u32 lo, u32 hi) {
  i16 s = static_cast<i16>(norm64(lo, hi));
  i64 v = static_cast<i64>((static_cast<u64>(hi) << 32) | static_cast<u64>(lo));
  i64 r = s < 0 ? allshr(v, static_cast<u8>(wneg<i16>(s))) : allshl(v, static_cast<u8>(s));
  return {s, static_cast<i32>(r)};
}
void acc64(u32& lo, i32& hi, u32 x) {
  u64 carry = (static_cast<u64>(lo) + static_cast<u64>(x)) >> 32;
  lo = wadd<u32>(lo, x);
  hi = wadd<i32>(wadd<i32>(hi, static_cast<i32>(x) >> 31), static_cast<i32>(carry));
}
i32 mul32x16(u32 a, i32 g) {
  return wadd<i32>(wmul<i32>(static_cast<i32>(a & 0xffff), g) >> 15, wmul<i32>(wmul<i32>(static_cast<i16>(a >> 16), g), 2));
}
i32 sat64(i64 v) { return v > INT32_MAX ? INT32_MAX : (v < INT32_MIN ? INT32_MIN : static_cast<i32>(v)); }
i64 allshl(i64 v, u8 count) {
  u32 hi = static_cast<u32>(v >> 32);
  u32 lo = static_cast<u32>(v);
  if (count >= 0x40) return 0;
  u32 nh, nl;
  if (count >= 0x20) {
    nh = lo << (count & 0x1f);
    nl = 0;
  } else if (count == 0) {
    nh = hi;
    nl = lo;
  } else {
    nh = (hi << count) | (lo >> (32 - count));
    nl = lo << count;
  }
  return static_cast<i64>((static_cast<u64>(nh) << 32) | nl);
}

static i16 high16(i32 x) { return static_cast<i16>(x >> 16); }
static i32 shift32(i32 x, i16 s) {
  if (s < 0) {
    if (s < -0x1e) return x >> 31;
    return x >> (static_cast<u32>(static_cast<u8>(wneg<i16>(s))) & 0x1f);
  }
  return static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(static_cast<u8>(s)) & 0x1f));
}
static i32 norm_shift(u32 x) {
  u32 mag = static_cast<i32>(x) < 0 ? ~x : x;
  i32 i = 31;
  if (mag != 0) {
    while ((mag >> i) == 0) i -= 1;
  }
  return 0x1e - i;
}
static i32 shift_i32_l1_batch4(i32 v, i16 sh) {
  if (sh < 0) {
    if (sh < -0x1e) return v >> 31;
    return v >> (static_cast<u32>(wneg<i32>(sh)) & 0x1f);
  }
  return wshl<i32>(v, static_cast<u32>(static_cast<i32>(sh)) & 0x1f);
}
static i32 shift_i32_l1_batch5(i32 v, i16 sh) { return shift_i32_l1_batch4(v, sh); }  // identical code
static u32 sum_norm64(std::span<const i32> x, i16 n, i16& exp) {
  i64 acc = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) acc = wadd<i64>(acc, static_cast<i64>(x[i]));
  }
  i16 s = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
  i64 r = s < 0 ? allshr(acc, static_cast<u8>(static_cast<u32>(wneg<i32>(s)) & 0xff))
                : allshl_l1_batch4(acc, static_cast<u8>(static_cast<u32>(static_cast<i32>(s)) & 0xff));
  exp = wsub<i16>(exp, s);
  return static_cast<u32>(r);
}
static std::pair<u32, i16> norm_sum64(i64 sum) {
  i16 sh = static_cast<i16>(norm64(static_cast<u32>(sum), static_cast<u32>(sum >> 32)));
  i64 shifted = sh < 0 ? (sum >> (static_cast<u32>(-static_cast<i32>(sh)) & 0x3f)) : static_cast<i64>(static_cast<u64>(sum) << (static_cast<u32>(static_cast<i32>(sh)) & 0x3f));
  return {static_cast<u32>(shifted), sh};
}
static const i16 LOG_FIXUP[4] = {16977, 19484, 23170, 27554};

i16 atan_bfp(i16 x, i16 exp, i16& out_exp) {
  bool neg = x < 0;
  i16 mag = x < 0 ? (x == INT16_MIN ? static_cast<i16>(0x7fff) : static_cast<i16>(-x)) : x;
  i16 e = exp;
  i16 m, quad, oct;
  if (mant_exp_gt(mag, exp, 0x4d41, 2)) {
    i16 ex = 0;
    i16 q = div_norm(0x4000'0000, 1, mag, exp, ex);
    m = q == INT16_MIN ? static_cast<i16>(0x7fff) : static_cast<i16>(-q);
    quad = 0x6488;
    oct = 1;
    e = ex;
  } else if (!mant_exp_gt(mag, exp, 0x6a0a, -1)) {
    m = mag;
    quad = 0;
    oct = 0;
  } else {
    i16 ea = 0, es = 0;
    i16 den = add_norm(0x4000, 1, mag, exp, ea);
    i16 d = sub_norm(0x4000, 1, mag, exp, es);
    i32 n32 = wmul<i32>(d, 0x10000);
    i16 ex = 0;
    if (n32 < 0) {
      i32 p = n32 == INT32_MIN ? INT32_MAX : -n32;
      i16 q = div_norm(p, es, den, ea, ex);
      m = q == INT16_MIN ? static_cast<i16>(0x7fff) : static_cast<i16>(-q);
    } else {
      m = div_norm(n32, es, den, ea, ex);
    }
    quad = 0x6488;
    oct = 0;
    e = ex;
  }
  i32 xm = m;
  i32 zw = wmul<i32>(wmul<i32>(xm, xm), 2);
  u32 n0 = norm_u16(zw);
  u32 ez = static_cast<u32>(wsub<i32>(wmul<i32>(e, 2), static_cast<i32>(n0))) & 0xffff;
  i32 z = high16(static_cast<i32>(static_cast<u32>(zw) << (n0 & 0x1f)));
  i32 t = shift32(wmul<i32>(z, 0xe38e), static_cast<i16>(static_cast<u16>(wsub<u32>(ez, 1u))));
  i32 a1 = wadd<i32>(t, static_cast<i32>(0xb6db'0000u));
  u32 n1 = norm_u16(a1);
  i32 s1 = static_cast<i32>(wadd<u32>(wadd<u32>(ez, 2u), wsub<u32>(wsub<u32>(0u, n1), 2u) & 0xffff));
  t = shift32(wmul<i32>(wmul<i32>(high16(static_cast<i32>(static_cast<u32>(a1) << (n1 & 0x1f))), z), 2), static_cast<i16>(s1));
  i32 a2 = wadd<i32>(t, 0x6666'0000);
  u32 n2 = norm_u16(a2);
  i32 s2 = static_cast<i32>(wadd<u32>(wadd<u32>(ez, 1u), wsub<u32>(wsub<u32>(0u, n2), 2u) & 0xffff));
  t = shift32(wmul<i32>(wmul<i32>(high16(static_cast<i32>(static_cast<u32>(a2) << (n2 & 0x1f))), z), 2), static_cast<i16>(s2));
  i32 a3 = wadd<i32>(t, static_cast<i32>(0xaaab'0000u));
  u32 n3 = norm_u16(a3);
  i32 p = wmul<i32>(wmul<i32>(high16(static_cast<i32>(static_cast<u32>(a3) << (n3 & 0x1f))), z), 2);
  u32 n4 = norm_u16(p);
  i32 s3 = static_cast<i32>(wadd<u32>(wsub<u32>(wsub<u32>(0u, n3), 1u) & 0xffff, wsub<u32>(ez, n4)));
  i32 r = shift32(wmul<i32>(wmul<i32>(high16(static_cast<i32>(static_cast<u32>(p) << (n4 & 0x1f))), xm), 2), static_cast<i16>(s3));
  r = wadd<i32>(r, wmul<i32>(xm, 0x10000));
  r = shift32(r, static_cast<i16>(wsub<i32>(e, 1)));
  i16 os = wsub<i16>(oct, 1);
  i32 qterm;
  if (os < 0) {
    if (os < -0x1e) qterm = 0;
    else qterm = (static_cast<i32>(quad) << 16) >> (static_cast<u32>(static_cast<u8>(static_cast<u8>(0) - static_cast<u8>(os))) & 0x1f);
  } else {
    qterm = static_cast<i32>(static_cast<u32>(static_cast<i32>(quad) << 16) << (static_cast<u32>(static_cast<u8>(os)) & 0x1f));
  }
  i32 sum = wadd<i32>(r, qterm);
  i16 sh = sum == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_e(sum));
  u16 mant = static_cast<u16>(high16(static_cast<i32>(static_cast<u32>(sum) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f))));
  if (mant == 0) {
    out_exp = 0;
    return 0;
  }
  out_exp = wsub<i16>(1, sh);
  if (!neg) return static_cast<i16>(mant);
  if (mant == 0x8000) return 0x7fff;
  return wneg<i16>(static_cast<i16>(mant));
}

void build_voicing_table(std::span<u32> out, u32 prev, i32 nbits) {
  i16 end = wadd<i16>(static_cast<i16>(1 << (static_cast<u8>(nbits) & 0x1f)), 0x38);
  u32 i = 0;
  if (end <= 0) return;
  for (;;) {
    if (static_cast<i16>(i) == 0x39) i = static_cast<u32>(static_cast<i32>(wadd<i16>(static_cast<i16>((end - 0x38) >> 1), 0x38))) & 0xffff;
    u32 v = voicing_pattern(static_cast<i16>(i), prev, static_cast<i16>(nbits));
    i16 at = static_cast<i16>(i);
    i = wadd<u32>(i, 1u);
    out[static_cast<usz>(at)] = v;
    if (static_cast<i16>(i) >= end) break;
  }
}

void band_gain_envelope(CS16 mant, CS16 exp, S16 out) {
  i64 acc = 0;
  for (usz i = 0; i < 16; i++) {
    i32 v = shift32(static_cast<i32>(mant[i]) << 16, wsub<i16>(exp[i], 5));
    acc = wadd<i64>(acc, static_cast<i64>(v));
  }
  i32 t = static_cast<i32>(static_cast<u32>(static_cast<u64>(acc) >> 4)) >> 16;
  i32 mean = static_cast<i16>(t) < -0x5a00 ? -0x5a00 : static_cast<i32>(static_cast<i16>(t));
  i32 ramp = static_cast<i32>(wshl<u32>(static_cast<u32>(wadd<i32>(mean, 0x2800)), 16));
  i32 total = 0;
  for (usz i = 0; i < 16; i++) {
    i64 x = static_cast<i64>(static_cast<i32>(wneg<i16>(mant[i])) << 16);
    i32 s = wsub<i16>(exp[i], 5);
    i32 sv = static_cast<i32>(s < 0 ? allshr(x, static_cast<u8>(static_cast<u32>(-s))) : allshl_l1_batch3(x, static_cast<u8>(static_cast<u32>(s))));
    i16 v = high16(wadd<i32>(sv, ramp));
    if (v < 0x400) v = 0x400;
    else if (v > 0x2800) v = 0x2800;
    ramp = wsub<i32>(ramp, 0x266'6000);
    out[i] = v;
    total = wadd<i32>(total, v);
  }
  i16 g = static_cast<i16>(div32_half(0x3ff'ffff, static_cast<i16>(total >> 4)));
  for (usz i = 0; i < 16; i++) out[i] = high16(wmul<i32>(wmul<i32>(out[i], g), 2));
}

void noise_track_update(i16& a_m, i16& a_e, i16& b_m, i16& b_e, i16 c_m, i16 c_e, i16 d_m, i16 d_e, i16 e_m, i16 e_e, i16& cur_m, i16& cur_e,
                        i16& max_m, i16& max_e, i16 sel, i16 g1, i16 g2, i16 lvl, i16 ref_lvl) {
  if (d_m == 0) {
    i16 ex = 0;
    i16 s = sub_norm(max_m, max_e, cur_m, cur_e, ex);
    if (s > 0) {
      max_m = cur_m;
      max_e = cur_e;
    }
    cur_m = 0;
    cur_e = 0;
    return;
  }
  i32 t = (static_cast<i32>(sel) << 16) >> 5;
  t = t == INT32_MIN ? INT32_MAX : -t;
  i32 w = wmul<i32>(wmul<i32>(high16(wmul<i32>(wadd<i32>(t, 0x10000), 0x4000)), g1), 2);
  u32 nw = norm_u16(w);
  i16 e8 = 0;
  i16 den = add_norm(c_m, c_e, high16(static_cast<i32>(static_cast<u32>(w) << (nw & 0x1f))), static_cast<i16>((-0x13) - static_cast<i32>(nw)), e8);
  i16 p7 = 0;
  i16 q = div_norm(static_cast<i32>(c_m) << 16, c_e, den, e8, p7);
  p7 = wadd<i16>(p7, 5);
  i16 step = high16(wmul<i32>(wmul<i32>(q, g2), 2));
  i16 bm = b_m;
  i16 be = b_e;
  i16 r = add_norm(bm, be, step, p7, b_e);
  b_m = r;
  i16 lc = 0;
  i16 sn = sub_norm(d_m, d_e, e_m, e_e, lc);
  i16 mag = sn < 0 ? (sn == INT16_MIN ? static_cast<i16>(0x7fff) : wneg<i16>(sn)) : sn;
  i32 iv = mag;
  i32 prod = wmul<i32>(wmul<i32>(step, iv), 2);
  i16 sh = prod == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_e(prod));
  p7 = wadd<i16>(p7, wsub<i16>(lc, sh));
  i16 am = a_m;
  i16 ae = a_e;
  r = add_norm(am, ae, high16(static_cast<i32>(static_cast<u32>(prod) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f))), p7, a_e);
  a_m = r;
  i16 saved_e = lc;
  i16 out_m = mag;
  if ((static_cast<i32>(lvl) - static_cast<i32>(ref_lvl) > 0x800) || (lvl < -0x1900)) {
    if (sel == 0) out_m = static_cast<i16>((iv << 15) >> 16);
    else if (sel == 1) out_m = high16(wmul<i32>(iv, 0xc000));
  }
  i16 e8b = 0;
  i16 nm = add_norm(cur_m, cur_e, out_m, lc, e8b);
  i16 lc2 = 0;
  i16 s = sub_norm(max_m, max_e, nm, e8b, lc2);
  if (s > 0) {
    max_m = nm;
    max_e = e8b;
  }
  cur_m = out_m;
  cur_e = saved_e;
}

i32 parabolic_peak(i16 peak, CS16 y, i16 n) {
  i32 orig = peak;
  i16 p = peak;
  if (peak == 0) p = 1;
  else if (orig == static_cast<i32>(n) - 1) p = static_cast<i16>(peak - 1);
  i32 i = p;
  i32 ym = static_cast<i32>(y[static_cast<usz>(i - 1)]) << 16;
  i32 yp = static_cast<i32>(y[static_cast<usz>(i + 1)]) << 16;
  i32 d2 = wadd<i32>(wadd<i32>(ym >> 1, wmul<i32>(y[static_cast<usz>(i)], -0x10000)), yp >> 1);
  if (d2 < 0) {
    i16 q = static_cast<i16>(div32_sat15(wsub<i32>(ym >> 2, yp >> 2), static_cast<i16>(d2 >> 16)));
    return wmul<i32>(wadd<i32>(q, wmul<i32>(i, 0x8000)), 2);
  }
  return orig << 16;
}

i32 reciprocal_vec(S16 dst, CS16 src, i32 exp_in, i16 shift, u16 n) {
  i32 num;
  if (shift < 0) {
    if (shift < -0x1e) num = 0;
    else num = 0x8000 >> (static_cast<u32>(static_cast<u8>(static_cast<u8>(0) - static_cast<u8>(shift))) & 0x1f);
  } else {
    num = static_cast<i32>(0x8000u << (static_cast<u32>(static_cast<u8>(shift)) & 0x1f));
  }
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i16 s = src[i];
      dst[i] = s == 0 ? static_cast<i16>(0x7fff) : static_cast<i16>(div32_half(num, s));
    }
  }
  return static_cast<i32>(static_cast<u16>(wsub<i16>(0x10, shift))) - exp_in;
}

i16 voicing_threshold(i16& out_mant, CS16 st, i16 base, i32 exp, i16 band, i32 use_hist) {
  i16 k = wshl<i16>(band, 11);
  i16 s = static_cast<i16>(5 - exp);
  i32 a = wadd<i32>(wmul<i32>(k, -0x2222), 0x10000000);
  a = shift32(a, s);
  a = wadd<i32>(a, wmul<i32>(base, 0x10000));
  i32 b = wadd<i32>(wmul<i32>(k, -0x4444), -0x30000000);
  b = shift32(b, s);
  if (a < b) b = a;
  usz idx = static_cast<usz>(band);
  i32 floor_ = static_cast<i32>(st[0x29 + idx]) << 16;
  floor_ = shift32(floor_, wsub<i16>(st[0x39 + idx], static_cast<i16>(exp)));
  if (static_cast<i16>(use_hist) != 0 && band > 8) {
    i32 l = log2_q(st[0xa0 + idx], st[0xb0 + idx]);
    u32 ln = norm_u16(l);
    i16 v = high16(shl_sat(static_cast<i32>(static_cast<u32>(l) << (ln & 0x1f)), static_cast<i16>(wsub<i32>(static_cast<i32>(wsub<u32>(0x0fu, ln)), exp))));
    if (high16(floor_) < v) floor_ = static_cast<i32>(v) << 16;
  }
  i32 margin = wsub<i32>(b, floor_);
  if (margin < 0) {
    i32 arg = wadd<i32>(wmul<i32>(high16(margin), 0x8000), static_cast<i32>((static_cast<u32>(margin) & 0xffff) << 0xe) >> 0xf) >> 9;
    i16 e = 0;
    i32 m = pow2_frac(arg, e);
    i16 sh = m == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_e(m));
    out_mant = high16(static_cast<i32>(static_cast<u32>(m) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f)));
    return wsub<i16>(e, sh);
  }
  out_mant = 0x4000;
  return 1;
}

bool take_signalling_entry(i16& out_tag, S16 out_pair, u32 flags, S16 state) {
  if ((flags & 0x4000) != 0) {
    out_tag = state[2];
    out_pair[0] = state[1];
    out_pair[1] = state[1];
    state[0] = 1;
    return true;
  }
  u8 bytes[0x16] = {};
  for (usz k = 0; k < 0x0b; k++) {
    u16 w = static_cast<u16>(state[k]);
    bytes[2 * k] = static_cast<u8>(w & 0xff);
    bytes[2 * k + 1] = static_cast<u8>(w >> 8);
  }
  bool found = find_active_entry(std::span<const u8>(bytes), out_tag, out_pair);
  state[0] = 0;
  return found;
}

i16 level_follow_gain(CS16 st, CS16 ctrl, S16 p) {
  i16 lim = p[9];
  i32 meas = static_cast<i32>(st[0x25]) << 16;
  i16 d = high16(wadd<i32>(meas, wmul<i32>(p[0], -0x10000)));
  if (static_cast<i32>(d) < -static_cast<i32>(lim)) d = static_cast<i16>(-static_cast<i32>(lim));
  i32 coef = 0;
  bool update = true;
  if (lim < d) {
    coef = p[5];
  } else if (d < 1) {
    if (st[0x7b] < 0x4001 || ctrl[5] > 0xccb || high16(wadd<i32>(meas, wmul<i32>(st[0x28], -0x10000))) <= p[10]) {
      if (st[0x7b] < 0x2001) update = false;
      else coef = p[8];
    } else {
      coef = p[7];
    }
  } else {
    coef = p[6];
  }
  if (update) p[0] = high16(wmul<i32>(wadd<i32>(wmul<i32>(coef, d), wmul<i32>(p[0], 0x8000)), 2));
  if (p[0] < p[4]) p[0] = p[4];
  i16 g = high16(wadd<i32>(wmul<i32>(p[1], 0x10000), wmul<i32>(p[0], -0x10000)));
  if (p[2] < g) g = p[2];
  else if (g < p[3]) g = p[3];
  i16 above = high16(wadd<i32>(wmul<i32>(p[0], 0x10000), wmul<i32>(st[0x28], -0x10000)));
  if (above < p[10]) {
    i32 q = div32_half((static_cast<i32>(above) << 16) >> 4, p[10]);
    i16 f = static_cast<i16>((static_cast<u32>(q) << 0x14) >> 0x10);
    return high16(wmul<i32>(wmul<i32>(f, g), 2));
  }
  return g;
}

i16 sum_mant_exp(std::span<const i32> x, i16 n, i16& exp) { return static_cast<i16>(static_cast<i32>(sum_norm64(x, n, exp)) >> 16); }

void accumulate_weighted_diff(i16& acc_m, i16& acc_e, i16& wsum_m, i16& wsum_e, i16& out_m, i16& out_e, i16 a_m, i16 a_e, i16 b_m, i16 b_e, i16 w_m,
                              i16 w_e, i16 idx) {
  i16 e1 = 0;
  i16 r1 = sub_norm(a_m, a_e, b_m, b_e, e1);
  out_m = r1;
  out_e = e1;
  i16 e2 = 0;
  i16 r2 = sub_norm(b_m, b_e, a_m, a_e, e2);
  i16 mant = r1 < 0 ? r2 : r1;
  i16 exp = r1 < 0 ? e2 : e1;
  if (idx < 6) {
    i32 p = wmul<i32>(wmul<i32>(mant, w_m), 2);
    u16 nrm_ = p == 0 ? static_cast<u16>(0) : static_cast<u16>(static_cast<u16>(norm_shift(static_cast<u32>(p))) & 0xffff);
    i16 ne = static_cast<i16>(wadd<u16>(wsub<u16>(static_cast<u16>(exp), nrm_), static_cast<u16>(w_e)));
    i16 m = static_cast<i16>(wshl<i32>(p, static_cast<u32>(nrm_) & 0x1f) >> 16);
    i16 am = acc_m, ae = acc_e;
    acc_m = add_norm(am, ae, m, ne, acc_e);
    i16 sm = wsum_m, se = wsum_e;
    wsum_m = add_norm(sm, se, w_m, w_e, wsum_e);
  }
}

i16 align_block_exp(S16 dst, CS16 src, CS16 exps, i16 n) {
  i16 m = vec_max(exps, n);
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) {
      i32 v = static_cast<i32>(src[i]) << 16;
      i16 sh = wsub<i16>(exps[i], m);
      if (sh < 0) {
        if (sh < -0x1e) dst[i] = static_cast<i16>(src[i] >> 15);
        else dst[i] = high16_al2b3(v >> (static_cast<u32>(-static_cast<i32>(sh)) & 0x1f));
      } else {
        dst[i] = high16_al2b3(wshl<i32>(v, static_cast<u32>(static_cast<i32>(sh)) & 0x1f));
      }
    }
  }
  return m;
}

u32 accumulate_exp(std::span<const i32> x, i16 n, i16& exp) { return sum_norm64(x, n, exp); }

i32 log2_at_exp(u32 x, i16 exp, i16 target_exp) {
  i16 e = exp;
  i32 m = static_cast<i32>(log2_transform(x, e, CS16(LOG_FIXUP)));
  return shift_round16_l1_batch4(m, wsub<i16>(e, target_exp));
}

u16 extract_bit_pattern(i16 word, u16 n, i16 scale) {
  i32 v = static_cast<i32>(scale) << 16;
  u16 nrm_ = v == 0 ? static_cast<u16>(0) : static_cast<u16>(static_cast<u16>(norm_shift(static_cast<u32>(v))) & 0xffff);
  i16 mant = static_cast<i16>(wshl<i32>(v, static_cast<u32>(nrm_) & 0x1f) >> 16);
  i32 q = static_cast<i16>(static_cast<u16>(div32_half(static_cast<i32>(0x3b39'c0ecu), mant)));
  i16 sh = static_cast<i16>(wsub<u16>(nrm_, 0xf));
  if (static_cast<i16>(n) <= 0) return 0;
  u16 acc = 0;
  i16 k = 0x20;
  i32 left = wsub<i32>(static_cast<i16>(n), 1);
  u16 floor_idx = wsub<u16>(n, 1);
  for (u16 r = 0; r < n; r++) {
    i32 e = shift_i32_l1_batch4(wmul<i32>(wmul<i32>(k, q), 2), sh);
    u16 idx = static_cast<u16>(wsub<i32>(7, wadd<i32>(e, 0x4000) >> 16));
    if (static_cast<i16>(idx) < 0) idx = 0;
    if (static_cast<i32>(static_cast<i16>(idx)) > left) idx = floor_idx;
    u16 c = idx & 0x1f;
    u16 bit = c >= 16 ? static_cast<u16>((word >> 15) & 1) : static_cast<u16>((word >> c) & 1);
    acc = static_cast<u16>(wmul<u16>(acc, 2) | bit);
    k = wadd<i16>(k, 0x30);
    floor_idx = wsub<u16>(floor_idx, 1);
    left -= 1;
  }
  return acc;
}

i32 level_penalty(CS16 state, i16& level) {
  i32 s48 = state[0x24];
  i16 s4a = state[0x25];
  u16 flags = static_cast<u16>(state[0x22]);
  i32 d = wsub<i32>(static_cast<i32>(s4a) << 16, s48 << 16);
  i16 t = static_cast<i16>((static_cast<i32>(state[0x26]) << 24) >> 16);
  i32 acc = static_cast<i32>(level) << 16;
  if (t > -0x5a00 && s4a <= t) {
    if (state[0x9c] < 8) {
      acc = wsub<i32>(acc, 0x3800'0000);
    } else if (d > -0x13ff'fb00 && ((flags & 1) != 0 || wsub<i32>(s48, state[0x28]) < 0x800)) {
      acc = wsub<i32>(acc, 0x0999'8000);
    } else if (d > -0x1400'0000) {
      acc = wsub<i32>(acc, 0x1000'0000);
    } else if (d > -0x1e00'0000) {
      i32 s = wmul<i32>(static_cast<i16>(wadd<i32>(d, 0x1e00'0000) >> 16), 0xccca);
      acc = wadd<i32>(wadd<i32>(acc, static_cast<i32>(0xe800'0000u)), s);
    } else if (d > -0x2800'0000) {
      i32 s = wmul<i32>(static_cast<i16>(wadd<i32>(d, 0x2800'0000) >> 16), static_cast<i32>(0xffff'3334u));
      acc = wadd<i32>(acc, wsub<i32>(static_cast<i32>(0xe400'0000u), s >> 1));
    } else if (d > -0x3200'0000 && popcount_bits(flags, 0x12) >= 2) {
      i32 s = wmul<i32>(static_cast<i16>(wadd<i32>(d, 0x3200'0000) >> 16), 0x33328);
      acc = wadd<i32>(wadd<i32>(acc, static_cast<i32>(0xc400'0000u)), s);
    } else {
      acc = wsub<i32>(acc, 0x1c00'0000);
    }
  }
  i16 r = high16_al2b3(acc);
  level = r;
  return r > 0 ? 1 : 0;
}

u16 unvoiced_ratio(i16 num, i16 num_exp, CS16 tail, i16 hint, i16 hint_exp) {
  i32 v = shift_i32_l1_batch4(static_cast<i32>(hint) << 16, wsub<i16>(hint_exp, 0x10));
  i16 off = static_cast<i16>(wadd<i32>(v, 0x10000) >> 16);
  i16 cnt = static_cast<i16>(0x20 - static_cast<i32>(static_cast<u16>(off)));
  i32 sum = sum_i16(tail.subspan(static_cast<usz>(off)), cnt);
  if (sum <= 0) return 0x7fff;
  u16 nrm_ = static_cast<u16>(0x1e - (31 - __builtin_clz(static_cast<u32>(sum))));
  i16 den = static_cast<i16>(wshl<i32>(sum, static_cast<u32>(nrm_) & 0x1f) >> 16);
  i16 e = 0;
  i16 q = div_norm(static_cast<i32>(num) << 16, num_exp, den, static_cast<i16>(wsub<u16>(0, nrm_)), e);
  if (e >= 1) return 0;
  i32 val = shift_i32_l1_batch4(static_cast<i32>(q) << 16, e);
  if (val == INT32_MIN) return 0xfffe;
  return static_cast<u16>((static_cast<u32>(wadd<i32>(wneg<i32>(val), 0x7fff'0000) >> 16)) & 0xffff);
}

i16 log_energy_to_gain(i16& out, CS16 state, i16 half) {
  i32 acc = wadd<i32>((static_cast<i32>(state[0x9b]) << 16) >> 1, (static_cast<i32>(state[0x9a]) << 16) >> 1);
  if (half > 0) {
    i32 c = wadd<i32>(state[0x91], state[0x89]);
    acc = wmul<i32>(wsub<i32>(wmul<i32>(c, 0x10000), acc >> 1), 2);
  }
  i32 d = wsub<i32>(acc, 0x1200'0000);
  if (d >= 0) {
    out = 0x4000;
    return 1;
  }
  i32 hi = wmul<i32>(static_cast<i16>(d >> 16), 0xaac0);
  i32 lo = wmul<i32>(static_cast<i32>(static_cast<u32>(d) & 0xffff), 0x5560) >> 15;
  i16 exp = 0;
  i32 m = pow2_frac(wadd<i32>(hi, lo) >> 9, exp);
  u16 nrm_ = m == 0 ? static_cast<u16>(0) : static_cast<u16>(static_cast<u16>(norm_shift(static_cast<u32>(m))) & 0xffff);
  i16 e = wsub<i16>(exp, static_cast<i16>(nrm_));
  i16 mant = static_cast<i16>(wshl<i32>(m, static_cast<u32>(nrm_) & 0x1f) >> 16);
  if (compare_scaled(mant, e, 0x51eb, -6)) {
    out = 0x51eb;
    return -6;
  }
  out = mant;
  return e;
}

i32 speech_activity_gate(CS16 state, i16& level, i16& exp, i16& flag) {
  i16 best_m = state[0x29];
  i16 best_e = state[0x39];
  for (usz i = 0; i < 15; i++) {
    if (mant_exp_gt(state[0x2a + i], state[0x3a + i], best_m, best_e)) {
      best_m = state[0x2a + i];
      best_e = state[0x3a + i];
    }
  }
  i16 tot_e = 0, tot_m = 0;
  for (usz j = 0; j < 8; j++) tot_m = add_norm(tot_m, tot_e, state[0xa8 + j], state[0xb8 + j], tot_e);
  flag = state[0xc0];
  bool active = mant_exp_gt(tot_m, tot_e, 0x7a12, -10);
  if (active) {
    flag = 1;
  } else {
    if (mant_exp_gt(tot_m, tot_e, 0xc35, -10)) {
      u16 t = static_cast<u16>((static_cast<u32>(wmul<i32>(state[0x9e], 0x8ccc) >> 16)) & 0xffff);
      if (mant_exp_gt(tot_m, tot_e, static_cast<i16>(t), state[0x9f])) {
        flag = 1;
        active = true;
      }
    }
    if (!active && wadd<i32>(state[0x28], 0x1e00) <= 0 && !mant_exp_gt(best_m, best_e, static_cast<i16>(0xb000u), 5) &&
        wsub<i32>(wsub<i32>(state[0x24], state[0x28]), 0xf00) >= 0) {
      return 0;
    }
  }
  i64 acc = 0;
  for (usz k = 0; k < 8; k++) {
    i32 v = shift_i32_l1_batch4(static_cast<i32>(state[0x29 + k]) << 16, wsub<i16>(state[0x39 + k], 5));
    acc = wadd<i64>(acc, static_cast<i64>(v));
  }
  i16 r = static_cast<i16>(static_cast<i32>(static_cast<u32>(acc >> 3)) >> 16);
  if (r < -0x7000) r = -0x7000;
  level = r;
  exp = 5;
  return 1;
}

u16 one_minus_ratio(i16 num, i16 num_exp, i16 den, i16 den_exp) {
  if (den < 1) return 0x7fff;
  i16 e = 0;
  i16 q = div_norm(up16_l1_batch5(num), num_exp, den, den_exp, e);
  if (e > 0) return 0;
  i32 v = shift_i32_l1_batch5(up16_l1_batch5(q), e);
  if (v == INT32_MIN) return 0xfffe;
  return static_cast<u16>(wsub<i32>(static_cast<i32>(0x7fff'0000u), v) >> 16);
}

void band_ratio_weights(CS16 a, S16 out, CS16 c, CS16 d, S16 w, CS16 f, i32 exp_d, i32 exp_f, i16 coef) {
  for (usz i = 0; i < 3; i++) {
    out[i] = a[i] < c[i] ? a[i] : c[i];
    w[i] = high16_l1_batch5(wmul<i32>(wmul<i32>(d[i], coef), 2));
  }
  auto combine = [](i32 x, i32 ex, i32 y, i32 ey) -> std::pair<i32, u32> {
    u16 nx = nrm(static_cast<u32>(x));
    u16 ny = nrm(static_cast<u32>(y));
    i32 ax = wshl<i32>(x, static_cast<u32>(nx) & 0x1f);
    i32 ay = wshl<i32>(y, static_cast<u32>(ny) & 0x1f);
    i32 e1 = wsub<i32>(ex, nx);
    i32 e2 = wsub<i32>(ey, ny);
    i32 top = static_cast<i16>(e1) <= static_cast<i16>(e2) ? e2 : e1;
    u32 common = static_cast<u32>(static_cast<i32>(static_cast<i16>(top)) + 1) & 0xffff;
    i32 sx = shift_i32_l1_batch5(ax, static_cast<i16>((e1 & 0xffff) - static_cast<i32>(common)));
    i32 sy = shift_i32_l1_batch5(ay, static_cast<i16>((e2 & 0xffff) - static_cast<i32>(common)));
    return {wadd<i32>(sx, sy), common};
  };
  for (usz k = 3; k < 8; k++) {
    i32 dk = d[k];
    auto [s1, e_s1] = combine(wshl<i32>(dk, 16), exp_d, up16_l1_batch5(f[k]), exp_f);
    u16 n1 = nrm(static_cast<u32>(s1));
    i32 num = wshl<i32>(s1, static_cast<u32>(n1) & 0x1f);
    i32 p = wmul<i32>(wmul<i32>(a[k], dk), 2);
    i32 q = wmul<i32>(wmul<i32>(c[k], f[k]), 2);
    auto [s2, e_s2] = combine(p, exp_d, q, exp_f);
    u16 n2 = nrm(static_cast<u32>(s2));
    if (high16_l1_batch5(num) == 0) {
      out[k] = 0x7fff;
    } else {
      i16 e = 0;
      i16 r = div_norm(wshl<i32>(s2, static_cast<u32>(n2) & 0x1f), static_cast<i16>(wsub<u32>(e_s2, n2)), high16_l1_batch5(num),
                       static_cast<i16>(wsub<u32>(e_s1, n1)), e);
      out[k] = high16_l1_batch5(shl_sat(up16_l1_batch5(r), e));
    }
    w[k] = high16_l1_batch5(wmul<i32>(wmul<i32>(d[k], coef), 2));
  }
  u16 any = 0;
  for (usz i = 0; i < 8; i++) any = static_cast<u16>(any | static_cast<u16>(w[i]));
  if (any == 0) {
    for (usz i = 0; i < 8; i++) w[i] = 1;
  }
}

i16 ratio_of_block_sums(CS16 num, i32 num_exp, CS16 den, i16 den_exp, i16& exp) {
  i64 dsum = 0;
  for (usz i = 0; i < 0x80; i++) dsum += static_cast<i64>(rd32(den, 2 * i));
  auto [dlo, dsh] = norm_sum64(dsum);
  i16 dm = static_cast<i16>((dlo >> 16) & 0xffff);
  i16 de = wsub<i16>(den_exp, dsh);
  if (dm == 0) {
    dm = 0x68db;
    de = -13;
  }
  i64 nsum = 0;
  for (usz i = 0; i < 0x80; i++) nsum += static_cast<i64>(rd32(num, 2 * i));
  auto [nlo, nsh] = norm_sum64(nsum);
  return div_norm(static_cast<i32>(nlo), static_cast<i16>(wsub<i32>(num_exp, static_cast<u16>(nsh))), dm, de, exp);
}

i16 ratio_percent(u32 a, i32 b, i16& exp) {
  u32 d = wsub<u32>(static_cast<u32>(b), a);
  u32 shifted = a;
  if (static_cast<i32>(d) > 0) {
    u16 na = nrm(a);
    i32 i = 31;
    if (d != 0) {
      while ((d >> i) == 0) i -= 1;
    }
    u16 nd = static_cast<u16>(wsub<u32>(0x1eu, static_cast<u32>(i)) & 0xffff);
    shifted = wshl<u32>(a, static_cast<u32>(na) & 0x1f);
    i16 e = 0;
    i16 q = div_norm(static_cast<i32>(shifted), wneg<i16>(static_cast<i16>(na)), high16_l1_batch5(static_cast<i32>(wshl<u32>(d, static_cast<u32>(nd) & 0x1f))),
                     wneg<i16>(static_cast<i16>(nd)), e);
    if (e < 8) {
      if (e == 7 && static_cast<u16>(q) > 0x63ff) q = 0x6400;
      if (static_cast<i32>(shifted) < 0) q = 0;
      exp = e;
      return q;
    }
  }
  exp = 7;
  return static_cast<i32>(shifted) < 0 ? static_cast<i16>(0) : static_cast<i16>(0x6400);
}

i32 track_level_ar(i16& mant, i16& exp, i16 nw, i16 new_exp, i32 hold, i16 thr) {
  i16 m = mant;
  i16 e = exp;
  i16 e_lo = 0;
  i16 lo = sub_norm(0x4000, 2, m, e, e_lo);
  i16 e_hi = 0;
  i16 hi = add_norm(thr, 7, m, e, e_hi);
  i16 out_m, out_e;
  if (!compare_scaled(nw, new_exp, lo, e_lo)) {
    i16 r = m;
    if (compare_scaled(nw, new_exp, hi, e_hi)) {
      i32 a = wmul<i32>(m, static_cast<i32>(0xe666u));
      u16 na = nrm(static_cast<u32>(a));
      i16 ea = wsub<i16>(e, static_cast<i16>(na));
      i16 ma = high16_l1_batch5(wshl<i32>(a, static_cast<u32>(na) & 0x1f));
      i32 b = wmul<i32>(nw, static_cast<i32>(0xccccu));
      u16 nb = nrm(static_cast<u32>(b));
      i16 eb = wsub<i16>(wsub<i16>(new_exp, static_cast<i16>(nb)), 3);
      r = add_norm(ma, ea, high16_l1_batch5(wshl<i32>(b, static_cast<u32>(nb) & 0x1f)), eb, e);
    }
    out_m = r;
    out_e = e;
  } else {
    out_m = nw;
    out_e = new_exp;
  }
  mant = out_m;
  exp = out_e;
  return wsub<i32>(hold, 1);
}

i32 sum_bands(CS16 v, i16 step, i16 step_exp, i16 count) {
  i16 half = static_cast<i16>(count >> 1);
  i32 s = shift_i32_l1_batch5(up16_l1_batch5(step), wsub<i16>(step_exp, 0xf));
  i32 acc = 0;
  i32 pos = wadd<i32>(wsub<i32>(s, s >> 2), 0x10000);
  while (high16_l1_batch5(pos) < half) {
    i32 nxt = wadd<i32>(pos, s >> 1);
    u32 hi = half < high16_l1_batch5(nxt) ? static_cast<u32>(static_cast<u16>(half)) : ((static_cast<u32>(nxt) >> 16) & 0xffff);
    nxt = wadd<i32>(nxt, s >> 1);
    i16 start = high16_l1_batch5(pos);
    i16 cnt = static_cast<i16>(wsub<u32>(hi, (static_cast<u32>(pos) >> 16) & 0xffff));
    acc = wadd<i32>(acc, sum_i16(v.subspan(static_cast<usz>(start)), cnt));
    pos = nxt;
  }
  return acc;
}

i16 lag_to_pitch_index(i16 lag) {
  i16 q = static_cast<i16>(div32_half(0x800'0000, lag));
  i16 v = static_cast<i16>(wsub<i32>((static_cast<i32>(q) << 16) >> 5, 0x4e'0000) >> 0x11);
  if (v > 0xcf) return 0xcf;
  if (v < 0) return 0;
  return v;
}

i32 log_mean_energy(std::span<const i32> v, i32 exp, i16 den) {
  i32 x0 = v[0];
  i32 neg = static_cast<u32>(x0) == 0x8000'0000u ? 0x7fff'ffff : wneg<i32>(x0);
  i64 acc = static_cast<i64>(neg >> 1);
  for (usz i = 0; i < 0x10; i++) acc = wadd<i64>(acc, static_cast<i64>(v[i]));
  auto [lo, sh] = norm_sum64(acc);
  i16 e = static_cast<i16>(static_cast<u16>(wadd<i32>(wsub<i32>(exp, static_cast<u16>(sh)), 1) & 0xffff));
  i32 dv = up16_l1_batch5(den);
  u16 nd = nrm(static_cast<u32>(dv));
  i16 q = div_norm(static_cast<i32>(lo), e, high16_l1_batch5(wshl<i32>(dv, static_cast<u32>(nd) & 0x1f)), static_cast<i16>(0xf - static_cast<i32>(nd)), e);
  if (q == 0 || e < -0x1f) return -0x59ff;
  i32 l = log2_q(q, e);
  u16 n = nrm(static_cast<u32>(l));
  i32 m = high16_l1_batch5(wshl<i32>(l, static_cast<u32>(n) & 0x1f));
  i32 s = wsub<i32>(static_cast<i32>(wsub<u32>(0xfu, n) & 0xffff), 5);
  i32 scaled = wmul<i32>(m, static_cast<i32>(0xc0a8u));
  i32 t;
  if (static_cast<i16>(s) >= 0) t = wshl<i32>(scaled, static_cast<u32>(s) & 0x1f);
  else if (static_cast<i16>(s) < -0x1e) t = scaled >> 31;
  else t = scaled >> (static_cast<u32>(static_cast<u8>(static_cast<u8>(0) - static_cast<u8>(s))) & 0x1f);
  return wadd<i32>(t, 0x4ba'7000) >> 16;
}

void voiced_band_mask(S16 dst, CS16 amp, CS16 lvl, i16 exp) {
  u32 lo = 0;
  u32 hi = 0xd;
  for (usz k = 0; k < 8; k++) {
    if (k == 7) hi = 0x80;
    i32 v = up16_l1_batch5(lvl[k]);
    i16 n = v == 0 ? static_cast<i16>(0) : static_cast<i16>(0x1e - top_bit_l1_batch5(static_cast<u32>(v)));
    i16 word;
    if (wshl<i32>(v, static_cast<u32>(static_cast<i32>(n)) & 0x1f) == 0) word = 0;
    else if (wsub<i16>(exp, n) < -0x2a) word = 0;
    else if (amp[k] < 0x1999) word = 0;
    else word = 0x4000;
    fill16(dst.subspan(static_cast<usz>(static_cast<i16>(lo))), word, static_cast<i16>(wsub<u32>(hi, lo)));
    lo = hi & 0xffff;
    hi = wadd<u32>(hi, 0x10u);
  }
}

}  // namespace b25::eng::enc
