// Port of blip25-vocoder src/engine/dsp.rs (MIT, (c) Chance Lindsey).
// Fixed-point DSP layer: primitives, block-floating-point arithmetic, FFTs.
// Function names and order follow the Rust source; comments are kept short -
// the Rust file carries the full per-function documentation and the original
// addresses.
#include <algorithm>
#include <cassert>

#include "b25/dsp.h"
#include "b25/dsp_tables.h"
#include "b25/engine_tables.h"

namespace b25::eng {

using namespace rs;

// ======================= from prim =======================

void rand_step(i16& seed) { seed = wadd<i16>(wmul<i16>(seed, static_cast<i16>(0xAD)), 0x3619); }

usz fill16(S16 dst, i16 val, i16 n) {
  if (n <= 0) return 0;
  usz un = static_cast<usz>(n);
  std::fill_n(dst.first(un).begin(), un, val);
  return un;
}

void subc(i16 divisor, i32& acc) {
  i32 diff = wsub<i32>(acc, static_cast<i32>(divisor) << 15);
  acc = diff < 0 ? wmul<i32>(acc, 2) : wadd<i32>(wmul<i32>(diff, 2), 1);
}

// ======================= from batch1 =======================

i32 norm_shift_batch1(u32 x) {
  u32 mag = static_cast<i32>(x) < 0 ? ~x : x;
  return 0x1e - top_bit_batch3(mag);
}

i32 shift_i32_batch1(i32 v, i16 sh) {
  if (sh < 0) {
    if (sh < -0x1e) return v >> 31;
    return v >> (static_cast<u32>(wneg<i32>(sh)) & 0x1f);
  }
  return wshl<i32>(v, static_cast<u32>(static_cast<i32>(sh)) & 0x1f);
}

i32 sat_sub_batch1(i32 a, i32 b) {
  i32 d = wsub<i32>(a, b);
  if (((~b ^ d) & (d ^ a)) < 0) return a < 0 ? INT32_MIN : INT32_MAX;
  return d;
}

i64 allshr(i64 v, u8 count) {
  i32 hi = static_cast<i32>(v >> 32);
  u32 lo = static_cast<u32>(v);
  i32 fill = hi >> 31;
  if (count >= 0x40) {
    return static_cast<i64>((static_cast<u64>(static_cast<u32>(fill)) << 32) | static_cast<u32>(fill));
  }
  if (count < 0x20) {
    u32 n = count & 0x1f;
    u32 new_lo = n == 0 ? lo : ((lo >> n) | (static_cast<u32>(hi) << (32 - n)));
    i32 new_hi = hi >> n;
    return static_cast<i64>((static_cast<u64>(static_cast<u32>(new_hi)) << 32) | new_lo);
  }
  i32 new_lo = hi >> (count & 0x1f);
  return static_cast<i64>((static_cast<u64>(static_cast<u32>(fill)) << 32) | static_cast<u32>(new_lo));
}

i32 popcount_low(u32 x, u16 n) {
  i32 acc = 0;
  if (static_cast<i16>(n) > 0) {
    i32 xs = static_cast<i32>(x);
    for (u16 i = 0; i < n; i++) {
      acc += xs & 1;
      xs >>= 1;
    }
  }
  return acc;
}

i8 frame_size_class(i16 x) {
  u16 u = static_cast<u16>(x);
  if (static_cast<u16>(u - 5) < 0x76) return 0;
  if (static_cast<u16>(u - 0x80) < 0x10) return 1;
  if (static_cast<u16>(u - 0x90) < 0x10) return 2;
  return static_cast<i8>(3 + (static_cast<u16>(u - 0xa0) > 3 ? 1 : 0));
}

i32 sum_norm(i16& out, CS16 v, i32 exp, u16 n) {
  u32 sum = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) sum = wadd<u32>(sum, static_cast<u32>(static_cast<i32>(v[i])));
  }
  u32 sh = sum == 0 ? 0 : (static_cast<u32>(norm_shift_batch1(sum)) & 0xffff);
  out = static_cast<i16>((sum << (sh & 0x1f)) >> 16);
  return wadd<i32>(wsub<i32>(exp, static_cast<i32>(sh)), 0x10);
}

void expand_dibits(u32& out0, u32& out1, u32 bits) {
  i32 src = static_cast<i32>(bits);
  for (int w = 0; w < 2; w++) {
    u32 acc = 0;
    i32 b = 30;
    for (int k = 0; k < 8; k++) {
      u32 d = static_cast<u32>(src & 3);
      src >>= 2;
      acc |= (d << (b & 31)) | (d << ((b - 2) & 31));
      b -= 4;
    }
    if (w == 0) out0 = acc; else out1 = acc;
  }
}

void voiced_offset(i16& out, u32 mask_a, i16 a, u32 mask_b, i16 b) {
  if ((mask_a & 0xaaaa'aaaau) == 0) { out = 0; return; }
  if (a > 0 && (mask_b & 0xaaaa'aaaau) != 0) {
    out = wsub<i16>(a, wmul<i16>(b, 256));
    return;
  }
  out = wmul<i16>(static_cast<i16>(static_cast<i8>(b)), 128);
}

i32 norm_exp_absmax(CS16 v, u16 n) {
  i16 m = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i16 x = v[i];
      i16 a = x < 0 ? (x == INT16_MIN ? static_cast<i16>(0x7fff) : static_cast<i16>(-x)) : x;
      if (m < a) m = a;
    }
  }
  i32 x = static_cast<i32>(m) << 16;
  if (x != 0) return -static_cast<i32>(static_cast<u32>(norm_shift_batch1(static_cast<u32>(x))) & 0xffff);
  return 0;
}

i32 norm_exp_max_tagged(CS16 vals, CS16 tags, i16 from, i16 to, i16 tag) {
  u32 best = 0;
  if (from < to) {
    usz count = static_cast<u16>(wsub<i16>(to, from));
    for (usz i = static_cast<usz>(from); i < static_cast<usz>(from) + count; i++) {
      u32 scaled = static_cast<u32>(static_cast<i32>(vals[i]) << 16);
      if (tags[i] == tag && static_cast<i32>(best) < static_cast<i32>(scaled)) best = scaled;
    }
  }
  if (best != 0) return -static_cast<i32>(static_cast<u32>(norm_shift_batch1(best)) & 0xffff);
  return 0;
}

u32 voicing_pattern(i16 idx, u32 prev, i16 nbits) {
  if (idx <= 0x37) return dsp_tbl::VOICING_LUT[idx];
  i16 nb = nbits > 8 ? 8 : nbits;
  u16 pat = 0;
  i16 s = 0;
  while (s < nb) {
    assert(nb >= 3);
    i16 rep = dsp_tbl::BIT_ALLOC[(static_cast<usz>(nb) - 3) * 8 + static_cast<usz>(s)];
    if (rep > 0) {
      u16 bit = static_cast<u16>(((static_cast<i32>(wsub<i16>(idx, 0x38))) >> (static_cast<u32>(s) & 0x1f)) & 1);
      for (u16 r = 0; r < static_cast<u16>(rep); r++) {
        pat = static_cast<u16>((static_cast<u16>(static_cast<i16>(pat) >> 2)) | (bit << 14));
      }
    }
    s++;
  }
  u32 x = (prev & 0x5555) ^ static_cast<u32>(pat);
  return (wmul<u32>(x, 4) & x) | (prev & 0x5555 & static_cast<u32>(pat)) |
         static_cast<u32>(static_cast<i32>(static_cast<i16>(pat)) << 16);
}

void fft_butterfly_pass(S16 buf, usz a_off, usz b_off, i16 stride, u16 n) {
  if (static_cast<i16>(n) <= 0) return;
  isz pa = static_cast<isz>(a_off), pb = static_cast<isz>(b_off);
  isz ti = 0;
  for (u16 k = 0; k < n; k++) {
    ti += stride;
    i32 c = tables::COS512[ti];
    i32 s = tables::COS512[128 + ti];
    i32 half = wshl<i32>(buf[pa], 16) >> 1;
    i32 rot = wsub<i32>(wmul<i32>(c, buf[pb]), wmul<i32>(s, buf[pb + 1]));
    buf[pa] = static_cast<i16>(static_cast<u32>(wadd<i32>(half, rot)) >> 16);
    i32 br = buf[pb];
    buf[pb] = static_cast<i16>(static_cast<u32>(wsub<i32>(half, rot)) >> 16);
    i32 rot2 = wadd<i32>(wmul<i32>(s, br), wmul<i32>(c, buf[pb + 1]));
    i32 half2 = wshl<i32>(buf[pa + 1], 16) >> 1;
    buf[pa + 1] = static_cast<i16>(static_cast<u32>(wadd<i32>(half2, rot2)) >> 16);
    buf[pb + 1] = static_cast<i16>(static_cast<u32>(wsub<i32>(rot2, half2)) >> 16);
    pa += 2;
    pb -= 2;
  }
}

i32 pow2_frac(i32 x, i16& exp) {
  exp = wadd<i16>(static_cast<i16>(x >> 16), 1);
  i32 t = static_cast<i32>((static_cast<u32>(x) & 0xffff) >> 1);
  auto step = [](i32 acc, i32 mul, u32 sh, u32 mask, i32 add) -> i16 {
    return static_cast<i16>(wadd<i32>((wadd<i32>(wmul<i32>(acc, mul), 0x8000) >> sh) & static_cast<i32>(mask), add) >> 16);
  };
  i16 r = step(t, 0xaec4, 3, 0xffff'e000u, static_cast<i32>(0x4ecb'0000u));
  r = step(wmul<i32>(wmul<i32>(r, t), 2), 1, 3, 0xffff'e000u, 0x38d6'0000);
  r = step(wmul<i32>(wmul<i32>(r, t), 2), 1, 2, 0xffff'c000u, 0x3d7f'8000);
  r = step(wmul<i32>(wmul<i32>(r, t), 2), 1, 1, 0xffff'8000u, 0x58b9'0000);
  return wadd<i32>((wadd<i32>(wmul<i32>(wmul<i32>(r, t), 2), 0x8000) >> 1) & static_cast<i32>(0xffff'8000u), 0x4000'0000);
}

void spread_peak(S16 arr, i16 peak, i16 level, i16 dec) {
  constexpr i16 FLOOR = -0x77ff;
  i16 start = level < FLOOR ? FLOOR : level;
  auto decay = [&](i16 cur) -> i16 {
    i32 d = sat_sub_batch1(wshl<i32>(cur, 16), wshl<i32>(dec, 16));
    i16 next = static_cast<i16>(d >> 16);
    return next < FLOOR ? FLOOR : next;
  };
  i16 cur = start;
  i32 i = peak - 1;
  while (i >= 0) {
    if (arr[i] < cur) arr[i] = cur;
    cur = decay(cur);
    i--;
  }
  cur = start;
  i16 first = wadd<i16>(peak, 1);
  if (first < 0x38) {
    for (i32 k = first; k < 0x38; k++) {
      if (arr[k] < cur) arr[k] = cur;
      cur = decay(cur);
    }
  }
}

static i16 addsub_norm(i16 a, i16 a_exp, i16 b, i16 b_exp, i16& out_exp, bool sub) {
  if (a != 0 && static_cast<i32>(b_exp) - static_cast<i32>(a_exp) < 0x20) {
    if (b == 0 || static_cast<i32>(a_exp) - static_cast<i32>(b_exp) > 0x1f) {
      out_exp = a_exp;
      if (!sub) return a;
      return a == INT16_MIN ? static_cast<i16>(0x7fff) : static_cast<i16>(-a);
    }
    i16 m = wadd<i16>(a_exp <= b_exp ? b_exp : a_exp, 1);
    u32 va = static_cast<u32>(shift32_al2b2(wmul<i32>(a, 0x10000), wsub<i16>(a_exp, m)));
    u32 vb = static_cast<u32>(shift32_al2b2(wmul<i32>(b, 0x10000), wsub<i16>(b_exp, m)));
    u32 sum = sub ? wsub<u32>(vb, va) : wadd<u32>(vb, va);
    i16 nrm = sum == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_exp_batch3(sum));
    i16 r = static_cast<i16>((sum << (static_cast<u32>(static_cast<i32>(nrm)) & 0x1f)) >> 16);
    if (r == 0) { out_exp = 0; return 0; }
    out_exp = wsub<i16>(m, nrm);
    return r;
  }
  if (b == 0) { out_exp = 0; return 0; }
  out_exp = b_exp;
  return b;
}

i16 sub_norm(i16 m1, i16 e1, i16 m2, i16 e2, i16& exp) { return addsub_norm(m1, e1, m2, e2, exp, true); }

void bit_reverse(S16 buf, i16 size) {
  const i16* tbl;
  switch (size) {
    case 8: tbl = dsp_tbl::BITREV8; break;
    case 16: tbl = dsp_tbl::BITREV16; break;
    case 32: tbl = dsp_tbl::BITREV32; break;
    case 64: tbl = dsp_tbl::BITREV64; break;
    default: tbl = dsp_tbl::BITREV128; break;
  }
  if (tbl[0] <= 0) return;
  isz pa = 0, pb = 0;
  for (usz k = 0; k < static_cast<u16>(tbl[0]); k++) {
    isz ia = pa + tbl[1 + 2 * k];
    pa = ia + 1;
    isz ib = pb + tbl[2 + 2 * k];
    pb = ib + 1;
    std::swap(buf[ia], buf[ib]);
    std::swap(buf[pa], buf[pb]);
  }
}

u32 shr_or_shl(i32 x, i16 sh) { return static_cast<u32>(shift_i32_batch1(x, sh)); }

u32 gather_dibits(i32 x) {
  i32 shift = 14;
  u16 out = 0;
  for (int k = 0; k < 8; k++) {
    u16 lo = static_cast<u16>(static_cast<u16>(x) & 3);
    x >>= 4;
    out = static_cast<u16>(out | static_cast<u16>(lo << (shift & 0x1f)));
    shift -= 2;
  }
  return 0xffff'0000u | out;
}

void vsub_sat(S16 a, CS16 b, u16 n) {
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 v = static_cast<i32>(a[i]) - static_cast<i32>(b[i]);
      if (v >= 0x8000) v = 0x7fff;
      else if (v < -0x7fff) v = -0x7fff;
      a[i] = static_cast<i16>(v);
    }
  }
}

i32 norm64(u32 lo, u32 hi) {
  if (lo == 0 && hi == 0) return 0;
  u32 p1 = lo, p2 = hi;
  if (static_cast<i32>(p2) < 1 && static_cast<i32>(p2) < 0) { p1 = ~p1; p2 = ~p2; }
  u32 u2 = 0, u1 = 0x80;
  i32 i = 0;
  for (;;) {
    if ((u2 & p1) != 0 || (u1 & p2) != 0) break;
    u2 = (u2 >> 1) | (u1 << 0x1f);
    i += 1;
    u1 = static_cast<u32>(static_cast<i32>(u1) >> 1);
    if (!(static_cast<i16>(i) < 0x28)) break;
  }
  return i - 9;
}

// ======================= from batch2 =======================

u32 shift_round16_batch2(i32 x, i16 sh) {
  i16 s = static_cast<i16>(sh - 0xf);
  u32 u = shr_or_shl(x, s);
  if (static_cast<i32>(u) < 0) {
    if (u == 0x8000'0000u) u = 0x7fff'ffffu; else u = wneg<u32>(u);
    if ((u & 0xffff'0000u) == 0x8000'0000u) return 0x7fff'ffffu;
    return wneg<u32>(u & 0xffff'0000u);
  }
  return u & 0xffff'0000u;
}

i32 cos_lookup(i16 x, CS16 table) {
  i32 u = wshl<i32>(x, 16) >> 6;
  if (u < 0) {
    if (static_cast<u32>(u) == 0x8000'0000u) u = 0x7fff'ffff; else u = wneg<i32>(u);
  }
  u32 uu = static_cast<u32>(u);
  usz idx0 = static_cast<usz>(static_cast<i16>(uu >> 16));
  usz idx1 = static_cast<usz>((wadd<u32>((static_cast<u32>(u >> 16)) & 0xffff, 1)) & 0x1ff);
  i32 frac = static_cast<i32>(uu & 0xffff);
  return (static_cast<i32>(table[idx0]) * (0x10000 - frac) + static_cast<i32>(table[idx1]) * frac) >> 16;
}

static void butterfly(S16 p1, S16 p2, usz p2_start, u16 n, bool flip) {
  if (static_cast<i16>(n) > 0) {
    usz i1 = 0, i2 = p2_start;
    for (u16 k = 0; k < n; k++) {
      i32 s1 = p2[i2], s2 = p2[i2 + 1], s3 = p1[i1], s4 = p1[i1 + 1];
      i32 d1, d2;
      if (flip) { d1 = s2 - s4; d2 = s3 - s1; } else { d1 = s4 - s2; d2 = s1 - s3; }
      p1[i1] = static_cast<i16>((s1 + s3) >> 1);
      p1[i1 + 1] = static_cast<i16>(d1 >> 1);
      p2[i2] = static_cast<i16>((s4 + s2) >> 1);
      p2[i2 + 1] = static_cast<i16>(d2 >> 1);
      i2 -= 2;  // usize wrap on the final step is never dereferenced
      i1 += 2;
    }
  }
}
void butterfly_a(S16 p1, S16 p2, usz p2_start, u16 n) { butterfly(p1, p2, p2_start, n, false); }
void butterfly_b(S16 p1, S16 p2, usz p2_start, u16 n) { butterfly(p1, p2, p2_start, n, true); }

void copy_params_batch2(S16 dst, CS16 src, i16 p3) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[3] = src[3];
  if (src[0] == 1) {
    dst[2] = src[2];
    dst[6] = src[6];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[0x45] = p3;
    dst[7] = src[7];
    i16 cnt = src[2];
    if (cnt > 0) {
      for (usz i = 0; i < static_cast<usz>(cnt); i++) dst[8 + i] = src[8 + i];
    }
    u16 uv1 = static_cast<u16>(src[2]);
    if (static_cast<i16>(uv1) < 0x38) {
      i16 fillval = src[static_cast<usz>(src[2]) + 7];
      u32 count = wsub<u32>(0x38u, uv1) & 0xffff;
      usz j = static_cast<usz>(uv1) + 8;
      for (u32 k = 0; k < count; k++) dst[j++] = fillval;
    }
    dst[0x43] = 0;
  } else {
    dst[4] = 0;
    dst[5] = 0;
    dst[0x45] = 0;
    dst[7] = 0;
    dst[0x43] = 0;
  }
}

i32 refine_pitch(i32 param_1, i16& out_r, i16& out_f3, i16& out_f4, i16& out_f5, i16 p6, i16 p7, i16 p8, i16 p9,
                 i16 p10) {
  out_r = p6;
  out_f3 = 0;
  out_f4 = 0;
  out_f5 = 0;
  i16 param_6 = p6;
  i32 iv5 = 0;
  u32 uv2 = 0;
  bool go_final = false;
  if (p8 == 0) {
    if (p9 == 0) {
      iv5 = 0;
      go_final = true;
    } else {
      out_r = p7;
      param_6 = p7;
      iv5 = static_cast<i32>(param_6) * static_cast<i32>(p10);
    }
  } else if (p9 == 0) {
    iv5 = static_cast<i32>(param_6) * static_cast<i32>(p10);
  } else {
    i16 param_7 = p7;
    i32 ip7 = p7, ip6 = p6;
    if (param_6 < static_cast<i16>(static_cast<u32>(wmul<i32>(ip7, 0x9998)) >> 16) &&
        static_cast<i16>(static_cast<u32>(wmul<i32>(ip7, 0x6666)) >> 16) < param_6 && param_6 < 0x4000) {
      param_7 = static_cast<i16>(p7 >> 1);
      out_f4 = 1;
    } else if (static_cast<i16>(static_cast<u32>(wmul<i32>(ip6, 0x6aaa)) >> 16) < param_7 &&
               param_7 < static_cast<i16>(static_cast<u32>(wmul<i32>(ip6, static_cast<i32>(0xa000u))) >> 16) &&
               param_7 < 0x4000) {
      param_7 = static_cast<i16>(static_cast<u32>(wshl<i32>(ip7, 0x11)) >> 16);
      out_f5 = 1;
    }
    out_f3 = static_cast<i16>(
        static_cast<u32>(wadd<i32>(wmul<i32>(param_7, 0x10000), wmul<i32>(ip6, -0x10000))) >> 16);
    i32 t = wadd<i32>(wshl<i32>(param_7, 16) >> 1, wmul<i32>(ip6, 0x10000) >> 1);
    iv5 = wmul<i32>(static_cast<i32>(static_cast<i16>(static_cast<u32>(t) >> 16)), p10);
  }
  if (!go_final) {
    uv2 = static_cast<u32>(wadd<i32>(wmul<i32>(iv5, 2) >> 4, param_1));
    if (uv2 == 0) {
      iv5 = 0;
      uv2 = 0xf;
    } else {
      u32 nrm = static_cast<u32>(norm_shift_batch1(uv2)) & 0xffff;
      iv5 = static_cast<i32>(wshl<u32>(uv2, nrm & 0x1f));
      uv2 = wsub<u32>(0xfu, nrm) & 0xffff;
    }
  }
  u32 diff = wsub<u32>(uv2, 0xfu);
  i16 s4 = static_cast<i16>(diff);
  return shift_i32_batch1(iv5, s4);
}

void unpack_nibbles(SU16 out, i16 p2, CS16 table) {
  i16 s8 = static_cast<i16>(p2 - 2);
  u32 uv5 = static_cast<u32>(wmul<i32>(static_cast<i32>(p2) - 9, static_cast<i32>(p2) + 6) >> 1);
  i16 uv3_init = static_cast<i16>(static_cast<i16>(static_cast<u16>(uv5)) >> 2);
  i32 iv2 = static_cast<i32>(wadd<u32>(uv5 & 0xffff, wmul<u32>(static_cast<u32>(static_cast<i32>(uv3_init)), 0xffff'fffcu)));
  i32 iv7 = static_cast<i32>(uv3_init) + 1;
  u32 shift0 = (static_cast<u32>(static_cast<u8>(iv2) & 7)) << 2;
  u16 uv3 = static_cast<u16>(static_cast<i32>(table[static_cast<usz>(uv3_init)]) >> shift0);
  usz oi = 0;
  while (s8 >= 0) {
    u16 uv4 = uv3 & 0xf;
    uv3 = static_cast<u16>(static_cast<i16>(uv3) >> 4);
    iv2 += 1;
    out[oi++] = uv4;
    s8 = static_cast<i16>(s8 - 1);
    if (static_cast<i16>(iv2) > 3) {
      i16 s6 = static_cast<i16>(iv7);
      iv7 += 1;
      uv3 = static_cast<u16>(table[static_cast<usz>(s6)]);
      iv2 = 0;
    }
  }
}

i32 top_bit_batch3(u32 v) { return 31 - static_cast<i32>((v ? __builtin_clz(v) : 32) & 31); }
i32 norm_exp_batch3(u32 x) { return norm_shift_batch1(x); }
i32 shift32_al2b2(i32 v, i16 s) { return shift_i32_batch1(v, s); }
i32 sat_add32(i32 a, i32 b) {
  i32 r = wadd<i32>(a, b);
  if (((r ^ a) & (r ^ b)) < 0) return static_cast<i32>(wadd<u32>(0x7fffffffu, a < 0 ? 1u : 0u));
  return r;
}
i16 high16_batch3(i32 x) { return static_cast<i16>(static_cast<u16>(static_cast<u32>(x) >> 16)); }

// ======================= from batch3 =======================

void bump_frame_counter(S16 state) {
  i16& c = state[FRAME_COUNTER_WORD];
  if (c < 100) c = static_cast<i16>(c + 1);
}

u16 pack_bits_i16(CS16 bits, i16 n) {
  u16 acc = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) acc = static_cast<u16>(static_cast<u16>(acc * 2) | static_cast<u16>(bits[i]));
  }
  return acc;
}

u16 pack_bits(CSU16 bits, i16 n) {
  u16 acc = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) acc = static_cast<u16>(static_cast<u16>(acc * 2) | bits[i]);
  }
  return acc;
}

void scale_vec(S16 dst, CS16 src, i16 n, i16 coef) {
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i32 p = wmul<i32>(wmul<i32>(src[i], coef), 2);
      dst[i] = high16_batch3(p);
    }
  }
}

usz copy16_batch3(S16 dst, CS16 src, i16 n) {
  if (n <= 0) return 0;
  usz un = static_cast<usz>(n);
  std::copy_n(src.begin(), un, dst.begin());
  return un;
}

i32 scale_ceil16(i32 x, i16 exp) {
  i16 s = wsub<i16>(exp, 0xf);
  i32 v = shift32_al2b2(x, s);
  i32 r = wadd<i32>(v, 0xffff);
  if (((r ^ v) & r) < 0) r = static_cast<i32>(wadd<u32>(0x7fffffffu, v < 0 ? 1u : 0u));
  return static_cast<i32>(static_cast<u32>(r) & 0xffff'0000u);
}

i32 peak_exp_unflagged(CS16 vals, CS16 flags, i16 n) {
  i32 best = 0;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i32 v = wmul<i32>(vals[i], 0x10000);
      if (flags[i] == 0 && best < v) best = v;
    }
  }
  if (best == 0) return 0;
  u32 m = best < 0 ? ~static_cast<u32>(best) : static_cast<u32>(best);
  i32 i = top_bit_batch3(m);
  return wneg<i32>(static_cast<i32>(wsub<u32>(0x1eu, static_cast<u32>(i)) & 0xffff));
}

static i32 div32_common(i32 num, i16 den, u32 q_shift) {
  i32 sign = static_cast<i32>(den) ^ num;
  i32 n = num < 0 ? (num == INT32_MIN ? INT32_MAX : -num) : num;
  i32 d = den < 0 ? (den == INT16_MIN ? static_cast<i32>(INT16_MAX) : static_cast<i32>(-den)) : static_cast<i32>(den);
  assert(d != 0 || n < d);
  i32 q = n < d ? 0 : ((n / d) >> q_shift);
  return sign < 0 ? -q : q;
}
i32 div32(i32 num, i16 den) { return div32_common(num, den, 0); }
i32 div32_half(i32 num, i16 den) { return div32_common(num, den, 1); }

i32 shl_sat(i32 x, i16 s) {
  if (s < 0) {
    if (s < -0x1e) return x >> 31;
    return x >> (static_cast<u32>(static_cast<u8>(-s)) & 0x1f);
  }
  u8 b = static_cast<u8>(s);
  i32 mask = static_cast<i32>(0xffff'ffffu << (wsub<u32>(0x1fu, b) & 0x1f));
  i32 kept = mask & x;
  if (x < 0) kept = wsub<i32>(kept, mask);
  if (kept != 0) return static_cast<i32>(wadd<u32>(0x7fffffffu, x < 0 ? 1u : 0u));
  return static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(b) & 0x1f));
}

void scale_vec_shift(S16 dst, CS16 src, i16 n, i16 coef, i16 exp_in, i16 exp_out) {
  if (n <= 0) return;
  i16 s = wsub<i16>(exp_in, exp_out);
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    i32 p = wmul<i32>(wmul<i32>(src[i], coef), 2);
    dst[i] = high16_batch3(shift_i32_batch1(p, s));
  }
}

i32 q15mul32(i32 v, i32 c) {
  return wadd<i32>(wmul<i32>(high16_batch3(v), wmul<i32>(c, 2)),
                   static_cast<i32>(wmul<u32>(static_cast<u32>(v) & 0xffff, static_cast<u32>(c))) >> 15);
}

i32 sqrt_norm(i32 x, i16& exp) {
  if (x == 0) return 0;
  i32 n = norm_exp_batch3(static_cast<u32>(x));
  u16 e = static_cast<u16>(wsub<i16>(exp, static_cast<i16>(n)));
  i32 m = high16_batch3(static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(n) & 0x1f)));
  i32 t = high16_batch3(wadd<i32>(wmul<i32>(wadd<i32>(wmul<i32>(m, -0x198a), static_cast<i32>(0x38a5'8000u)), 2), 0x8000));
  i32 acc = static_cast<i32>(static_cast<u32>(wadd<i32>(wmul<i32>(wadd<i32>(wmul<i32>(t, m), 0x141f'8000), 2), 0x8000)) &
                             0xffff'0000u);
  if (e & 1) acc = wmul<i32>(high16_batch3(acc), 0xb504);
  acc = static_cast<i32>(static_cast<u32>(wadd<i32>(acc, 0x8000)) & 0xffff'0000u);
  exp = static_cast<i16>((static_cast<i32>(static_cast<i16>(e)) + 1) >> 1);
  return acc;
}

void track_level(i16& val, i16& exp, i16 add, i16 add_exp) {
  i16 e0 = exp;
  i16 m = add_exp <= e0 ? e0 : add_exp;
  i32 a = shift32_al2b2(wmul<i32>(val, 0xf334), wsub<i16>(e0, m));
  i32 b = shift32_al2b2(wmul<i32>(add, 0xcccc), wsub<i16>(wsub<i16>(add_exp, m), 4));
  i32 sum = wadd<i32>(a, b);
  i16 nrm = sum == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_exp_batch3(static_cast<u32>(sum)));
  i16 ex = wsub<i16>(m, nrm);
  i32 v = static_cast<i32>(static_cast<u32>(sum) << (static_cast<u32>(static_cast<i32>(nrm)) & 0x1f));
  if (v == 0 || ex < -0x10) {
    v = 0x7fff'0000;
    ex = -0x11;
  }
  val = high16_batch3(v);
  exp = ex;
}

i16 quantize_codebook64(i16& x) {
  i32 a = wmul<i32>(x, 0x10000);
  i32 d0 = high16_batch3(sat_add32(a, wmul<i32>(dsp_tbl::GAIN_CODEBOOK[0], -0x10000)));
  i32 best = wmul<i32>(wmul<i32>(d0, d0), 2);
  i16 best_i = 0;
  for (usz i = 1; i < 64; i++) {
    i16 d = high16_batch3(sat_add32(a, wmul<i32>(dsp_tbl::GAIN_CODEBOOK[i], -0x10000)));
    i32 metric = d == INT16_MIN ? INT32_MAX : wmul<i32>(wmul<i32>(d, d), 2);
    if (metric < best) {
      best = metric;
      best_i = static_cast<i16>(i);
    }
  }
  x = dsp_tbl::GAIN_CODEBOOK[best_i];
  return best_i;
}

i16 div_norm(i32 num, i16 num_exp, i16 den, i16 exp_bias, i16& out_exp) {
  i32 sign = static_cast<i32>(den) ^ num;
  i32 n = num < 0 ? (num == INT32_MIN ? INT32_MAX : -num) : num;
  i32 d = den < 0 ? (den == INT16_MIN ? static_cast<i32>(INT16_MAX) : static_cast<i32>(-den)) : static_cast<i32>(den);
  i16 e = num_exp;
  if (wmul<i32>(d, 0x10000) <= n) {
    n >>= 1;
    e = wadd<i16>(e, 1);
  }
  assert(d != 0);
  i16 q = static_cast<i16>((n / d) >> 1);
  if (sign < 0) q = wneg<i16>(q);
  i32 v = wmul<i32>(q, 0x10000);
  i16 nrm = v == 0 ? static_cast<i16>(0) : static_cast<i16>(norm_exp_batch3(static_cast<u32>(v)));
  i16 m = high16_batch3(static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(nrm)) & 0x1f)));
  if (m == 0) {
    out_exp = 0;
    return 0;
  }
  out_exp = wsub<i16>(wsub<i16>(e, nrm), exp_bias);
  return m;
}

u32 map_voicing_bits(i16 bits, i32 nbands, i16 w0) {
  u32 acc = 0;
  i32 ctr = 7, limit = 7, centre = 0x0c00'0000;
  i32 doubled = 0;
  u16 last = 0;
  for (;;) {
    i32 u = wmul<i32>(wsub<i32>(wmul<i32>(high16_batch3(centre), w0), 0xece6), 0x10);
    i32 idx;
    if (u < 0) {
      idx = 0;
    } else {
      u32 e = u == 0 ? 0u : (wsub<u32>(0x1eu, static_cast<u32>(top_bit_batch3(static_cast<u32>(u)))) & 0xffff);
      i32 m = high16_batch3(static_cast<i32>(static_cast<u32>(u) << (e & 0x1f)));
      i32 sh = wsub<i32>(static_cast<i32>(wsub<u32>(0xfu, e) & 0xffff), 0x13);
      idx = shift32_al2b2(wmul<i32>(m, 0xb86c), static_cast<i16>(sh));
    }
    idx = wsub<i32>(wsub<i32>(nbands, idx >> 16), 1);
    if (static_cast<i32>(static_cast<i16>(idx)) < limit) idx = ctr;
    ctr -= 1;
    centre = wadd<i32>(centre, 0x1000'0000);
    doubled = wmul<i32>(static_cast<i32>(acc), 2);
    limit -= 1;
    last = static_cast<u16>(static_cast<u16>((static_cast<i32>(bits) >> (static_cast<u32>(idx) & 0x1f)) & 1) | static_cast<u16>(doubled));
    acc = last;
    if (static_cast<u16>(ctr) >= 0x8000) break;
  }
  return (static_cast<u32>(static_cast<u16>(high16_batch3(doubled))) << 16) | last;
}

void ramp_mix(std::span<i32> dst, std::span<const i32> src, i16 step, i16 step_exp, i16 n) {
  if (n <= 0) return;
  i32 base = wmul<i32>(step, 0x10000);
  i32 acc = 0;
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    i32 x = src[i];
    i32 g = high16_batch3(acc);
    i32 y = wadd<i32>(
        wsub<i32>(wmul<i32>(wmul<i32>(high16_batch3(x), g), -2),
                  static_cast<i32>(wmul<u32>(static_cast<u32>(x) & 0xffff, static_cast<u32>(g))) >> 15),
        x);
    dst[i] = sat_add32(y, dst[i]);
    acc = sat_add32(acc, shift32_al2b2(base, step_exp));
  }
}

i16 add_norm(i16 a, i16 a_exp, i16 b, i16 b_exp, i16& out_exp) { return addsub_norm(a, a_exp, b, b_exp, out_exp, false); }

static void radix2_stage(S16 xe, S16 xo, i16 stride, i16 n, bool scaled) {
  if (n <= 0) return;
  usz iters = static_cast<usz>(static_cast<u16>(static_cast<u16>(static_cast<u16>(n) - 1) >> 1) + 1);
  i32 st = stride;
  i32 j = 0;
  u32 sh = scaled ? 16u : 15u;
  usz b = 0;
  for (usz it = 0; it < iters; it++) {
    for (int k = 0; k < 2; k++) {
      i32 c = dsp_tbl::TWIDDLE[j];
      i32 s = dsp_tbl::TWIDDLE[j + 128];
      i32 yr = xo[b], yi = xo[b + 1];
      i32 t = wsub<i32>(wmul<i32>(c, yr), wmul<i32>(s, yi));
      i32 h = wmul<i32>(xe[b], 0x10000) >> 1;
      xe[b] = static_cast<i16>(wadd<i32>(h, t) >> sh);
      xo[b] = static_cast<i16>(wsub<i32>(h, t) >> sh);
      i32 u = wadd<i32>(wmul<i32>(c, yi), wmul<i32>(s, yr));
      i32 h2 = wmul<i32>(xe[b + 1], 0x10000) >> 1;
      xe[b + 1] = static_cast<i16>(wadd<i32>(h2, u) >> sh);
      xo[b + 1] = static_cast<i16>(wsub<i32>(h2, u) >> sh);
      j += st;
      b += 2;
    }
  }
}
void fft_butterfly_stage(S16 sum, S16 dif, i16 stride, i16 n) { radix2_stage(sum, dif, stride, n, true); }

void resonator3(std::span<i32> dst, std::span<const i32> src, std::span<i32> state, i16 n) {
  if (n <= 0) return;
  auto clamp32 = [](i64 v) { return static_cast<i32>(std::clamp<i64>(v, INT32_MIN, INT32_MAX)); };
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    i32 x = src[i];
    i64 acc1 = static_cast<i64>(q15mul32(state[1], 0x78ab)) - static_cast<i64>(state[0]) + static_cast<i64>(x);
    i32 sat1 = clamp32(acc1);
    state[1] = sat1;
    i32 s2 = state[2];
    state[0] = x;
    i32 s4 = state[4];
    i32 s3 = state[3];
    state[2] = sat1;
    state[4] = s2;
    i64 acc2 = acc1 + static_cast<i64>(s4) - 2 * static_cast<i64>(s2) + 2 * static_cast<i64>(q15mul32(s3, 0x7c20));
    i32 s5 = state[5];
    state[5] = s3;
    i64 acc3 = acc2 - static_cast<i64>(q15mul32(s5, 0x78af));
    i32 out = clamp32(acc3);
    state[3] = out;
    dst[i] = out;
  }
}

// ======================= from batch4 =======================

i32 shl_var(i32 x, i16 n) { return wshl<i32>(x, static_cast<u32>(static_cast<i32>(n)) & 0x1f); }
i32 shr_neg(i32 x, i16 n) {
  if (n < -0x1e) return x >> 31;
  return x >> (static_cast<u32>(static_cast<i32>(wneg<i16>(n))) & 0x1f);
}
i32 shift_var(i32 x, i16 n) { return n < 0 ? shr_neg(x, n) : shl_var(x, n); }
i32 msb(u32 x) { return 31 - (x ? __builtin_clz(x) : 32); }
i16 norm_shift_batch4(i32 x) {
  if (x == 0) return 0;
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  return static_cast<i16>(0x1e - msb(m));
}
i32 l_sub(i32 a, i32 b) {
  i32 neg = wneg<i32>(b);
  i32 r = wadd<i32>(a, neg);
  if (((~b ^ r) & (r ^ a)) < 0) return a < 0 ? INT32_MIN : INT32_MAX;
  return r;
}
i32 hi16_at_exp(i32 x, i16 exp) {
  i16 s = wsub<i16>(exp, 0xf);
  i32 v = s < 0 ? shr_neg(x, s) : shl_var(x, s);
  return static_cast<i32>(static_cast<u32>(v) & 0xffff'0000u);
}
i32 frac_at_exp(i32 x, i16 exp) {
  i16 s = wsub<i16>(exp, 0xf);
  i16 hi;
  if (s < 0) {
    if (s < -0x1e) hi = static_cast<i16>(x >> 31);
    else hi = static_cast<i16>((x >> (static_cast<u32>(static_cast<i32>(wneg<i16>(s))) & 0x1f)) >> 16);
  } else {
    hi = static_cast<i16>(shl_var(x, s) >> 16);
  }
  i32 v = static_cast<i32>(hi) << 16;
  if (v < 0) v = wadd<i32>(v, 0x10000);
  i16 back = wsub<i16>(0xf, exp);
  v = back < 0 ? shr_neg(v, back) : shl_var(v, back);
  i32 rem = wsub<i32>(x, v);
  return exp < 0 ? shr_neg(rem, exp) : shl_var(rem, exp);
}

i32 log2_q(i16 x, i16 exp) {
  const i16* C = dsp_tbl::LOG2_C;
  i32 acc = static_cast<i32>(x) << 16;
  i16 n = acc == 0 ? static_cast<i16>(0) : norm_shift_batch4(acc);
  i32 d = static_cast<i16>(wsub<i32>(shl_var(acc, n), static_cast<i32>(C[0]) << 16) >> 16);
  i32 t = wmul<i32>(C[1], d);
  t = wadd<i32>(wadd<i32>(wmul<i32>(t, 2), 0x8000) & static_cast<i32>(0xffff'0000u), static_cast<i32>(C[2]) << 16);
  t = wmul<i32>(static_cast<i16>(t >> 16), d);
  t = wadd<i32>(wadd<i32>(wmul<i32>(t, 2), 0x8000) & static_cast<i32>(0xffff'0000u), static_cast<i32>(C[3]) << 16);
  t = wmul<i32>(static_cast<i16>(t >> 16), d);
  t = wadd<i32>((wadd<i32>(wmul<i32>(t, 2), 0x8000) >> 1) & static_cast<i32>(0xffff'8000u), static_cast<i32>(C[4]) << 16);
  t = wmul<i32>(static_cast<i16>(t >> 16), d);
  t = wadd<i32>(wadd<i32>(wmul<i32>(t, 8), 0x20000) & static_cast<i32>(0xfffc'0000u), static_cast<i32>(C[5]) << 16);
  return wadd<i32>(t >> 15, wshl<i32>(wsub<i16>(exp, n), 16));
}

i32 cplx_block_exp(CS16 x, i16 n) {
  i32 max = 0x0100'0000;
  if (n > 0) {
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i32 re = x[2 * i], im = x[2 * i + 1];
      i32 v = wadd<i32>(wmul<i32>(re, re), wmul<i32>(im, im));
      if (max < v) max = v;
    }
  }
  if (max == 0) return -1;
  return static_cast<i32>(static_cast<u16>(norm_shift_batch4(max))) - 1;
}

i16 vq_search(CS16 target, CS16 book, i16 n, i16 cand, i16 stride) {
  if (cand <= 0) return 0;
  i16 best = 0;
  i32 min = INT32_MAX;
  for (i16 k = 0; k < cand; k++) {
    i32 acc = 0;
    if (n > 0) {
      isz off = static_cast<isz>(k) * static_cast<isz>(stride);
      for (isz i = 0; i < n; i++) {
        i32 a = static_cast<i32>(target[i]) << 16;
        i32 b = static_cast<i32>(book[off + i]) << 16;
        i16 d = static_cast<i16>(l_sub(a, b) >> 16);
        i32 sq = d == INT16_MIN ? INT32_MAX : wmul<i32>(wmul<i32>(d, d), 2);
        i64 s = static_cast<i64>(acc) + sq;  // saturating_add
        acc = s > INT32_MAX ? INT32_MAX : static_cast<i32>(s);
      }
    }
    if (acc < min) {
      min = acc;
      best = k;
    }
  }
  return best;
}

i16 unpack_bits(i16 word, S16 dst, i16 n) {
  i16 w = word;
  i32 i = static_cast<i16>(wsub<i16>(n, 1));
  while (i >= 0) {
    dst[static_cast<usz>(i)] = static_cast<i16>(w & 1);
    w = static_cast<i16>(w >> 1);
    i--;
  }
  return n;
}

i16 copy16_batch4(S16 dst, S16 src, i16 n, bool store) {
  if (n <= 0) return 0;
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    if (store) src[i] = dst[i]; else dst[i] = src[i];
  }
  return n;
}

void spread_hold(S16 dst, CS16 src, i16 n) {
  dst[0] = src[0];
  for (usz i = 1; i <= 0x38; i++) dst[i] = src[i - 1];
  if (n > 0x39) {
    i16 pad = src[0x37];
    for (usz i = 0x39; i < static_cast<usz>(n); i++) dst[i] = pad;
  }
}
void spread_hold_alt(S16 dst, i16 n, CS16 src) { spread_hold(dst, src, n); }

i16 interleave_walk(S16 out, usz& out_pos, usz out_end, S16 data, isz& data_pos, CS16 steps, isz& step_pos, i16& count,
                    isz base_pos, i16 off, i16 limit, bool store) {
  while (out_pos < out_end) {
    i16 s = wsub<i16>(steps[static_cast<usz>(step_pos)], off);
    if (s >= 0) {
      usz idx = static_cast<usz>(data_pos + s);
      if (store) data[idx] = out[out_pos]; else out[out_pos] = data[idx];
      out_pos += 1;
    }
    i16 d = steps[static_cast<usz>(step_pos)];
    step_pos += 1;
    count = wadd<i16>(count, 1);
    data_pos += d;
    if (static_cast<i32>(limit) + 1 <= static_cast<i32>(count)) {
      count = 1;
      off = wsub<i16>(off, 1);
      step_pos -= limit;
      data_pos = base_pos;
    }
  }
  return off;
}

void imbe_bit_alloc(A13 t) {
  t[0] = 0; t[1] = 1; t[3] = 0; t[4] = 12; t[5] = 8; t[6] = 6; t[7] = 0x3e;
  t[8] = 0; t[9] = 0; t[10] = 0; t[11] = 0; t[12] = 0;
  i16 total = wadd<i16>(t[5], t[4]);
  t[2] = total;
  for (int i = 0; i < 7; i++) {
    total = wadd<i16>(total, t[6 + i]);
    t[2] = total;
  }
}

void ambe_bit_alloc(A13 t, i16 mode) {
  mode = std::clamp<i16>(mode, 0, 4);
  t[3] = mode; t[0] = 1; t[1] = 1; t[4] = 5; t[5] = 7; t[6] = 5; t[7] = 9;
  t[8] = 7; t[9] = 5; t[10] = 4; t[11] = 4; t[12] = 3;
  switch (mode) {
    case 1: t[12] = 2; break;
    case 3: t[11] = 3; t[12] = 1; break;
    case 4: t[10] = 3; t[11] = 3; t[12] = 1; break;
    case 2: t[12] = 1; break;
    default: break;
  }
  i16 total = wadd<i16>(t[4], t[5]);
  t[2] = total;
  for (int i = 0; i < 7; i++) {
    total = wadd<i16>(total, t[6 + i]);
    t[2] = total;
  }
}

void voicing_expand(S16 dst, i32 mask, i16 step, i16 n) {
  i16 bands[16];
  i32 m = mask;
  for (auto& b : bands) {
    b = static_cast<i16>(static_cast<u16>(m) & 3);
    m >>= 2;
  }
  i32 inc = (static_cast<i32>(step) << 6) >> 4;
  if (n > 0) {
    i32 acc = inc;
    for (usz i = 0; i < static_cast<usz>(n); i++) {
      i16 lo = static_cast<i16>(acc >> 16);
      if (lo > 0xf) lo = 0xf;
      i16 hi = wadd<i16>(lo, 1);
      i16 v = bands[lo];
      if (hi > 0xf) hi = 0xf;
      v = static_cast<i16>(v | bands[hi]);
      if (v > 2) v = 1;
      dst[i] = v;
      acc = wadd<i32>(acc, inc);
    }
  }
  if (n < 0x38) {
    for (usz i = static_cast<usz>(std::max<i16>(n, 0)); i < 0x38; i++) dst[i] = 0;
  }
}

// ======================= from batch5 =======================

static inline i32 satadd32(i32 a, i32 b) {
  i64 s = static_cast<i64>(a) + static_cast<i64>(b);
  return s > INT32_MAX ? INT32_MAX : (s < INT32_MIN ? INT32_MIN : static_cast<i32>(s));
}

void mac_ramp(std::span<i32> dst, std::span<const i32> src, i16 step, i16 shift, i16 n) {
  if (n <= 0) return;
  i32 base = static_cast<i32>(step) << 16;
  i32 phase = 0;
  for (usz i = 0; i < static_cast<usz>(n); i++) {
    i32 s = src[i];
    i32 g = static_cast<i16>(phase >> 16);
    i32 lo = static_cast<i32>(wmul<u32>(static_cast<u32>(s) & 0xffff, static_cast<u32>(g))) >> 15;
    i32 hi = wmul<i32>(wmul<i32>(static_cast<i16>(s >> 16), g), 2);
    dst[i] = satadd32(dst[i], wadd<i32>(lo, hi));
    phase = satadd32(shift_var(base, shift), phase);
  }
}

void bfly_pairs(S16 x, i16 n, i16 mode) {
  if (n <= 0) return;
  u32 cnt = static_cast<u16>(static_cast<u16>(static_cast<u16>(wsub<i16>(n, 1)) >> 1) + 1);
  int sh = mode == -2 ? 2 : (mode == 0 ? 0 : 1);
  usz b = 0;
  for (u32 k = 0; k < cnt; k++) {
    i32 a0 = x[b], a2 = x[b + 2];
    x[b] = static_cast<i16>((a0 + a2) >> sh);
    x[b + 2] = static_cast<i16>((a0 - a2) >> sh);
    i32 a1 = x[b + 1], a3 = x[b + 3];
    x[b + 1] = static_cast<i16>((a1 + a3) >> sh);
    x[b + 3] = static_cast<i16>((a1 - a3) >> sh);
    b += 4;
  }
}

void fft_stage(S16 xe, S16 xo, i16 stride, i16 n) { radix2_stage(xe, xo, stride, n, false); }

void refine_mags(S16 state) {
  u16 l_raw = static_cast<u16>(state[2]);
  i16 l = static_cast<i16>(l_raw);
  i16 exps[56] = {}, mags[56] = {};
  if (l > 0) {
    i16 e0 = state[66];
    for (usz i = 0; i < static_cast<usz>(l); i++) {
      exps[i] = e0;
      mags[i] = state[8 + i];
    }
  }
  u16 u9 = l_raw;
  i16 f0 = state[6];
  i32 acc = wmul<i32>(wmul<i32>(l, f0), 0x2000);
  i16 hi = static_cast<i16>(acc >> 16);
  u16 u7 = l_raw;
  while (hi > 0x4ccd) {
    i16 e0 = state[66];
    u7 = wsub<u16>(u7, 1);
    i32 v = hi;
    i32 m = static_cast<i16>(wmul<i32>(v, 0xd556) >> 16);
    i32 prod = wmul<i32>(wmul<i32>(m, mags[static_cast<usz>(static_cast<i16>(u7))]), 2);
    i16 sh = prod == 0 ? static_cast<i16>(0) : norm_shift_batch4(prod);
    i32 next = wsub<i32>(v << 16, (static_cast<i32>(f0) << 16) >> 3);
    exps[static_cast<usz>(static_cast<i16>(u7))] = wadd<i16>(wsub<i16>(e0, sh), 1);
    mags[static_cast<usz>(static_cast<i16>(u7))] = static_cast<i16>(shl_var(prod, sh) >> 16);
    hi = static_cast<i16>(next >> 16);
  }
  i16 max = exps[0];
  if (static_cast<i16>(u9) > 1) {
    for (usz i = 1; i < u9; i++) if (max < exps[i]) max = exps[i];
  }
  if (static_cast<i16>(u9) > 0) {
    i16 i = 0;
    for (;;) {
      i32 v = static_cast<i32>(static_cast<u16>(mags[i])) << 16;
      i16 d = wsub<i16>(exps[i], max);
      i16 r;
      if (d < 0) {
        if (d < -0x1e) r = static_cast<i16>(mags[i] >> 15);
        else r = static_cast<i16>((v >> (static_cast<u32>(static_cast<i32>(wneg<i16>(d))) & 0x1f)) >> 16);
      } else {
        r = static_cast<i16>(shl_var(v, d) >> 16);
      }
      state[8 + static_cast<usz>(i)] = r;
      i = static_cast<i16>(i + 1);
      if (i >= state[2]) break;
    }
  }
  state[66] = max;
}

i32 top_bit_al2b5(u32 v) { return top_bit_batch3(v); }
i32 norm_exp_l1_batch1(i32 x) { return norm_shift_batch1(static_cast<u32>(x)); }
i32 shift32_al3b2(i32 v, i16 s) { return shift_i32_batch1(v, s); }

void dec_ctrl(S16 state, CS16 ctrl) {
  constexpr usz D = 0x614 / 2;
  copy16_batch3(state.subspan(D), ctrl.first(14), 14);
  state[D + 14] = ctrl[14];
  state[D + 15] = ctrl[15];
  state[D + 16] = ctrl[16];
  copy16_batch3(state.subspan(D + 17), ctrl.subspan(17, 7), 7);
  state[D + 24] = ctrl[24];
  state[D + 25] = ctrl[25];
}

i32 sqrt_to_exp(i32 x, i16 exp, i16 target) {
  i16 e = exp;
  i32 m = sqrt_norm(x, e);
  i16 s = wsub<i16>(e, target);
  return wadd<i32>(shift32_al3b2(m, s), 0x8000) >> 16;
}

void scale_block_exp(S16 dst, CS16 src, u16 n, i32 e_from, i32 e_to) {
  if (static_cast<i16>(n) > 0) {
    i16 s = static_cast<i16>(wsub<i32>(e_from, e_to));
    for (usz i = 0; i < n; i++) dst[i] = static_cast<i16>(shl_sat(static_cast<i32>(src[i]) << 16, s) >> 16);
  }
}

void ceil_and_residual(i16& out_int, i16& out_frac, i16 x, i16 exp) {
  i32 v = static_cast<i32>(x) << 16;
  i16 n = static_cast<i16>(scale_ceil16(v, exp) >> 16);
  out_int = n;
  i16 s = wsub<i16>(exp, 0xf);
  i32 shifted = shift32_al3b2(v, s);
  out_frac = static_cast<i16>(wshl<i32>(wsub<i32>(static_cast<i32>(n) << 16, shifted), 15) >> 16);
}

void avg_and_extend(S16 a, CS16 b, u16 n) {
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 s = wadd<i32>((static_cast<i32>(b[i]) << 16) >> 1, (static_cast<i32>(a[i]) << 16) >> 1);
      i16 v = static_cast<i16>(shl_sat(s, 1) >> 16);
      if (v > 0x77ff) v = 0x77ff;
      else if (v < -0x7800) v = -0x7800;
      a[i] = v;
    }
  }
  i16 ns = static_cast<i16>(n);
  i16 last = a[static_cast<usz>(ns - 1)];
  if (ns < 0x38) {
    for (usz i = static_cast<usz>(ns); i < 0x38; i++) a[i] = last;
  }
}

void nearest_codeword(u16 pattern, i16& best, i16& best_idx, i16 from, i16 to, CSU16 table) {
  if (from > to) return;
  i16 i = from;
  for (;;) {
    u16 x = static_cast<u16>(table[static_cast<u16>(i)] ^ pattern);
    if ((static_cast<u32>(x) & 0xfc00) == 0) {
      i16 d = static_cast<i16>(popcount_low(x, 10));
      if (d < best) {
        best = d;
        best_idx = i;
      }
    }
    if (i == to) break;
    i = wadd<i16>(i, 1);
  }
}

void voicing_setup(CS16 cfg, S16 dst) {
  switch (cfg[0]) {
    case 1: {
      u32 mask = static_cast<u32>(static_cast<u16>(cfg[4])) | (static_cast<u32>(static_cast<u16>(cfg[5])) << 16);
      voicing_expand(dst, static_cast<i32>(mask), cfg[6], cfg[2]);
      break;
    }
    case 2:
      fill16(dst, 0, 0x38);
      break;
    case 3: {
      fill16(dst, 0, 0x38);
      i32 cls = frame_size_class(cfg[1]);
      if (cls == 0) {
        dst[static_cast<usz>(cfg[3])] = 1;
      } else if (static_cast<u32>(cls - 1) < 3) {
        u16 u = static_cast<u16>(cfg[3]);
        dst[u >> 8] = 1;
        dst[u & 0xff] = 1;
      }
      break;
    }
    default: break;
  }
}

void voicing_from_table(i16 mode, u32& out0, u32& out1, i16 idx, i32 sh, std::span<const u32> table) {
  if (mode == 1) {
    u32 c = static_cast<u32>(wsub<i32>(7, sh)) & 0x1f;
    i16 i = static_cast<i16>(static_cast<u16>(static_cast<u32>(static_cast<u16>(idx)) << c));
    expand_dibits(out0, out1, table[static_cast<usz>(i)]);
  } else {
    out1 = 0;
    out0 = 0;
  }
}

void hilbert_diff(S16 out, CS16 src, u16 n, CS16 taps) {
  if (static_cast<i16>(n) > 0) {
    usz base = 0x13;
    for (usz i = 0; i < n; i++) {
      i32 acc = 0;
      for (usz k = 0; k < 10; k++) {
        i32 t = taps[k];
        i32 hi = src[base + 2 + 2 * k];
        i32 lo = src[base - 2 * k];
        acc = wsub<i32>(wadd<i32>(acc, wmul<i32>(wmul<i32>(hi, t), 2)), wmul<i32>(wmul<i32>(lo, t), 2));
      }
      out[i] = static_cast<i16>(frac_at_exp(acc, 1) >> 16);
      base += 1;
    }
  }
  i16 ns = static_cast<i16>(n);
  fill16(out.subspan(static_cast<usz>(ns)), 0, wsub<i16>(0x38, ns));
}

i32 harmonic_index(i16& index, i16 l, i16 w, i16 cap, i16 enable) {
  i32 lv = static_cast<i32>(l) << 16;
  u16 n1 = lv == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(lv));
  i16 e1 = 0;
  i16 recip = div_norm(0x4100'0000, 8,
                       static_cast<i16>(static_cast<i32>(static_cast<u32>(lv) << (static_cast<u32>(n1) & 0x1f)) >> 16),
                       static_cast<i16>(wsub<i32>(0xf, n1)), e1);
  u16 recip_u = static_cast<u16>(recip);
  if (enable != 0) {
    i32 v = wadd<i32>((static_cast<i32>(w) << 16) >> 1, 0x2080'0000);
    u16 n2 = v == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_exp_l1_batch1(v));
    i16 e2 = 0;
    i16 r = div_norm(static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(n2) & 0x1f)),
                     static_cast<i16>(wsub<i32>(8, n2)), static_cast<i16>(recip_u), e1, e2);
    i32 sh = shift32_al3b2(static_cast<i32>(r) << 16, wsub<i16>(e2, 0xf));
    i16 d = static_cast<i16>(sh >> 16);
    if (static_cast<i32>(d) > static_cast<i32>(l) - 1) d = static_cast<i16>(static_cast<i32>(l) - 1);
    else if (d < 0) d = 0;
    if (d > cap) d = cap;
    else if (d < 0) d = 0;
    index = d;
  }
  i32 base = shift32_al3b2(-0x4100'0000, wsub<i16>(-7, e1));
  i16 t = static_cast<i16>(wadd<i32>(static_cast<i32>(index) << 17, 0x10000) >> 16);
  i32 acc = wadd<i32>(base, wmul<i32>(wmul<i32>(t, static_cast<i16>(recip_u)), 2));
  return wadd<i32>(shift32_al3b2(acc, wadd<i16>(e1, 7)), 0x8000) >> 16;
}

void synth_harmonic(i32& phase, CS16 table, i16 step, i32 taper_phase, usz& cursor, std::span<i32> dst, i16 shift,
                    i16 taper_step, u16 n, CS16 cos_tbl) {
  if (static_cast<i16>(n) <= 0) return;
  i32 inc = (static_cast<i32>(step) << 16) >> 11;
  i32 tp = taper_phase;
  usz p = cursor;
  for (u16 k = 0; k < n; k++) {
    i32 acc = phase;
    phase = wadd<i32>(acc, inc);
    u16 hi = static_cast<u16>(static_cast<u32>(acc) >> 16);
    usz idx = hi & 0xff;
    i16 frac = static_cast<i16>(wshl<i32>(wsub<i32>(acc, static_cast<i32>(static_cast<i16>(hi)) << 16), 15) >> 16);
    i32 t0 = (static_cast<i32>(table[idx]) << 16) >> 1;
    i32 t1 = (static_cast<i32>(table[idx + 1]) << 16) >> 1;
    i16 d = static_cast<i16>(wsub<i32>(t1, t0) >> 16);
    i32 v = wadd<i32>(t0, wmul<i32>(wmul<i32>(d, frac), 2));
    if (taper_step != 0) {
      i16 c = static_cast<i16>(cos_lookup(static_cast<i16>(((tp << 10) >> 16)), cos_tbl));
      i16 g = static_cast<i16>(wshl<i32>(wsub<i32>(0x8000, c), 15) >> 16);
      i32 lo = static_cast<i32>(static_cast<u32>(v) & 0xffff);
      i32 vhi = static_cast<i16>(v >> 16);
      v = wadd<i32>(wmul<i32>(lo, g) >> 15, wmul<i32>(wmul<i32>(vhi, g), 2));
      tp = wadd<i32>(tp, static_cast<i32>(taper_step) << 16);
    }
    v = shift32_al3b2(v, shift);
    i32 cur = dst[p];
    i32 r = wadd<i32>(cur, v);
    dst[p] = ((r ^ cur) & (r ^ v)) < 0 ? static_cast<i32>(wadd<u32>(0x7fff'ffffu, v < 0 ? 1u : 0u)) : r;
    p += 1;
  }
  cursor = p;
}

i32 norm_shift_l1_batch2(u32 x) { return 0x1e - top_bit_l1_batch5(x); }
u32 norm_exp_l1_batch2(u32 x) { return static_cast<u32>(nrm(x)); }
i16 high16_al2b1(i32 x) { return high16_batch3(x); }

i32 pow2_scaled(i16 x, i16 in_exp, i16 out_exp) {
  i32 v = shift_i32_batch1(static_cast<i32>(x) << 16, wsub<i16>(in_exp, 0xf));
  i16 e = 0;
  i32 m = pow2_frac(v, e);
  return wadd<i32>(shift_i32_batch1(m, wsub<i16>(e, out_exp)), 0x8000) >> 16;
}

void init_agc_config(S16 st) {
  auto wr32 = [&](usz w, u32 v) {
    st[w] = static_cast<i16>(static_cast<u16>(v));
    st[w + 1] = static_cast<i16>(static_cast<u16>(v >> 16));
  };
  wr32(0, 0x00ff'0001);
  wr32(2, 0x0000'000f);
  wr32(4, 0);
  wr32(6, 0x0000'4027);
  fill16(st.subspan(8), 0, 0x38);
  wr32(64, 0x3800'3333);
  wr32(66, 0);
  wr32(68, 0);
  fill16(st.subspan(70), 0, 3);
}

u32 band_count_for_mode(i32 mode, i16 a, i16 b, CS16 cos_tbl) {
  if (mode == 0) {
    i32 r = wsub<i32>(scale_ceil16(wmul<i32>(a, 0xa000), 0xc), 0x10000);
    if (r < 0) return 0;
    return static_cast<u16>(r >> 16);
  }
  if (wsub<u32>(static_cast<u32>(mode), 1u) < 3) {
    i32 idx = 300 + static_cast<i32>(b) + static_cast<i32>(a) * 2;
    return static_cast<u16>(wsub<i16>(cos_tbl[static_cast<usz>(idx)], 1));
  }
  return 0;
}

i32 ratio_q16(i16 num, i16 den) {
  if (num == den) return 0x10000;
  i32 d32 = static_cast<i32>(den) << 16;
  u32 n = norm_exp_l1_batch2(static_cast<u32>(d32));
  i32 dn = static_cast<i32>(static_cast<u32>(d32) << (n & 0x1f));
  i32 acc = static_cast<i32>(num) << 16;
  i32 e = -4;
  if (high16_al2b1(dn) <= num) {
    acc >>= 1;
    e = -3;
  }
  i16 q = static_cast<i16>(div32_half(acc, high16_al2b1(dn)));
  i32 v = static_cast<i32>(q) << 16;
  i16 sh = static_cast<i16>(e - static_cast<i32>(wsub<u32>(wsub<u32>(0u, n), 4u) & 0xffff) - 0xf);
  return shift_i32_batch1(v, sh);
}

void shift_copy16(S16 dst, CS16 src, u16 n, i16 sh) {
  if (static_cast<i16>(n) <= 0) return;
  for (usz i = 0; i < n; i++) dst[i] = high16_al2b1(shift_i32_batch1(static_cast<i32>(src[i]) << 16, sh));
}

void add_recip_split(i16& int_out, i16& frac_out, i16 a, i16 a_exp, i16 d) {
  i32 v = static_cast<i32>(d) << 16;
  u32 n = norm_exp_l1_batch2(static_cast<u32>(v));
  i16 e1 = 0;
  i16 q = div_norm(0x4000'0000, 1, high16_al2b1(static_cast<i32>(static_cast<u32>(v) << (n & 0x1f))),
                   static_cast<i16>(-4 - static_cast<i32>(n)), e1);
  i16 e2 = 0;
  i16 m = add_norm(a, a_exp, q, e1, e2);
  i32 acc = static_cast<i32>(m) << 16;
  i16 hi = high16_al2b1(hi16_at_exp(acc, e2));
  int_out = hi;
  i32 shifted = shift_i32_batch1(acc, wsub<i16>(e2, 0xf));
  frac_out = high16_al2b1(wshl<i32>(wsub<i32>(shifted, static_cast<i32>(hi) << 16), 0xf));
}

i32 resample_lerp(S16 out, u16 n, CS16 table, i16 phase) {
  i32 d = static_cast<i16>(div32_half(0x8'0000, static_cast<i16>(n)));
  i32 t = static_cast<i16>((static_cast<i32>(phase) << 0x19) >> 0x10);
  i32 step = wmul<i32>(wmul<i32>(d, t), 2) >> 0xc;
  if (static_cast<i16>(n) <= 0) return 0;
  i32 acc = 0;
  i32 ph = step;
  for (usz k = 0; k < n && k < out.size(); k++) {
    i16 hi = high16_al2b1(ph);
    i32 v;
    if (hi >= 0 && hi < 0x37 && hi == 0) {
      v = static_cast<i32>(table[0]) << 16;
    } else {
      usz idx = hi < 0x37 ? static_cast<usz>(static_cast<u16>(hi)) : 0x37;
      i32 f = static_cast<i32>((static_cast<u32>(ph >> 1)) & 0x7fff);
      i32 prod = wmul<i32>(wmul<i32>(f, table[idx - 1]), 2);
      i32 r = prod == INT32_MIN ? INT32_MAX : wneg<i32>(prod);
      v = wadd<i32>(r, wmul<i32>(wadd<i32>(wmul<i32>(table[idx], f), static_cast<i32>(table[idx - 1]) << 0xf), 2));
    }
    ph = wadd<i32>(ph, step);
    out[k] = high16_al2b1(v);
    acc = wadd<i32>(acc, v >> 6);
  }
  return acc;
}

void pow2_vector(S16 dst, i16& out_exp, CS16 src, u16 n) {
  fill16(dst, 0, static_cast<i16>(n));
  i16 exps[56] = {};
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) dst[i] = high16_al2b1(pow2_frac((static_cast<i32>(src[i]) << 16) >> 10, exps[i]));
  }
  i16 mx = exps[0];
  if (static_cast<i16>(n) > 1) {
    for (usz i = 1; i < n && i < 56; i++) if (mx < exps[i]) mx = exps[i];
  }
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 x = static_cast<i32>(static_cast<u32>(static_cast<u16>(dst[i])) << 16);
      i16 sh = wsub<i16>(exps[i], mx);
      dst[i] = high16_al2b1(shift_i32_batch1(x, sh));
    }
  }
  out_exp = mx;
}

i32 fft_merge_stage(S16 buf, i32 count, i16 log2n, i16 variant, i16 stage_flag) {
  i16 size = static_cast<i16>(static_cast<u16>(1u << (static_cast<u32>(log2n) & 0x1f)));
  u16 n = static_cast<u16>(wsub<i16>(static_cast<i16>(size >> 2), 1));
  i16 stride = static_cast<i16>(static_cast<u16>(0x200 >> (static_cast<u32>(log2n) & 0x1f)));
  usz sz = static_cast<usz>(size);
  if (stage_flag == 0) {
    i32 a = (static_cast<i32>(buf[1]) << 16) >> 1;
    i32 b = (static_cast<i32>(buf[0]) << 16) >> 1;
    buf[0] = high16_al2b1(wadd<i32>(b, a));
    buf[1] = high16_al2b1(wsub<i32>(b, a));
  } else if (variant == 0) {
    i32 a = (static_cast<i32>(buf[0]) << 16) >> 1;
    i32 b = (static_cast<i32>(buf[1]) << 16) >> 1;
    buf[0] = high16_al2b1(wadd<i32>(b, a));
    buf[sz] = high16_al2b1(wsub<i32>(a, b));
    buf[sz + 1] = 0;
    buf[1] = 0;
  } else {
    i32 a = (static_cast<i32>(buf[0]) << 16) >> 2;
    i32 b = (static_cast<i32>(buf[sz]) << 16) >> 2;
    buf[0] = high16_al2b1(wadd<i32>(b, a));
    buf[1] = high16_al2b1(wsub<i32>(a, b));
  }
  usz half = static_cast<usz>(static_cast<i32>(size) >> 1);
  {
    S16 front = buf.first(half).subspan(2);
    S16 back = buf.subspan(half);
    usz p2_start = sz - 2 - half;
    if (variant == 0) butterfly_a(front, back, p2_start, n);
    else butterfly_b(front, back, p2_start, n);
  }
  fft_butterfly_pass(buf, 2, sz - 2, stride, n);
  buf[half] = static_cast<i16>(((static_cast<i32>(buf[half]) << 16)) >> 17);
  i32 x = static_cast<i32>(buf[half + 1]) << 16;
  buf[half + 1] = x == INT32_MIN ? static_cast<i16>(INT32_MAX >> 17) : static_cast<i16>(wneg<i32>(x) >> 17);
  return wadd<i32>(count, 1);
}

// ======================= from batch6 =======================

void voicing_gains(i16& u_mant, i16& u_exp, i16& v_mant, i16& v_exp, i16 amp, i16 amp_exp, i16 cross, i16 cross_exp,
                   i16 gain) {
  i32 a = amp;
  u16 common = static_cast<u16>(wadd<i16>(wmul<i16>(amp_exp, 2), 1));
  i32 e2 = shift_i32_batch1(wmul<i32>(wmul<i32>(a, a), 2), wsub<i16>(wmul<i16>(amp_exp, 2), static_cast<i16>(common)));
  i16 sum, dif;
  if (cross == 0) {
    dif = high16_al2b1(e2);
    sum = dif;
  } else {
    i32 e3 = shift_i32_batch1(wmul<i32>(wmul<i32>(cross, cross), 2),
                              wsub<i16>(wmul<i16>(cross_exp, 2), static_cast<i16>(common)));
    sum = high16_al2b1(wadd<i32>(e3, e2));
    dif = high16_al2b1(wsub<i32>(e2, e3));
    if (dif == 0) {
      u_mant = 0;
      u_exp = 0;
      v_mant = 0x7fff;
      v_exp = 0;
      return;
    }
  }
  i32 den = wmul<i32>(wmul<i32>(high16_al2b1(wmul<i32>(wmul<i32>(dif, a), 2)), gain), 2);
  i16 nd = static_cast<i16>(norm_exp_l1_batch2(static_cast<u32>(den)));
  i16 recip = static_cast<i16>(
      div32_half(0x3cd0'13a9, high16_al2b1(static_cast<i32>(static_cast<u32>(den) << (static_cast<u32>(static_cast<i32>(nd)) & 0x1f)))));
  i16 e_common = wadd<i16>(wsub<i16>(wsub<i16>(nd, static_cast<i16>(common)), amp_exp), 4);
  i32 pv = wmul<i32>(wmul<i32>(sum, recip), 2);
  i16 nv = static_cast<i16>(norm_exp_l1_batch2(static_cast<u32>(pv)));
  v_exp = static_cast<i16>(-nv);
  i16 mv = high16_al2b1(static_cast<i32>(static_cast<u32>(pv) << (static_cast<u32>(static_cast<i32>(nv)) & 0x1f)));
  v_mant = mv;
  if (mv != 0) v_exp = wadd<i16>(wadd<i16>(v_exp, static_cast<i16>(common)), e_common);
  u32 pu = static_cast<u32>(wshl<i32>(
      high16_al2b1(wmul<i32>(wmul<i32>(high16_al2b1(wmul<i32>(wmul<i32>(cross, a), 2)), recip), 2)), 0xf));
  i16 nu = static_cast<i16>(norm_exp_l1_batch2(pu));
  u_exp = static_cast<i16>(-nu);
  i16 mu = high16_al2b1(static_cast<i32>(pu << (static_cast<u32>(static_cast<i32>(nu)) & 0x1f)));
  u_mant = mu;
  if (mu != 0) u_exp = wadd<i16>(wadd<i16>(wadd<i16>(wadd<i16>(u_exp, amp_exp), e_common), cross_exp), 2);
}

i32 norm_e(i32 x) { return norm_shift_batch1(static_cast<u32>(x)); }
u32 norm_u16(i32 x) { return x == 0 ? 0u : (static_cast<u32>(norm_e(x)) & 0xffff); }
i64 allshl_l1_batch3(i64 v, u8 count) { return allshl_l1_batch4(v, count); }

void copy_params_l1_batch3(S16 dst, CS16 src) {
  for (usz i = 0; i < 4; i++) dst[i] = src[i];
  dst[4] = src[4];
  dst[5] = src[5];
  dst[6] = src[6];
  dst[7] = src[7];
  copy16_batch3(dst.subspan(8), src.subspan(8), 0x38);
}
void copy_params_reset_phase(S16 dst, CS16 src) {
  dst[0x40] = 0;
  dst[0x41] = 0;
  dst[0x42] = src[0x42];
  copy_params_l1_batch3(dst, src);
}

i32 tone_seq_track(i16& sym, S16 state, i16 mode) {
  usz base = mode == 1 ? 0 : 0x14;
  std::copy(state.begin() + 2, state.begin() + 2 + 0x10, state.begin() + 1);  // copy_within(2..2+0x10, 1)
  state[0x11] = sym;
  i16 s = sym;
  if (s == 0xff) {
    state[0] = 0;
    return 0;
  }
  if (state[0] > 0 && s > 0x12 && s < 0x17) {
    u32 sh = static_cast<u32>(0x20 - static_cast<i32>(state[0])) & 0x1f;
    sym = static_cast<i16>(0x15 - static_cast<i16>((0x02b4'e4c3 >> sh) & 1));
    state[0] = static_cast<i16>(state[0] - 1);
    return 1;
  }
  i16 miss = 0x11;
  state[0] = 0;
  for (usz k = 0; k < 0x11; k++) {
    if (static_cast<u16>(state[1 + k]) == static_cast<u16>(dsp_tbl::TONE_SEQ[base + k])) miss = static_cast<i16>(miss - 1);
  }
  if (miss < 2) state[0] = 0x20;
  return 0;
}

i16 harmonic_count(i16 w0) {
  i16 l = static_cast<i16>(div32(0x3b39c, w0));
  if (static_cast<i32>(wadd<i16>(wmul<i16>(l, 2), 1)) * static_cast<i32>(w0) > 0x7ffff) {
    i16 q = static_cast<i16>(div32(0x80000, w0));
    l = static_cast<i16>(wsub<i32>(wmul<i32>(q, 0x10000), 1) >> 17);
  }
  if (l > 0x38) return 0x38;
  if (l < 9) return 9;
  return l;
}

void vec_add3_clamp(S16 a, CS16 b, u16 n, i16 k) {
  if (static_cast<i16>(n) <= 0) return;
  for (usz i = 0; i < n; i++) {
    i32 s = wadd<i32>(wadd<i32>((static_cast<i32>(a[i]) << 16) >> 2, (static_cast<i32>(b[i]) << 16) >> 2),
                      (static_cast<i32>(k) << 16) >> 2);
    i16 v = high16_al2b1(shl_sat(s, 2));
    a[i] = v >= 0x7800 ? static_cast<i16>(0x77ff) : (v < -0x7800 ? static_cast<i16>(-0x7800) : v);
  }
}

i32 weighted_slope(CS16 v, i16 n) {
  i32 acc = 0;
  i16 i = 0;
  if (n > 0) {
    while (i < n) {
      i = static_cast<i16>(i + 1);
      i16 w = high16_al2b1(wadd<i32>(wmul<i32>(wadd<i16>(n, 1), 0x10000), wmul<i32>(i, -0x20000)));
      acc = wadd<i32>(acc, wmul<i32>(wmul<i32>(w, v[static_cast<usz>(i - 1)]), 2));
    }
  }
  u32 na = norm_u16(acc);
  i32 sq = wmul<i32>(wmul<i32>(n, n), 2);
  u32 ns = norm_u16(sq);
  i16 q = static_cast<i16>(div32_half(static_cast<i32>(static_cast<u32>(acc) << (na & 0x1f)) >> 1,
                                      static_cast<i16>(static_cast<i32>(static_cast<u32>(sq) << (ns & 0x1f)) >> 16)));
  i32 qh = static_cast<i32>(q) << 16;
  i32 s = (static_cast<i32>(wsub<u32>(0x13u, na) & 0xffff) - 0x1f) + static_cast<i32>(ns);
  i16 sw = static_cast<i16>(s);
  if (sw >= 0) return static_cast<i32>(static_cast<u32>(qh) << (static_cast<u32>(static_cast<u8>(sw)) & 0x1f)) >> 16;
  if (sw < -0x1e) return qh >> 31;
  u32 sh = static_cast<u32>(static_cast<u8>(static_cast<u8>(0) - static_cast<u8>(sw)) & 0x1f);
  return (qh >> sh) >> 16;
}

void round_block_i32(S16 dst, i32 exp_dst, std::span<const i32> src, i32 exp_src, u16 n) {
  if (static_cast<i16>(n) <= 0) return;
  i16 sh = static_cast<i16>(wsub<i32>(exp_src, exp_dst));
  for (usz i = 0; i < n; i++) {
    i32 v = shl_sat(src[i], sh);
    i32 s = wadd<i32>(v, 0x8000);
    if (((s ^ v) & s) < 0) s = static_cast<i32>(wadd<u32>(0x7fff'ffffu, v < 0 ? 1u : 0u));
    dst[i] = high16_al2b1(s);
  }
}

i32 rd32_l1_batch3(CS16 s, usz w) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(s[w])) | (static_cast<u32>(static_cast<u16>(s[w + 1])) << 16));
}
void wr32_l1_batch3(S16 s, usz w, i32 v) {
  s[w] = static_cast<i16>(v);
  s[w + 1] = static_cast<i16>(v >> 16);
}

i32 dot_norm64(i16& out_exp, CS16 b, CS16 a, u16 n) {
  i64 acc = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 p = wmul<i32>(wmul<i32>(a[i], b[i]), 2);
      acc = wadd<i64>(acc, static_cast<i64>(p));
    }
  }
  i16 sh = static_cast<i16>(static_cast<u16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32))));
  i64 v = sh >= 0 ? allshl_l1_batch3(acc, static_cast<u8>(sh)) : allshr(acc, static_cast<u8>(static_cast<u32>(-static_cast<i32>(sh))));
  out_exp = wneg<i16>(sh);
  return static_cast<i32>(v);
}

void window_overlap(S16 out, i16& seed, S16 ring, CS16 win, u16 n, i16 total) {
  i16 cnt = static_cast<i16>(n);
  if (cnt > 0) {
    for (usz i = 0; i < n; i++) out[i] = high16_al2b1(wmul<i32>(wmul<i32>(ring[i], win[i]), 2));
  }
  i32 ni = cnt;
  i16 kept = 0;
  bool refill = true;
  if (0x54 - ni > 0) {
    i32 j = 0;
    for (;;) {
      kept = static_cast<i16>(kept + 1);
      ring[static_cast<usz>(j)] = ring[static_cast<usz>(j + ni)];
      j = kept;
      if (j >= 0x54 - ni) break;
    }
    if (kept > 0x53) refill = false;
  }
  if (refill) {
    usz p = static_cast<usz>(kept);
    u16 left = static_cast<u16>(wsub<i16>(0x54, kept));
    for (;;) {
      rand_step(seed);
      ring[p] = seed;
      p++;
      left = wsub<u16>(left, 1);
      if (left == 0) break;
    }
  }
  if (cnt > 0) {
    for (usz i = 0; i < n; i++) out[n + i] = high16_al2b1(wmul<i32>(wmul<i32>(ring[i], win[n - i]), 2));
  }
  i16 tail = static_cast<i16>(wsub<i32>(total, wmul<i32>(n, 2)));
  fill16(out.subspan(2 * static_cast<usz>(n)), 0, tail);
}

void pitch_advance(i32& out, i32& io, i16& dbl_sat, i16& dbl_wrap, i16 w, S16 st) {
  i32 io0 = io;
  i32 prev = rd32_l1_batch3(st, 0xa8);
  i32 v = wmul<i32>(prev, 2);
  if (high16_al2b1(v) > 0) {
    v = wsub<i32>(v, 0x10000);
    i32 neg = v == INT32_MIN ? INT32_MAX : -v;
    i32 u = neg >> 1;
    u32 nu = norm_u16(u);
    i32 wq = static_cast<i32>(w) << 16;
    u32 nw = norm_u16(wq);
    i16 ex = 0;
    i16 q = div_norm(static_cast<i32>(static_cast<u32>(u) << (nu & 0x1f)), static_cast<i16>(wsub<u32>(0x0fu, nu) & 0xffff),
                     high16_al2b1(static_cast<i32>(static_cast<u32>(wq) << (nw & 0x1f))),
                     static_cast<i16>(static_cast<u16>(wsub<u32>(wsub<u32>(0u, 4u), nw))), ex);
    st[0xab] = ex;
    st[0xaa] = q;
  }
  wr32_l1_batch3(st, 0xa8, v);
  out = v;
  dbl_wrap = high16_al2b1(static_cast<i32>(dbl_wrap) << 0x11);
  dbl_sat = high16_al2b1(shl_sat(static_cast<i32>(dbl_sat) << 16, 1));
  io = wadd<i32>(rd32_l1_batch3(st, 0xa8), wmul<i32>(wsub<i32>(io0, prev), 2));
}

i32 fft(S16 buf, i32 exp_in, i16 log2n) {
  i16 size = static_cast<i16>(1 << (static_cast<u8>(log2n) & 0x1f));
  i16 be = static_cast<i16>(cplx_block_exp(buf, size));
  u32 acc = 0;
  if (be < 2) acc = 0xffff'ffffu;
  if (be < 0) acc = wsub<u32>(acc, 1u);
  bit_reverse(buf, size);
  bfly_pairs(buf, size, static_cast<i16>(acc));
  acc &= 0xffff;
  i16 quarter = static_cast<i16>(size >> 2);
  i32 span = 2;
  i16 pass = 1;
  i16 stride = 0x80;
  if (log2n > 1) {
    for (;;) {
      be = wsub<i16>(be, 2);
      if ((pass & 1) == 0) be = static_cast<i16>(cplx_block_exp(buf, size));
      bool scaled = be <= 1;
      i32 half = static_cast<i16>(span);
      if (quarter > 0) {
        usz off = 0;
        for (u16 q = 0; q < static_cast<u16>(quarter); q++) {
          S16 lo = buf.subspan(off, static_cast<usz>(half * 2));
          S16 hi = buf.subspan(off + static_cast<usz>(half * 2));
          if (scaled) fft_butterfly_stage(lo, hi, stride, static_cast<i16>(span));
          else fft_stage(lo, hi, stride, static_cast<i16>(span));
          off += static_cast<usz>(half * 4);
        }
      }
      span = wmul<i32>(span, 2);
      pass = wadd<i16>(pass, 1);
      acc = wadd<u32>(acc, scaled ? 0xffffu : 0u);
      stride = static_cast<i16>(stride >> 1);
      quarter = static_cast<i16>(quarter >> 1);
      if (pass >= log2n) break;
    }
  }
  return wsub<i32>(exp_in, static_cast<i32>(acc));
}

i32 shift_round16_l1_batch4(i32 v, i16 sh) { return wadd<i32>(shift_i32_batch1(v, sh), 0x8000) >> 16; }
i16 high16_al2b3(i32 v) { return high16_batch3(v); }
i32 sat_add_l1_batch4(i32 a, i32 b) { return sat_add32(a, b); }

i16 db_to_gain(i16 db) {
  i32 v = db > 3 ? 3 : (db < -0x3c ? -0x3c : static_cast<i32>(db));
  return static_cast<i16>(wadd<i32>(wmul<i32>(v, 0x0154'2c00), 0x6bca'01ff) >> 16);
}

void init_level_tracker(S16 s) {
  s[0] = 2;
  s[6] = 0x4000;
  s[7] = 0;
  s[1] = 0x00ff;
  s[2] = 0x000e;
  s[0x40] = 0;
  s[0x41] = 0;
  s[4] = 0;
  s[5] = 0;
  fill16(s.subspan(8), 0x6666, 0x38);
  s[0x42] = -0x12;
}

i32 gain_for_rate(i16& out, u16 code, i16 db) {
  i16 cls;
  if (static_cast<u16>(code - 5) < 0x76) cls = 0;
  else if (static_cast<u16>(code - 0x80) < 0x10) cls = 1;
  else if (static_cast<u16>(code - 0x90) < 0x10) cls = 2;
  else cls = static_cast<i16>(3 + (static_cast<u16>(code - 0xa0) > 3 ? 1 : 0));
  if (cls != 0) {
    if (static_cast<i32>(cls) - 1 < 3) {
      i32 g = db_to_gain(db);
      out = static_cast<i16>(wsub<i32>(wmul<i32>(g, 0x10000), 0x07ff'0001) >> 16);
      return 1;
    }
    out = 0;
    return 0;
  }
  out = db_to_gain(db);
  return 1;
}

u16 read_field_l1_batch4(CSU16 bits, i32& pos, i16 n) {
  u16 v = pack_bits(bits.subspan(static_cast<usz>(pos)), n);
  pos = wadd<i32>(pos, n);
  return v;
}

void set_pitch_params(S16 cfg, i16 period) {
  i16 den = wadd<i16>(wadd<i16>(wmul<i16>(period, 4), 1), 0x9d);
  cfg[6] = static_cast<i16>(div32_half(0x0080'0000, den));
  i32 l32 = wmul<i32>(wadd<i16>(static_cast<i16>(period >> 2), 10), 0xece8);
  i16 l = static_cast<i16>(l32 >> 16);
  if (l > 0x38) {
    cfg[2] = 0x38;
    return;
  }
  if (l < 9) l = 9;
  cfg[2] = l;
}

u16 smooth_level_if_voiced(u16 state, i16 obs, u32 flags) {
  if (popcount_low(flags & 0x5555'5555u, 0x20) >= 8) {
    i32 a = wmul<i32>(obs, 0xcccc) >> 3;
    i32 b = wmul<i32>(static_cast<i16>(state), 0xe666);
    return static_cast<u16>((static_cast<u32>(wadd<i32>(a, b) >> 16)) & 0xffff);
  }
  return state;
}

i32 ratio_q15(i16 num, i16 den) {
  i32 dv = static_cast<i32>(den) << 16;
  u16 nd = dv == 0 ? static_cast<u16>(0) : static_cast<u16>(static_cast<u16>(norm_shift_batch1(static_cast<u32>(dv))) & 0xffff);
  i16 dm = static_cast<i16>(wshl<i32>(dv, static_cast<u32>(nd) & 0x1f) >> 16);
  i16 d_bias = static_cast<i16>(wsub<u32>(wsub<u32>(0u, 4u), nd));
  i32 nv = static_cast<i32>(num) << 16;
  u16 nn = nv == 0 ? static_cast<u16>(0) : static_cast<u16>(static_cast<u16>(norm_shift_batch1(static_cast<u32>(nv))) & 0xffff);
  i16 n_exp = static_cast<i16>(wsub<u32>(wsub<u32>(0u, 4u), nn));
  i16 e = 0;
  i16 q = div_norm(wshl<i32>(nv, static_cast<u32>(nn) & 0x1f), n_exp, dm, d_bias, e);
  return shift_round16_l1_batch4(static_cast<i32>(q) << 16, wsub<i16>(e, 0xf));
}

void scale_bias_vec(S16 v, i32 num, i16 n, i16 coef) {
  i32 den = (static_cast<i32>(n) << 25) >> 16;
  i32 q = div32_half(num, static_cast<i16>(den));
  i32 p = wmul<i32>(wmul<i32>(static_cast<i16>(q), coef), 2);
  i32 bias = (p == INT32_MIN ? INT32_MAX : wneg<i32>(p)) >> 1;
  if (n > 0) {
    usz cnt = static_cast<u16>(n);
    for (usz i = 0; i < cnt; i++) {
      i32 t = wadd<i32>(wmul<i32>(wmul<i32>(v[i], coef), 2) >> 1, bias);
      v[i] = high16_al2b3(shl_sat(t, 1));
    }
  }
}

i32 voicing_level(i16 mant, i16 exp, i16 thresh, CS16 cfg) {
  i16 floor_;
  if (cfg[4] == 0 && thresh <= 0x666) floor_ = static_cast<i16>(wsub<i32>(0x2c00'0000, wmul<i32>(thresh, 0x64000)) >> 16);
  else floor_ = 0x400;
  i32 l = mant < 1 ? static_cast<i32>(0xfff6'0000u) : log2_q(mant, wadd<i16>(exp, 0x1e));
  i32 h = high16_al2b3(shl_sat(l, 10));
  return wadd<i32>(wmul<i32>(h, 0xc000), static_cast<i32>(floor_) << 16) >> 16;
}

i64 allshl_l1_batch4(i64 v, u8 count) {
  u32 hi = static_cast<u32>(v >> 32);
  u32 lo = static_cast<u32>(v);
  if (count >= 0x40) return 0;
  if (count < 0x20) {
    u32 n = count & 0x1f;
    u32 new_hi = n == 0 ? hi : ((hi << n) | (lo >> (32 - n)));
    return static_cast<i64>((static_cast<u64>(new_hi) << 32) | static_cast<u64>(lo << n));
  }
  return static_cast<i64>(static_cast<u64>(lo << (count & 0x1f)) << 32);
}

void harmonic_dft(S16 buf, u16 n, i16 count, CS16 cos_tbl) {
  i16 local[64] = {};
  i16 ni = static_cast<i16>(n);
  copy16_batch3(S16(local), buf, ni);
  i32 recip = dsp_tbl::RECIP_N[static_cast<usz>(ni)];
  i16 cnt = ni <= count ? ni : count;
  if (cnt <= 0) return;
  i32 step_acc = 0;
  for (usz k = 0; k < static_cast<u16>(cnt); k++) {
    i64 acc = 0;
    i32 phase = wadd<i32>(step_acc >> 1, 0x40);
    i32 step = static_cast<i16>(static_cast<u16>(step_acc));
    if (ni > 0) {
      for (usz i = 0; i < n; i++) {
        i32 c = cos_tbl[static_cast<usz>((phase >> 7) & 0x1ff)];
        acc = wadd<i64>(acc, static_cast<i64>(wmul<i32>(wmul<i32>(local[i], c), 2)));
        phase = wadd<i32>(phase, step);
      }
    }
    i16 s = static_cast<i16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32)));
    i64 r = s < 0 ? allshr(acc, static_cast<u8>(static_cast<u32>(wneg<i32>(s)) & 0xff))
                  : allshl_l1_batch4(acc, static_cast<u8>(static_cast<u32>(static_cast<i32>(s)) & 0xff));
    i32 hi = static_cast<i16>(static_cast<i32>(r) >> 16);
    i32 t = shift_i32_batch1(wmul<i32>(wmul<i32>(hi, recip), 2), static_cast<i16>(wsub<u16>(0, static_cast<u16>(s))));
    buf[k] = high16_al2b3(wadd<i32>(t, 0x8000));
    step_acc = wadd<i32>(step_acc, recip);
  }
}

void correlate_and_shift(std::span<i32> acc, CS16 src, i16 exp_new, S16 hist, i16& hist_exp, CS16 win, i16 n) {
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n); i++) {
      i16 a = hist[i];
      i16 b = win[static_cast<usz>(n) - i];
      i32 p1 = (a == INT16_MIN && b == INT16_MIN) ? INT32_MAX : wmul<i32>(wmul<i32>(a, b), 2);
      i32 u2 = shl_sat(p1, wsub<i16>(hist_exp, 2));
      i16 c = src[i];
      i16 d = win[i];
      i32 p2 = (c == INT16_MIN && d == INT16_MIN) ? INT32_MAX : wmul<i32>(wmul<i32>(d, c), 2);
      i32 u3 = shl_sat(p2, wsub<i16>(exp_new, 2));
      i32 s = sat_add_l1_batch4(u2, u3);
      acc[i] = sat_add_l1_batch4(s, acc[i]);
    }
  }
  usz n_us = static_cast<u16>(n);
  copy16_batch3(hist, src.subspan(n_us), n);
  fill16(hist.subspan(n_us), 0, static_cast<i16>(0x54 - static_cast<i32>(n)));
  hist_exp = exp_new;
}

i32 top_bit_l1_batch5(u32 x) {
  u32 mag = static_cast<i32>(x) < 0 ? ~x : x;
  if (mag == 0) return 0;
  return 31 - __builtin_clz(mag);
}
u16 nrm(u32 x) { return x == 0 ? static_cast<u16>(0) : static_cast<u16>(norm_shift_l1_batch2(x)); }
i16 high16_l1_batch5(i32 x) { return high16_batch3(x); }
i32 up16_l1_batch5(i16 x) { return wshl<i32>(x, 16); }

i32 log2_at_q(i16 x, i16 exp, i32 q) {
  i32 l = log2_q(x, exp);
  i32 s = shl_sat(l, static_cast<i16>(0xf - q));
  return sat_add_l1_batch4(s, 0x8000) >> 16;
}

i16 triads_capped(i16 n) {
  i16 q = static_cast<i16>(div32(static_cast<i32>(n) + 2, 3));
  return q > 0xc ? static_cast<i16>(0xc) : q;
}

i32 mean_i16(CS16 v, u16 n) {
  i32 acc = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) acc = wadd<i32>(acc, v[i]);
  }
  return div32(acc, static_cast<i16>(n));
}

i32 log2_q11(i16 x) {
  i32 v = wshl<i32>(x, 15);
  u16 n = nrm(static_cast<u32>(v));
  i32 l = log2_q(high16_l1_batch5(wshl<i32>(v, static_cast<u32>(n) & 0x1f)), static_cast<i16>(5 - static_cast<i32>(n)));
  return wshl<i32>(l, 0xd) >> 0x11;
}

void emit_bits(usz& cursor, S16 dst, i16 n, i16 word) {
  usz at = cursor;
  i16 written = unpack_bits(word, dst.subspan(at), n);
  cursor = at + static_cast<usz>(written);
}

void weight_harmonics_by_freq_sq(S16 state, i16 fmax) {
  i16 lim = fmax > 0x3333 ? static_cast<i16>(0x3333) : fmax;
  i32 v = up16_l1_batch5(lim);
  u16 n = nrm(static_cast<u32>(v));
  i16 e = 0;
  i16 recip = div_norm(0x4000'0000, 1, high16_l1_batch5(wshl<i32>(v, static_cast<u32>(n) & 0x1f)),
                       static_cast<i16>(-4 - static_cast<i32>(n)), e);
  usz i = 0;
  i16 f = state[6];
  while (f < lim) {
    i32 p = wmul<i32>(wmul<i32>(recip, f), 2);
    i16 pn = p == 0 ? static_cast<i16>(0) : static_cast<i16>(0x1e - top_bit_l1_batch5(static_cast<u32>(p)));
    i32 m = high16_l1_batch5(wshl<i32>(p, static_cast<u32>(static_cast<i32>(pn)) & 0x1f));
    i16 sh = wmul<i16>(wsub<i16>(wsub<i16>(e, pn), 4), 2);
    i32 sq = wmul<i32>(wmul<i32>(m, m), 2);
    i16 g = high16_l1_batch5(shift_i32_batch1(sq, sh));
    state[8 + i] = high16_l1_batch5(wmul<i32>(wmul<i32>(g, state[8 + i]), 2));
    i++;
    f = high16_l1_batch5(sat_add_l1_batch4(up16_l1_batch5(state[6]), up16_l1_batch5(f)));
  }
}

void select_harmonic(S16 state, S16 flags, i16 idx) {
  usz old = static_cast<usz>(state[3]);
  i16 held = state[8 + old];
  state[8 + old] = 0;
  flags[old] = 0;
  state[3] = idx;
  state[8 + static_cast<usz>(idx)] = held;
  flags[static_cast<usz>(state[3])] = 1;
  i32 num = up16_l1_batch5(state[1]);
  u16 n1 = nrm(static_cast<u32>(num));
  i32 den = up16_l1_batch5(wadd<i16>(state[3], 1));
  u16 n2 = nrm(static_cast<u32>(den));
  i16 e = 0;
  i16 m = div_norm(wshl<i32>(num, static_cast<u32>(n1) & 0x1f), static_cast<i16>(0xf - static_cast<i32>(n1)),
                   high16_l1_batch5(wshl<i32>(den, static_cast<u32>(n2) & 0x1f)), static_cast<i16>(0x17 - static_cast<i32>(n2)), e);
  e = wadd<i16>(e, 4);
  state[6] = high16_l1_batch5(shift_i32_batch1(up16_l1_batch5(m), e));
  if (state[2] <= state[3]) state[2] = wadd<i16>(state[3], 1);
  if (state[2] > 0x38) state[2] = 0x38;
  else if (state[2] < 9) state[2] = 9;
}

void build_spectrum_history(S16 state, CS16 src, i16 n) {
  state[0x13] = 0;
  copy16_batch3(state.subspan(0x14), src, n);
  i32 acc = up16_l1_batch5(src[static_cast<usz>(n - 1)]);
  i16 first = wadd<i16>(n, 1);
  if (first < 0x41) {
    usz count = static_cast<usz>(wsub<u32>(0x41u, static_cast<u32>(static_cast<i32>(n)) + 1) & 0xffff);
    usz at = 0x13 + static_cast<usz>(first);
    for (usz k = 0; k < count; k++) {
      acc = sat_add_l1_batch4(acc, static_cast<i32>(0xfa3b'0000u));
      state[at] = high16_l1_batch5(acc);
      at++;
    }
  }
  i32 end = static_cast<i32>(n) + 0x14;
  if (0x41 < end) {
    usz read = 0x52;
    i32 i = 0x41;
    i16 s = 0x41;
    for (;;) {
      i16 v = state[read];
      read -= 1;
      s = wadd<i16>(s, 1);
      state[0x13 + static_cast<usz>(i)] = v;
      i = s;
      if (i >= end) break;
    }
  }
  for (usz j = 0; j < 0x13; j++) state[0x12 - j] = state[0x14 + j];
}

void resample_linear(S16 out, i16 n, CS16 src, i16 m) {
  i32 vn = up16_l1_batch5(n);
  u16 nn = vn != 0 ? nrm(static_cast<u32>(vn)) : static_cast<u16>(0);
  i32 vm = up16_l1_batch5(m);
  u16 nm = nrm(static_cast<u32>(vm));
  i16 e = 0;
  i16 q = div_norm(wshl<i32>(vm, static_cast<u32>(nm) & 0x1f), wneg<i16>(static_cast<i16>(nm)),
                   high16_l1_batch5(wshl<i32>(vn, static_cast<u32>(nn) & 0x1f)), wneg<i16>(static_cast<i16>(nn)), e);
  i32 step = shift_i32_batch1(up16_l1_batch5(q), wsub<i16>(e, 0xf));
  out[0] = src[0];
  if (n > 1) {
    i32 acc = step;
    for (usz i = 1; i < static_cast<usz>(n); i++) {
      i32 idx = high16_l1_batch5(acc);
      i32 f = static_cast<i16>(static_cast<u16>(static_cast<u16>(acc >> 1) & 0x7fff));
      i32 a = src[static_cast<usz>(idx)];
      i32 b = src[static_cast<usz>(idx + 1)];
      out[i] = high16_l1_batch5(wadd<i32>(wmul<i32>(wadd<i32>(wmul<i32>(b, f), wmul<i32>(a, 0x8000)), 2),
                                          wmul<i32>(wmul<i32>(a, f), -2)));
      acc = wadd<i32>(acc, step);
    }
  }
  out[static_cast<usz>(n)] = src[static_cast<usz>(m)];
}

void frame_bit_permute(S16 buf, CS16 cfg, i16 store) {
  i16 img[192] = {};
  bool st = store != 0;
  if (!st) copy16_batch3(S16(img), buf, cfg[2]);
  i16 b0 = cfg[4], b1 = cfg[5], b2 = cfg[6], b3 = cfg[7];
  i16 b4 = cfg[8], b5 = cfg[9], b6 = cfg[10], b7 = cfg[11], b8 = cfg[12];
  i16 p1 = b1;
  i16 p2 = wadd<i16>(p1, b0);
  i16 p3 = wadd<i16>(p2, b2);
  i16 p4 = wadd<i16>(p3, b3);
  i16 p5 = wadd<i16>(p4, b4);
  i16 p6 = wadd<i16>(p5, b5);
  i16 p7 = wadd<i16>(p6, b6);
  i16 q = wadd<i16>(wadd<i16>(wadd<i16>(b3, b2), b0), b1);
  i16 q1 = wadd<i16>(q, b4);
  i16 q2 = wadd<i16>(q1, b5);
  i16 q3 = wadd<i16>(q2, b6);
  const std::pair<i16, i16> sched[18] = {
      {0, 4},
      {p1, 4},
      {p2, 4},
      {p3, 8},
      {p4, 4},
      {p5, 4},
      {p6, 3},
      {p7, 3},
      {wadd<i16>(p7, b7), 1},
      {wadd<i16>(b1, 4), static_cast<i16>(b0 - 4)},
      {wadd<i16>(wadd<i16>(b0, b1), 4), static_cast<i16>(b2 - 4)},
      {4, static_cast<i16>(b1 - 4)},
      {wadd<i16>(q, 4), static_cast<i16>(b4 - 4)},
      {wadd<i16>(wsub<i16>(q, b3), 8), static_cast<i16>(wsub<u16>(static_cast<u16>(b3), 8))},
      {wadd<i16>(q1, 4), static_cast<i16>(b5 - 4)},
      {wadd<i16>(q2, 3), static_cast<i16>(b6 - 3)},
      {wadd<i16>(q3, 3), static_cast<i16>(b7 - 3)},
      {wadd<i16>(wadd<i16>(b7, q3), 1), static_cast<i16>(b8 - 1)},
  };
  usz cur = 0;
  for (const auto& [off, n] : sched) {
    i16 moved = copy16_batch4(buf.subspan(cur), S16(img).subspan(static_cast<usz>(off)), n, st);
    cur += static_cast<usz>(moved);
  }
  if (st) copy16_batch3(buf, CS16(img), cfg[2]);
}

void solve_quad_ramp(S16 out, S16 exps, i32 x0, i16 a, i16 k, i16 c, u16 count) {
  if (static_cast<i16>(count) <= 0) return;
  i32 a2 = wmul<i32>(wmul<i32>(a, a), 2);
  i32 c32 = wmul<i32>(c, 0x10000);
  i32 ac = wmul<i32>(wmul<i32>(a, c), 2);
  i32 c2 = wmul<i32>(wmul<i32>(c, c), 2);
  i16 i = 0;
  for (usz slot = 0; slot < count; slot++) {
    i = wadd<i16>(i, 1);
    i32 t = wsub<i32>(wmul<i32>(i, 0x10000), x0);
    u16 n1 = nrm(static_cast<u32>(t));
    i32 u = wmul<i32>(wmul<i32>(high16_l1_batch5(wshl<i32>(t, n1 & 0x1f)), c), 2);
    u16 n2 = nrm(static_cast<u32>(u));
    i32 e_m = wadd<i32>(static_cast<i32>(wsub<u32>(0xfu, n1) & 0xffff), wsub<i32>(0x10, n2));
    u16 n3 = nrm(static_cast<u32>(a2));
    u16 n4 = nrm(static_cast<u32>(c2));
    i32 prod = wmul<i32>(wmul<i32>(high16_l1_batch5(wshl<i32>(c2, n4 & 0x1f)), high16_l1_batch5(wshl<i32>(a2, n3 & 0x1f))), 2);
    u16 n5 = nrm(static_cast<u32>(prod));
    i32 m = high16_l1_batch5(wshl<i32>(u, n2 & 0x1f));
    i16 e = static_cast<i16>(wsub<u32>(wadd<u32>(wsub<u32>(wneg<u32>(n3), 8u) & 0xffff, wsub<u32>(0x1eu, n4)), n5));
    i32 kw = wmul<i32>(wmul<i32>(k, m), 2);
    u16 n6 = nrm(static_cast<u32>(kw));
    i16 s = add_norm(high16_l1_batch5(wshl<i32>(prod, n5 & 0x1f)), e, high16_l1_batch5(wshl<i32>(kw, n6 & 0x1f)),
                     static_cast<i16>(wsub<i32>(wsub<i32>(e_m, n6), 4)), e);
    i32 r = sqrt_norm(up16_l1_batch5(s), e);
    u16 n7 = r != 0 ? nrm(static_cast<u32>(r)) : static_cast<u16>(0);
    e = wsub<i16>(e, static_cast<i16>(n7));
    u16 n8 = nrm(static_cast<u32>(ac));
    i16 g = add_norm(high16_l1_batch5(wshl<i32>(ac, n8 & 0x1f)), static_cast<i16>(wsub<u32>(0xbu, n8)),
                     high16_l1_batch5(wshl<i32>(r, n7 & 0x1f)), e, e);
    i16 v = div_norm(wmul<i32>(m, 0x10000), static_cast<i16>(e_m), g, e, e);
    i32 scaled = shift_i32_batch1(up16_l1_batch5(v), wsub<i16>(e, 0xf));
    if (wadd<i32>(scaled, wmul<i32>(c, -0x10000)) >= 0) {
      u16 n9 = nrm(static_cast<u32>(c32));
      e = static_cast<i16>(wsub<u32>(0xfu, n9) & 0xffff);
      v = high16_l1_batch5(wshl<i32>(c32, n9 & 0x1f));
    }
    out[slot] = v;
    exps[slot] = e;
  }
}

}  // namespace b25::eng
