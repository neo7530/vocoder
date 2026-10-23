// Port of blip25-vocoder src/engine/decode.rs (MIT, (c) Chance Lindsey).
// Decoder-side frame parsing, parameter reconstruction and synthesis. The
// encoder shares several of these helpers (field packing, coefficient
// quantisers), so the encode path of those functions is ported here as well.
#include <algorithm>
#include <cassert>

#include "b25/decode.h"
#include "b25/decode_tables.h"
#include "b25/dsp.h"
#include "b25/dsp_tables.h"
#include "b25/engine_tables.h"

namespace b25::eng {

using namespace rs;

i32 shift_i32_al2b1(i32 v, i16 sh) {
  if (sh >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f));
  if (sh < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(static_cast<i32>(wneg<i16>(sh))) & 0x1f);
}

i32 init_bit_alloc_and_agc(i16 mode, A13 alloc, S16 agc) {
  init_agc_config(agc);
  if (mode == 0) imbe_bit_alloc(alloc);
  else ambe_bit_alloc(alloc, 0);
  return 0;
}

i32 rfft_forward(S16 buf, i32 exp_in, i16 log2n, i16 stage_flag) {
  i32 c = static_cast<u16>(fft(buf, exp_in, wsub<i16>(log2n, 1)));
  i32 r = static_cast<u16>(fft_merge_stage(buf, c, log2n, 0, stage_flag));
  if (stage_flag == 0) buf[1] = 0;
  return r;
}

void expand_log_mags(S16 v, i16& exp, u16 n) {
  i16 cnt = static_cast<i16>(n);
  if (cnt > 0) {
    for (usz i = 0; i < n; i++) {
      i16 x = v[i];
      v[i] = x >= 0x7800 ? static_cast<i16>(0x77ff) : (x < -0x77ff ? static_cast<i16>(-0x77ff) : x);
    }
  }
  i32 max = INT32_MIN;
  if (cnt > 0) {
    for (usz i = 0; i < n; i++) {
      i32 p = static_cast<i32>(v[i]) << 16;
      if (max < p) max = p;
    }
  }
  i32 scaled = wadd<i32>(max >> 0xb, 0x10000);
  i16 out_exp = high16_al2b1(scaled);
  if (cnt > 0) {
    for (usz i = 0; i < n; i++) v[i] = static_cast<i16>(pow2_scaled(v[i], 4, out_exp));
  }
  exp = wsub<i16>(out_exp, 0xf);
  fill16(v.subspan(static_cast<usz>(cnt)), 0, wsub<i16>(0x38, cnt));
}

void code_higher_order_coeffs(S16 bits, i32& pos, const u16 (&nbits)[5], S16 params, bool encode) {
  for (usz i = 0; i < 5; i++) {
    u16 b = nbits[i];
    i32 prod = wmul<i32>(wmul<i32>(dec_tbl::HOC_STEP[b], dec_tbl::HOC_SCALE[i]), 2);
    u16 levels = (b & 0x1f) >= 16 ? static_cast<u16>(0) : static_cast<u16>(1u << (b & 0x1f));
    i16 step = static_cast<i16>(prod >> 0x11);
    u16 idx;
    if (!encode) {
      idx = read_field_i16(bits, pos, static_cast<i16>(b));
    } else {
      i32 acc = wadd<i32>(static_cast<i32>(params[1 + i]) << 16, (static_cast<i32>(step) << 16) >> 1);
      u16 v;
      if (acc < 0) {
        v = 0;
      } else if (high16_al2b1(acc) < step) {
        i32 a = acc;
        if (static_cast<i16>(b) > 0) {
          for (u16 k = 0; k < b; k++) subc(step, a);
        }
        v = static_cast<u16>(a);
      } else {
        v = wsub<u16>(levels, 1);
      }
      emit_bits_i16(bits, pos, static_cast<i16>(b), static_cast<i16>(v));
      idx = v;
    }
    i16 centre = high16_al2b1(wadd<i32>(wsub<i32>(static_cast<i32>(static_cast<i16>(idx)) << 17,
                                                  static_cast<i32>(static_cast<i16>(levels)) << 16),
                                        0x10000));
    i32 recon = wmul<i32>(wmul<i32>(centre, step), 2);
    params[1 + i] = high16_al2b1(shift_i32_al2b1(recon, static_cast<i16>(0xe - static_cast<i32>(b))));
  }
}

u16 read_field_i16(CS16 bits, i32& pos, i16 n) {
  u16 r = pack_bits_i16(bits.subspan(static_cast<usz>(pos)), n);
  pos = wadd<i32>(pos, n);
  return r;
}

void emit_bits_i16(S16 bits, i32& pos, i16 n, i16 word) {
  usz c = static_cast<usz>(pos);
  emit_bits(c, bits, n, word);
  pos = static_cast<i32>(c);
}

u32 rd32_al2b1(CS16 b, usz w) {
  return static_cast<u32>(static_cast<u16>(b[w])) | (static_cast<u32>(static_cast<u16>(b[w + 1])) << 16);
}
void wr32_al2b1(S16 b, usz w, u32 v) {
  b[w] = static_cast<i16>(static_cast<u16>(v));
  b[w + 1] = static_cast<i16>(static_cast<u16>(v >> 16));
}

void voicing_decision(S16 cur, S16 prev, CS16 st, u16& mask_io, i16 thresh, i16 frames, CS16 cfg, i16 mant, i16 exp) {
  if (thresh <= 0x28f && frames < 4) return;
  u32 cur_pat = rd32_al2b1(cur, 4);
  if ((cur_pat & 0xaaaa'aaaau) != 0) return;
  if ((rd32_al2b1(prev, 4) & 0xaaaa'aaaau) != 0) return;
  i16 w0 = cur[6];
  if (!(st[0x44] > 0 || cur_pat != 0 || w0 > 0x40b5 || w0 < 0x3d63)) return;
  i16 l = cur[2];
  i16 slope = static_cast<i16>(weighted_slope(cur.subspan(8), l));
  i16 level = static_cast<i16>(voicing_level(mant, exp, thresh, cfg));
  i32 floor_ = (static_cast<i32>(static_cast<i16>(log2_q11(w0))) << 16) >> 2;
  i16 triads = triads_capped(l);
  u32 bit = 1;
  u32 mask = mask_io;
  u16 k = static_cast<u16>(wsub<i16>(triads, 1));
  if (static_cast<i16>(k) >= 0) {
    i32 lo = wmul<i32>(static_cast<i16>(k), 3);
    u32 top = static_cast<u16>(wmul<u16>(k, 3));
    u32 hi = 0x38;
    for (;;) {
      u32 prev_top = top;
      i16 peak = -0x5000;
      u32 last = wsub<u32>(hi, 1);
      u16 c = static_cast<u16>(last);
      while (static_cast<i32>(static_cast<i16>(c)) >= lo) {
        i16 v = cur[8 + static_cast<usz>(static_cast<i32>(static_cast<i16>(c)))];
        if (peak < v) peak = v;
        c = wsub<u16>(c, 1);
      }
      i16 s = static_cast<i16>(wsub<i32>(static_cast<i32>(peak) << 16, floor_) >> 16);
      if ((static_cast<i32>(static_cast<i16>(static_cast<u16>(hi))) * 2) > static_cast<i32>(l) && slope < 0) {
        s = static_cast<i16>(wadd<i32>((static_cast<i32>(slope) << 16) >> 2, static_cast<i32>(s) << 16) >> 16);
      }
      if (s > level && (static_cast<u16>(bit) & static_cast<u16>(mask)) == 0) {
        mask |= bit;
        u16 j = static_cast<u16>(last);
        while (static_cast<i32>(static_cast<i16>(j)) >= lo) {
          usz idx = 8 + static_cast<usz>(static_cast<i32>(static_cast<i16>(j)));
          i32 a = static_cast<i32>(cur[idx]) << 16;
          i32 d = wsub<i32>(a, floor_);
          i32 r = ((d ^ a) & (~floor_ ^ d)) < 0 ? wadd<i32>(0x7fff'ffff, a < 0 ? 1 : 0) : d;
          i16 v = high16_al2b1(r);
          cur[idx] = v > 0x77ff ? static_cast<i16>(0x77ff) : (v < -0x77ff ? static_cast<i16>(-0x77ff) : v);
          j = wsub<u16>(j, 1);
        }
      }
      lo = wsub<i32>(lo, 3);
      bit = wshl<u32>(bit, 1);
      k = wsub<u16>(k, 1);
      hi = static_cast<u16>(prev_top);
      top = wsub<u32>(prev_top, 3);
      if (static_cast<i16>(k) < 0) break;
    }
  }
  mask_io = static_cast<u16>(mask);
  i16 nbands = triads;
  if (nbands > 8) {
    mask = static_cast<u16>(map_voicing_bits(static_cast<i16>(mask), nbands, l));
    nbands = 8;
  }
  u16 prev_dibits = static_cast<u16>(gather_dibits(static_cast<i32>(rd32_al2b1(st, 4))));
  u32 pat = voicing_pattern(static_cast<i16>(wadd<u16>(static_cast<u16>(mask), 0x38)), prev_dibits, nbands);
  u32 o0 = rd32_al2b1(prev, 4);
  u32 o1 = rd32_al2b1(cur, 4);
  expand_dibits(o0, o1, pat);
  wr32_al2b1(prev, 4, o0);
  wr32_al2b1(cur, 4, o1);
}

i32 copy_seg(S16 chan, i32 o, S16 scratch, i32 s, i16 n, bool store) {
  copy16_batch4(chan.subspan(static_cast<usz>(o)), scratch.subspan(static_cast<usz>(s)), n, store);
  return n > 0 ? wadd<i32>(o, n) : o;
}

void interleave_imbe_frame(S16 chan, i16 nharm, CS16 hbits, CS16 alloc, bool store) {
  i16 scratch_a[192] = {};
  S16 scratch(scratch_a);
  i16 total = alloc[2];
  i16 f5 = alloc[5];
  i16 f6 = alloc[6];
  if (!store) copy16_batch3(scratch, chan, total);
  i32 o = copy_seg(chan, 0, scratch, 0, 6, store);
  i32 tail = wsub<i16>(wsub<i16>(total, f6), 1);
  o = copy_seg(chan, o, scratch, tail, 3, store);
  i16 limit = wsub<i16>(nharm, 1);
  i16 off = 0;
  if (limit > 0) {
    for (usz i = 0; i < static_cast<u16>(limit); i++) {
      if (off < hbits[i]) off = hbits[i];
    }
  }
  i16 triads = triads_capped(nharm);
  i16 cnt = 1;
  isz base = static_cast<isz>(wadd<i32>(triads, f5));
  isz dpos = base;
  isz spos = 0;
  if (o < 0x30) {
    usz op = static_cast<usz>(o);
    off = interleave_walk(chan, op, 0x30, scratch, dpos, hbits, spos, cnt, base, off, limit, store);
    o = static_cast<i32>(op);
  }
  o = copy_seg(chan, o, scratch, f5, triads, store);
  o = copy_seg(chan, o, scratch, wadd<i32>(tail, 3), 2, store);
  i32 end2 = wsub<i32>(total, 4);
  if (o < end2) {
    usz op = static_cast<usz>(o);
    interleave_walk(chan, op, static_cast<usz>(end2), scratch, dpos, hbits, spos, cnt, base, off, limit, store);
    o = static_cast<i32>(op);
  }
  o = copy_seg(chan, o, scratch, wadd<i32>(tail, 5), 1, store);
  o = copy_seg(chan, o, scratch, 6, 2, store);
  copy_seg(chan, o, scratch, wsub<i32>(total, 1), 1, store);
  if (store) copy16_batch3(chan, scratch, total);
}

// ======================= from batch2 =======================

i32 shift32_wide(i32 v, i32 s) {
  if (s < 0) {
    if (s < -0x1e) return v >> 31;
    return v >> (static_cast<u32>(-s) & 0x1f);
  }
  return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(s) & 0x1f));
}

u16 shl16_al2b2(u16 v, u32 count) {
  u32 c = count & 0x1f;
  return c >= 16 ? static_cast<u16>(0) : static_cast<u16>(v << c);
}

i32 norm_bsr(i32 x) {
  if (x == 0) return 0;
  u32 mag = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  i32 b = mag == 0 ? 0 : 31 - __builtin_clz(mag);
  return static_cast<u16>(0x1e - b);
}

i32 rd32_al2b2(CS16 v, usz word) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(v[word])) | (static_cast<u32>(static_cast<i32>(v[word + 1])) << 16));
}

void tone_frame_fixup(S16 p, CS16 cos_tbl) {
  i16 code = p[1];
  i32 cls = frame_size_class(code);
  i16 ia = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, code, 0, cos_tbl)));
  i16 ib = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, code, 1, cos_tbl)));
  i16 n = p[2];
  for (i16 i = 0; i < n; i = static_cast<i16>(i + 1)) {
    if (i != ia && i != ib) {
      i32 x = p[8 + static_cast<usz>(i)];
      p[8 + static_cast<usz>(i)] = static_cast<i16>(wmul<i32>(x, 0xa3c) >> 16);
    }
  }
  if (p[1] >= 0x80 && p[1] < 0xa0) {
    i32 a = p[8 + static_cast<usz>(ia)];
    i32 b = p[8 + static_cast<usz>(ib)];
    i32 m = wadd<i32>(wmul<i32>(b, b), wmul<i32>(a, a));
    i16 v = static_cast<i16>(sqrt_to_exp(m, 8, 4));
    p[8 + static_cast<usz>(ib)] = v;
    p[8 + static_cast<usz>(ia)] = v;
  }
  code = p[1];
  if (code >= 0x14 && code <= 0x16) {
    i16 den = wmul<i16>(wadd<i16>(p[3], 1), 0x100);
    p[6] = static_cast<i16>(div32_half(static_cast<i32>(code) << 0x14, den));
  }
}

void detect_tone_frame(S16 p, i16 prev_code, CS16 hist, CS16 cos_tbl, CSU16 codewords) {
  p[1] = 0xff;
  u16 n = static_cast<u16>(p[2]);
  i16 lin[56] = {};
  i16 blk_exp = 0;
  pow2_vector(S16(lin), blk_exp, p.subspan(8), n);
  i16 mean = 0;
  i32 sum_exp = static_cast<u16>(sum_norm(mean, CS16(lin), blk_exp, n));
  i32 blk = blk_exp;
  i16 peak = lin[0];
  i16 peak_i = 0;
  for (i16 i = 1; i < static_cast<i16>(n); i = static_cast<i16>(i + 1)) {
    if (peak < lin[i]) {
      peak = lin[i];
      peak_i = i;
    }
  }
  if (voiced_dibit(p, peak_i) == 1 && peak_i <= 10) {
    i16 thresh;
    if (peak_i <= 0) {
      thresh = 0x7fdf;
    } else {
      thresh = 0x7eb8;
      if (peak_i <= 1 && peak_i == static_cast<i16>(band_count_for_mode(0, prev_code, 0, cos_tbl))) thresh = 0x7f5c;
    }
    i32 scaled = shift32_al2b2(static_cast<i32>(peak) << 16, static_cast<i16>(wsub<i32>(blk, sum_exp)));
    i32 cmp = wsub<i32>(wmul<i32>(wmul<i32>(thresh, mean), 2), scaled);
    if (cmp < 0) {
      i16 forced = 0;
      i32 lo, hi;
      bool matched = false;
      if (peak_i == 1 && wsub<u16>(static_cast<u16>(prev_code), 0x14) < 3) {
        i16 d = hist[2];
        for (i16 k = 0; k < 2; k = static_cast<i16>(k + 1)) {
          if (hist[k] != d) {
            matched = true;
            break;
          }
        }
      }
      if (matched) {
        forced = 1;
        lo = 0x14;
        hi = 0x16;
      } else {
        i32 c = wmul<i32>(peak_i, 0xccc00);
        lo = static_cast<u16>(wadd<i32>(c >> 16, 1));
        hi = static_cast<u16>(wadd<i32>(c, 0xccc00) >> 16);
      }
      i16 dist = 0x7fff;
      i16 code = static_cast<i16>(lo);
      if (static_cast<u16>(prev_code) < 0x80) {
        nearest_codeword(static_cast<u16>(hist[2]), dist, code, prev_code, prev_code, codewords);
      }
      nearest_codeword(static_cast<u16>(hist[2]), dist, code, static_cast<i16>(lo), static_cast<i16>(hi), codewords);
      bool accept;
      if (dist <= 1 || (dist <= 2 && code == prev_code)) accept = true;
      else if (dist > 0x10) return;
      else accept = forced != 0;
      if (accept) {
        p[1] = code;
        p[0] = 3;
        p[3] = static_cast<i16>(static_cast<u16>(band_count_for_mode(0, code, 0, cos_tbl)));
      }
      return;
    }
  }
  i16 dist = 0x7fff;
  i16 code = 0x80;
  if (prev_code >= 0x80 && prev_code <= 0xa3) {
    nearest_codeword(static_cast<u16>(hist[2]), dist, code, prev_code, prev_code, codewords);
  }
  nearest_codeword(static_cast<u16>(hist[2]), dist, code, 0x80, 0xa3, codewords);
  if (dist > 1 && (dist > 2 || code != prev_code)) return;
  i16 leg_a = static_cast<i16>(static_cast<u16>(band_count_for_mode(1, code, 0, cos_tbl)));
  i32 e_a = voiced_dibit(p, leg_a) == 1 ? ((static_cast<i32>(lin[static_cast<usz>(leg_a)]) << 16) >> 1) : 0;
  i16 leg_b = static_cast<i16>(static_cast<u16>(band_count_for_mode(1, code, 1, cos_tbl)));
  i32 e_b = voiced_dibit(p, leg_b) == 1 ? ((static_cast<i32>(lin[static_cast<usz>(leg_b)]) << 16) >> 1) : 0;
  i32 energy = wadd<i32>(e_a, e_b);
  i32 norm = norm_bsr(energy);
  i32 e_exp = static_cast<u16>(wadd<i32>(wsub<i32>(blk, norm), 1));
  i16 mant = static_cast<i16>(static_cast<i32>(static_cast<u32>(energy) << (static_cast<u32>(norm) & 0x1f)) >> 16);
  i16 sel = code < 0x90 ? static_cast<i16>(0x3200) : (code < 0xa0 ? static_cast<i16>(0x3600) : static_cast<i16>(0x7800));
  i32 lim = (static_cast<i32>(sel) << 16) >> 10;
  i32 cap = wmul<i32>(p[2], 0xe666);
  if (lim > cap) lim = cap;
  i16 floor_ = high16_al2b1(wshl<i32>(lim, 10));
  bool below;
  if (wadd<i32>(static_cast<i16>(e_exp), 0xf) > wadd<i32>(static_cast<i16>(sum_exp), 5)) {
    i32 d = wmul<i32>(wmul<i32>(floor_, mean), 2);
    i16 cnt = static_cast<i16>(wsub<i32>(wsub<i32>(sum_exp, e_exp), 10));
    i32 a = wmul<i32>(wmul<i32>(mant, p[2]), 2);
    below = wsub<i32>(a, shift32_al2b2(d, cnt)) < 0;
  } else {
    i32 a = wmul<i32>(wmul<i32>(mant, p[2]), 2);
    i16 cnt = static_cast<i16>(wadd<i32>(wsub<i32>(e_exp, sum_exp), 10));
    i32 d = wmul<i32>(wmul<i32>(floor_, mean), 2);
    below = wsub<i32>(shift32_al2b2(a, cnt), d) < 0;
  }
  if (!below) {
    p[1] = code;
    p[0] = 3;
    p[3] = static_cast<i16>((static_cast<i32>(leg_a) << 8) | static_cast<i32>(static_cast<u16>(leg_b)));
  }
}

u8 voiced_dibit(CS16 p, i16 k) {
  i32 word = rd32_al2b2(p, 4);
  i32 t = wadd<i32>(wmul<i32>(wmul<i32>(static_cast<i16>(wadd<i32>(k, 1)), p[6]), 4), 0x8000) >> 0x11;
  t = static_cast<i16>(t);
  return static_cast<u8>(static_cast<u8>(shift32_wide(word, wmul<i32>(t, -4))) & 3);
}

void quantize_coeff_blocks(S16 payload, i32& pos, i16 blocks, u16 per_block, S16 vals, CSU16 allocs, i16 encode,
                           CS16 weight, CS16 loading) {
  i16 left = wsub<i16>(blocks, 1);
  usz vi = 0, ai = 0;
  while (left >= 0) {
    left = wsub<i16>(left, 1);
    vi += 1;
    if (static_cast<i16>(per_block) > 0) {
      usz j = 0;
      u16 rem = per_block;
      for (;;) {
        u16 bits = allocs[ai];
        ai += 1;
        if (static_cast<i16>(bits) < 1) {
          vals[vi] = 0;
        } else {
          i32 prod = wmul<i32>(wmul<i32>(loading[bits], weight[j]), 2);
          u16 cells = shl16_al2b2(1, bits);
          i16 step = static_cast<i16>(prod >> 0x11);
          u16 idx;
          if (encode == 0) {
            idx = read_field_al2b2(payload, pos, static_cast<i16>(bits));
          } else {
            i32 acc = wadd<i32>((static_cast<i32>(step) << 16) >> 1, wmul<i32>(vals[vi], 0x10000));
            u16 v;
            if (acc < 0) {
              v = 0;
            } else if (high16_al2b1(acc) < step) {
              i32 a = acc;
              for (u16 b = 0; b < bits; b++) subc(step, a);
              v = static_cast<u16>(a);
            } else {
              v = wsub<u16>(cells, 1);
            }
            usz cur = static_cast<usz>(pos);
            emit_bits(cur, payload, static_cast<i16>(bits), static_cast<i16>(v));
            pos = static_cast<i32>(cur);
            idx = v;
          }
          i32 centre = wadd<i32>(wadd<i32>(wmul<i32>(static_cast<i16>(idx), 0x20000), wmul<i32>(static_cast<i16>(cells), -0x10000)), 0x10000);
          i32 acc = wmul<i32>(wmul<i32>(high16_al2b1(centre), step), 2);
          vals[vi] = high16_al2b1(shift32_al2b2(acc, static_cast<i16>(wsub<u16>(0x0e, bits))));
        }
        j += 1;
        vi += 1;
        rem = static_cast<u16>(rem - 1);
        if (rem == 0) break;
      }
    }
  }
}

u16 read_field_al2b2(CS16 payload, i32& pos, i16 n) {
  usz at = static_cast<usz>(pos);
  u16 v = pack_bits_i16(payload.subspan(at), n);
  pos = wadd<i32>(pos, n);
  return v;
}

i32 split_voiced_freq_code(i16& code, S16 a, S16 b, i16 freq_a, i16 freq_b, u32 bits, i32 enable) {
  i32 map_a = rd32_al2b2(a, 4);
  i32 map_b = rd32_al2b2(b, 4);
  bool a_unvoiced = (static_cast<u32>(map_a) & 0x5555'5555u) != 0;
  bool b_unvoiced = (static_cast<u32>(map_b) & 0x5555'5555u) != 0;
  bool a_voiced = (static_cast<u32>(map_a) & 0xaaaa'aaaau) != 0;
  bool b_voiced = (static_cast<u32>(map_b) & 0xaaaa'aaaau) != 0;
  bool on = static_cast<i16>(enable) != 0;
  bool ok = on ? b[0] == 1 : true;
  if (a_unvoiced || b_unvoiced) {
    b[7] = 0;
    a[7] = 0;
    return 0;
  }
  if (!ok || (!a_voiced && !b_voiced)) {
    b[7] = 0;
    a[7] = 0;
    return 0;
  }
  i16 cap = on ? wsub<i16>(static_cast<i16>(shl16_al2b2(0x3c, wsub<u32>(bits, 6u))), 1) : static_cast<i16>(0);
  if (!a_voiced) {
    if (b_voiced) {
      a[7] = 0;
      i16 n = static_cast<i16>(shl16_al2b2(1, bits));
      b[7] = static_cast<i16>(harmonic_index(code, n, freq_b, cap, static_cast<i16>(enable)));
    }
    return 1;
  }
  if (!b_voiced) {
    b[7] = 0;
    i16 n = static_cast<i16>(shl16_al2b2(1, bits));
    a[7] = static_cast<i16>(harmonic_index(code, n, freq_a, cap, static_cast<i16>(enable)));
    return 1;
  }
  u16 n = shl16_al2b2(1, static_cast<u32>(static_cast<i32>(bits) >> 1));
  if (bits & 1) n = static_cast<u16>(high16_al2b1(wmul<i32>(static_cast<i16>(n), 0x16a08)));
  i16 ni = static_cast<i16>(n);
  i16 q = 0, r = 0;
  if (!on) {
    i16 quo = static_cast<i16>(div32(code, ni));
    i16 rem = wsub<i16>(code, wmul<i16>(quo, ni));
    q = quo;
    r = rem;
    if (static_cast<i32>(quo) > static_cast<i32>(ni) - 1) q = static_cast<i16>(wsub<u16>(n, 1));
    if (static_cast<i32>(rem) > static_cast<i32>(ni) - 1) r = static_cast<i16>(wsub<u16>(n, 1));
  }
  a[7] = static_cast<i16>(harmonic_index(r, ni, freq_a, cap, static_cast<i16>(enable)));
  b[7] = static_cast<i16>(harmonic_index(q, ni, freq_b, cap, static_cast<i16>(enable)));
  if (on) {
    i16 v = wadd<i16>(r, wmul<i16>(q, ni));
    code = v;
    if (cap < v) code = cap;
    else if (v < 1) code = 0;
  }
  return 1;
}

void spectral_slope(S16 out, CS16 src, i16 n, CS16 taps) {
  i16 scratch[0x68] = {};
  build_spectrum_history(S16(scratch), src, n);
  hilbert_diff(out, CS16(scratch), static_cast<u16>(n), taps);
}

i32 inverse_real_fft(S16 buf, i32 exp, i32 log2n, i16 first) {
  if (first == 0) buf[1] = 0;
  i32 e = static_cast<u16>(fft_merge_stage(buf, exp, static_cast<i16>(log2n), 1, first));
  return wadd<i32>(wsub<i32>(fft(buf, e, static_cast<i16>(log2n - 1)), log2n), 1);
}

i64 allshl_al2b3(i64 v, u8 count) {
  u32 hi = static_cast<u32>(v >> 32);
  u32 lo = static_cast<u32>(v);
  if (count >= 0x40) return 0;
  if (count >= 0x20) return static_cast<i64>(static_cast<u64>(lo << (count & 0x1f)) << 32);
  u32 n = count & 0x1f;
  u32 new_hi, new_lo;
  if (n == 0) {
    new_hi = hi;
    new_lo = lo;
  } else {
    new_hi = (hi << n) | (lo >> (32 - n));
    new_lo = lo << n;
  }
  return static_cast<i64>((static_cast<u64>(new_hi) << 32) | static_cast<u64>(new_lo));
}

i32 sat_sub32(i32 a, i32 b) {
  i32 r = wsub<i32>(a, b);
  if (((~b ^ r) & (r ^ a)) < 0) return wadd<i32>(0x7fff'ffff, a < 0 ? 1 : 0);
  return r;
}

u32 mask32(CS16 rec) {
  return static_cast<u32>(static_cast<u16>(rec[4])) | (static_cast<u32>(static_cast<u16>(rec[5])) << 16);
}

void retune_from_pitch_msg(S16 state, S16 flags, CS16 msg) {
  if (frame_size_class(state[1]) != 0) return;
  if (msg[0] != 3) return;
  if (frame_size_class(msg[1]) != 0) return;
  i16 idx = msg[3];
  i16 d = wsub<i16>(state[3], idx);
  d = d < 0 ? wneg<i16>(d) : d;
  if (d <= 1 && static_cast<i32>(state[1]) < (static_cast<i32>(idx) + 1) * 0x10) {
    select_harmonic(state, flags, idx);
    return;
  }
  i32 num = wadd<i32>(wmul<i32>(static_cast<i32>(idx) + 1, state[1]), static_cast<i32>(msg[1]) / 2);
  u32 e = static_cast<u16>(div32(num, msg[1]));
  while (static_cast<i32>(static_cast<i16>(static_cast<u16>(e))) * 0x10 <= static_cast<i32>(state[1])) e = wadd<u32>(e, 1u);
  select_harmonic(state, flags, static_cast<i16>(static_cast<u16>(wsub<u32>(e, 1u))));
}

i32 block_dft_dc(S16 dst, S16 src, u16 count, u32 n, CS16 cos_tbl) {
  i32 off = 0;
  if (static_cast<i16>(count) > 0) {
    for (usz k = 0; k < count; k++) {
      harmonic_dft(src.subspan(static_cast<usz>(off)), static_cast<u16>(n), static_cast<i16>(n), cos_tbl);
      dst[k] = src[static_cast<usz>(off)];
      off = wadd<i32>(off, static_cast<i16>(static_cast<u16>(n)));
    }
  }
  return off;
}

void harmonic_idct(S16 buf, i16 n, i16 count, CS16 cos_tbl) {
  i16 cnt = count >= n ? n : count;
  i16 local[56] = {};
  local[0] = buf[0];
  if (cnt > 1) shift_copy16(S16(local).subspan(1), buf.subspan(1), static_cast<u16>(cnt - 1), 1);
  if (n <= 0) return;
  i16 recip = dsp_tbl::RECIP_N[static_cast<usz>(n)];
  i16 step = static_cast<i16>(recip >> 1);
  usz ncs = cnt > 0 ? static_cast<usz>(cnt) : 0;
  for (usz s = 0; s < static_cast<usz>(n); s++) {
    i32 acc = 0;
    i32 phase = 0x40;
    for (usz ci = 0; ci < ncs; ci++) {
      usz ti = static_cast<usz>((phase >> 7) & 0x1ff);
      phase = wadd<i32>(phase, step);
      acc = wadd<i32>(acc, wmul<i32>(wmul<i32>(cos_tbl[ti], local[ci]), 2));
    }
    step = wadd<i16>(step, recip);
    buf[s] = high16_al2b3(wadd<i32>(acc, 0x8000));
  }
}

u16 merge_voiced_amplitude(u32 flags, CS16 a, CS16 b) {
  if ((mask32(b) & 0x5555'5555u) != 0 && (flags & 0xaaaa'aaaau) == 0) {
    if ((mask32(a) & 0x5555'5555u) != 0) {
      i32 p = wmul<i32>(wmul<i32>(a[6], b[6]), 2);
      return static_cast<u16>(sqrt_to_exp(p, -8, -4));
    }
    return static_cast<u16>(b[6]);
  }
  return static_cast<u16>(a[6]);
}

void attenuate_unvoiced_bands(S16 out, CS16 inp, i16 mask, i16 level, i16 nbits, i16 negate) {
  i32 l = log2_q(level, 4);
  i16 off = high16_al2b3(wsub<i32>(wshl<i32>(l, 10), 0x0314'fc01));
  if (negate != 0) off = wneg<i16>(off);
  i16 bands = wsub<i16>(triads_capped(nbits), 1);
  if (bands < 0) return;
  i16 m = mask;
  i32 lo = static_cast<i32>(bands) * 3;
  i16 idx = 0x37;
  i32 cur = 0x37;
  for (;;) {
    bool voiced = (m & 1) != 0;
    m = static_cast<i16>(m >> 1);
    while (lo <= cur) {
      i16 v = inp[static_cast<usz>(cur)];
      if (!voiced) {
        i32 r = sat_sub32(static_cast<i32>(v) << 16, static_cast<i32>(off) << 16);
        v = high16_al2b3(r);
        if (v > 0x77ff) v = 0x77ff;
        else if (v < -0x7800) v = -0x7800;
      }
      idx = wsub<i16>(idx, 1);
      out[static_cast<usz>(cur)] = v;
      cur = idx;
    }
    bands = wsub<i16>(bands, 1);
    lo -= 3;
    if (bands < 0) return;
  }
}

void harmonic_energy_and_corr(i16& out_energy, i16& out_energy_exp, i16& out_corr, i16& out_corr_exp, S16 cos_out,
                              i16 omega, CS16 amps, i16 exp_adj, u16 n, CS16 cos_tbl) {
  i16 e = 0;
  i32 energy = dot_norm64(e, amps, amps, n);
  out_energy = high16_al2b3(energy);
  i16 adj = wmul<i16>(exp_adj, 2);
  out_energy_exp = wadd<i16>(e, adj);
  if (out_energy == 0) {
    out_corr = 0;
    out_corr_exp = 0;
    return;
  }
  i32 step = (static_cast<i32>(omega) << 10) >> 4;
  i32 phase = step;
  i64 acc = 0;
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 ti = high16_al2b3(phase);
      i16 c = cos_tbl[static_cast<usz>(ti)];
      cos_out[i] = c;
      i16 sq = high16_al2b3(wmul<i32>(wmul<i32>(amps[i], amps[i]), 2));
      acc = wadd<i64>(acc, static_cast<i64>(wmul<i32>(wmul<i32>(sq, c), 2)));
      phase = wadd<i32>(phase, step);
    }
  }
  i16 sh = static_cast<i16>(static_cast<u16>(norm64(static_cast<u32>(acc), static_cast<u32>(acc >> 32))));
  i64 shifted = sh >= 0 ? allshl_al2b3(acc, static_cast<u8>(static_cast<u32>(static_cast<i32>(sh)) & 0xff))
                        : allshr(acc, static_cast<u8>(static_cast<u32>(wneg<i32>(sh)) & 0xff));
  out_corr_exp = wsub<i16>(adj, sh);
  i16 m = high16_al2b3(static_cast<i32>(shifted));
  out_corr = m == -0x8000 ? static_cast<i16>(-0x7fff) : m;
}

i32 bsr(u32 v) { return 31 - (v ? __builtin_clz(v) : 32); }
i32 up16_al2b4(i16 x) { return static_cast<i32>(static_cast<u32>(static_cast<i32>(x)) << 16); }
i32 add_sat_sgn_al2b4(i32 a, i32 b, i32 sgn) {
  i32 r = wadd<i32>(a, b);
  if (((r ^ a) & (r ^ b)) < 0) return sgn < 0 ? INT32_MIN : INT32_MAX;
  return r;
}
i16 minus_1p25(i16 x) {
  i32 a = up16_al2b4(x);
  i32 r = wadd<i32>(a, static_cast<i32>(0xb000'0000u));
  if (((r ^ a) & ~r) < 0) r = a < 0 ? INT32_MIN : INT32_MAX;
  return high16_al2b1(r);
}

u32 pitch_config_init(S16 cfg, i16 rate, i16 lo, i16 hi, i16 apply, CS16 tbl) {
  u16 u = static_cast<u16>(rate);
  i16 cls;
  if (static_cast<u16>(u - 5) < 0x76) cls = 0;
  else if (static_cast<u16>(u - 0x80) < 0x10) cls = 1;
  else if (static_cast<u16>(u - 0x90) < 0x10) cls = 2;
  else if (static_cast<u16>(u - 0xa0) < 4) cls = 3;
  else return 4;
  cfg[0] = 3;
  cfg[4] = 1;
  cfg[5] = 0;
  cfg[7] = 0;
  fill16(cfg.subspan(8), -0x4fff, 0x38);
  auto clamp_l = [](i16 v, i16 top) -> i16 { return v > top ? top : (v < -0x5000 ? static_cast<i16>(-0x5000) : v); };
  if (cls == 0) {
    i16 band = static_cast<i16>(static_cast<u16>(band_count_for_mode(0, rate, 0, tbl)));
    cfg[1] = rate;
    cfg[3] = band;
    i32 num = static_cast<i32>(wshl<u32>(static_cast<u32>(static_cast<i32>(rate)), 20));
    i16 den = high16_al2b1(static_cast<i32>(wshl<u32>(static_cast<u32>(static_cast<i32>(band) + 1), 24)));
    cfg[6] = static_cast<i16>(div32_half(num, den));
    i16 l = wadd<i16>(band, 1);
    cfg[2] = l > 0x38 ? static_cast<i16>(0x38) : (l < 9 ? static_cast<i16>(9) : l);
    i16 peak = clamp_l(hi, 0x77ff);
    cfg[8 + static_cast<usz>(band)] = peak;
    if (apply != 0) spread_peak(cfg.subspan(8), band, minus_1p25(peak), 0x400);
    return 0;
  }
  i16 b_lo = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, rate, 0, tbl)));
  i16 b_hi = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, rate, 1, tbl)));
  cfg[1] = rate;
  usz k = static_cast<usz>(wsub<i16>(rate, 0x80));
  cfg[3] = static_cast<i16>(wadd<u32>(static_cast<u32>(static_cast<u16>(b_lo)) << 8, static_cast<u32>(static_cast<u16>(b_hi))));
  cfg[6] = tbl[dec_tbl::T_STEP + k];
  i16 l = wadd<i16>(b_hi, 1);
  cfg[2] = l > 0x38 ? static_cast<i16>(0x38) : (l < 9 ? static_cast<i16>(9) : l);
  i16 v_lo = lo, v_hi = hi;
  if (apply != 0) {
    i32 a = up16_al2b4(hi) >> 1;
    i32 b = up16_al2b4(lo) >> 1;
    i16 mid = high16_al2b1(add_sat_sgn_al2b4(a, b, b));
    i32 dev = up16_al2b4(tbl[dec_tbl::T_DEV + k]);
    i32 mv = up16_al2b4(mid);
    i32 d = wsub<i32>(mv, dev);
    if (((~dev ^ d) & (d ^ mv)) < 0) d = mv < 0 ? INT32_MIN : INT32_MAX;
    v_lo = high16_al2b1(d);
    v_hi = high16_al2b1(add_sat_sgn_al2b4(dev, mv, mv));
  }
  i16 p_lo = clamp_l(v_lo, 0x6fff);
  cfg[8 + static_cast<usz>(b_lo)] = p_lo;
  if (apply != 0) spread_peak(cfg.subspan(8), b_lo, minus_1p25(p_lo), 0x400);
  i16 p_hi = clamp_l(v_hi, 0x6fff);
  cfg[8 + static_cast<usz>(b_hi)] = p_hi;
  if (apply != 0) spread_peak(cfg.subspan(8), b_hi, minus_1p25(p_hi), 0x400);
  return 0;
}

i32 band_pair_mean(CS16 arr, i16 rate, CS16 tbl) {
  i32 cls = static_cast<i16>(frame_size_class(rate));
  i16 i_lo = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, rate, 0, tbl)));
  i16 i_hi = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, rate, 1, tbl)));
  i32 a = up16_al2b4(arr[static_cast<usz>(i_hi)]) >> 1;
  i32 b = up16_al2b4(arr[static_cast<usz>(i_lo)]) >> 1;
  return wadd<i32>(a, b) >> 16;
}

void interp_envelope(S16 out, i16 num, i16 den, CS16 src, u16 n) {
  i16 tmp[60] = {};
  spread_hold(S16(tmp), src, 0x3c);
  i32 step = ratio_q16(num, den);
  if (static_cast<i16>(n) <= 0) return;
  i32 acc = step;
  for (usz k = 0; k < n; k++) {
    i16 idx = high16_al2b1(acc);
    i16 w = high16_al2b1(static_cast<i32>(static_cast<u32>(wsub<i32>(acc, up16_al2b4(idx))) << 15));
    i16 i0 = idx > 0x3b ? static_cast<i16>(0x3b) : idx;
    i16 t = wadd<i16>(i0, 1);
    i16 i1 = t > 0x3b ? static_cast<i16>(0x3b) : t;
    i32 lo = wmul<i32>(wmul<i32>(tmp[static_cast<usz>(i0)], wsub<i16>(0x7fff, w)), 2);
    i32 hi = (w == INT16_MIN && tmp[static_cast<usz>(i1)] == INT16_MIN) ? INT32_MAX
                                                                        : wmul<i32>(wmul<i32>(tmp[static_cast<usz>(i1)], w), 2);
    out[k] = high16_al2b1(add_sat_sgn_al2b4(hi, lo, lo));
    acc = wadd<i32>(acc, step);
  }
}

void rebase_block_exp(S16 dst, CS16 src, u16 n, i32 exp_src, i32 exp_dst) {
  shift_copy16(dst, src, n, static_cast<i16>(wsub<i32>(exp_src, exp_dst)));
}

void vector_rms(i16& out_mant, i16& out_exp, CS16 vec, i16 n) {
  i16 sum_exp = 0;
  i32 sum = dot_norm64(sum_exp, vec, vec, static_cast<u16>(wmul<i16>(n, 2)));
  i16 m = high16_al2b1(sum);
  if (m <= 0) {
    out_mant = 0;
    out_exp = 0;
    return;
  }
  i32 v = up16_al2b4(n);
  u16 nrm_ = 0;
  if (v != 0) {
    u32 mag = v < 0 ? ~static_cast<u32>(v) : static_cast<u32>(v);
    nrm_ = static_cast<u16>(0x1e - bsr(mag));
  }
  i32 den = static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(nrm_) & 0x1f));
  u16 exp = static_cast<u16>(wsub<u32>(0x0fu, nrm_) & 0xffff);
  if (m <= high16_al2b1(den)) {
    den >>= 1;
    exp = wadd<u16>(exp, 1);
  }
  i16 q = static_cast<i16>(div32_half(den, m));
  i16 e = wsub<i16>(static_cast<i16>(exp), sum_exp);
  i32 r = sqrt_norm(static_cast<i32>(wshl<u32>(static_cast<u32>(static_cast<i32>(q)), 16)), e);
  out_mant = high16_al2b1(r);
  out_exp = e;
}

i16 codebook_field(S16 out, u16 nwords, i16 bits_avail, CS16 book, i16 nbits, i16 stride, S16 bits, i32& pos, i16 encode) {
  i16 nb = nbits;
  i16 st = stride;
  if (bits_avail < nbits) {
    u8 k = static_cast<u8>(static_cast<u8>(static_cast<u8>(nbits) - static_cast<u8>(bits_avail)) & 0x1f);
    nb = bits_avail;
    i16 f = k >= 16 ? static_cast<i16>(0) : static_cast<i16>(1 << k);
    st = wmul<i16>(f, stride);
  }
  i16 idx;
  if (encode != 0) {
    u8 k = static_cast<u8>(static_cast<u8>(nb) & 0x1f);
    i16 cand = k >= 16 ? static_cast<i16>(0) : static_cast<i16>(1 << k);
    i16 found = vq_search(out, book, static_cast<i16>(nwords), cand, st);
    usz cur = static_cast<usz>(pos);
    emit_bits(cur, bits, nb, found);
    pos = static_cast<i32>(cur);
    idx = found;
  } else {
    idx = static_cast<i16>(read_field_i16(bits, pos, nb));
  }
  if (static_cast<i16>(nwords) > 0) {
    i32 base = wmul<i32>(idx, st);
    for (usz i = 0; i < nwords; i++) out[i] = book[static_cast<usz>(base + static_cast<i32>(i))];
  }
  return idx;
}

u16 norm_shift_al2b5(i32 x) {
  if (x == 0) return 0;
  u32 m = x < 0 ? ~static_cast<u32>(x) : static_cast<u32>(x);
  return static_cast<u16>(0x1e - top_bit_al2b5(m));
}
i32 shift32_al2b5(i32 v, i16 s) {
  if (s >= 0) return static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(s)) & 0x1f));
  if (s < -0x1e) return v >> 31;
  return v >> (static_cast<u32>(-static_cast<i32>(s)) & 0x1f);
}
i32 sat_add_al2b5(i32 a, i32 b, i32 lim) {
  i32 s = wadd<i32>(a, b);
  if (((s ^ a) & (s ^ b)) < 0) return wadd<i32>(0x7fff'ffff, lim < 0 ? 1 : 0);
  return s;
}
i32 sat_sub_al2b5(i32 a, i32 b, i32 lim) {
  i32 d = wsub<i32>(a, b);
  if (((a ^ b) & (d ^ a)) < 0) return wadd<i32>(0x7fff'ffff, lim < 0 ? 1 : 0);
  return d;
}

void fundamental_from_log2(i16& w0_out, i16& l_out, i16 log_pitch, i16 allow_high) {
  u8 cl = 0xfc;
  i16 x = log_pitch;
  if (log_pitch < -0x1000) {
    u16 units = wadd<u16>(static_cast<u16>(wsub<u16>(0xefff, static_cast<u16>(log_pitch)) >> 12), 1);
    x = wadd<i16>(log_pitch, wmul<i16>(static_cast<i16>(units), 0x1000));
    cl = static_cast<u8>(wsub<u16>(units, 4));
  }
  i32 d = x;
  i32 a = static_cast<i16>(wadd<i32>(wmul<i32>(d, 0x13b) >> 12, 0x71b));
  a = static_cast<i16>(wadd<i32>(wmul<i32>(a, d) >> 12, 0x1ec0));
  a = static_cast<i16>(wadd<i32>(wmul<i32>(a, d) >> 12, 0x58b9));
  i16 v = static_cast<i16>(wsub<i32>(wmul<i32>(a, d) >> 12, 0x8000));
  u32 c = cl & 0x1f;
  i16 w0 = c >= 16 ? static_cast<i16>(v >> 15) : static_cast<i16>(v >> c);
  if (w0 < 0x1079) w0 = 0x1079;
  else if (w0 > 0x6bca && allow_high == 0) w0 = 0x6bca;
  w0_out = w0;
  l_out = harmonic_count(w0);
}

i16 sat15(i32 v) { return v > 0x7fff ? static_cast<i16>(0x7fff) : (v < -0x7fff ? static_cast<i16>(-0x7fff) : static_cast<i16>(v)); }

void prba_transform(S16 out, S16 scratch, CS16 b, CS16 a, u16 n, CSU16 lens, CS16 cos_tbl) {
  if (static_cast<i16>(n) > 0) {
    for (usz i = 0; i < n; i++) {
      i32 d = static_cast<i32>(a[i]) - static_cast<i32>(b[i]);
      scratch[i] = d > 0x7fff ? static_cast<i16>(0x7fff) : (d < -0x7fff ? static_cast<i16>(-0x7fff) : static_cast<i16>(d));
    }
  }
  usz off = 0;
  for (usz k = 0; k < 4; k++) {
    u16 len = lens[k];
    harmonic_dft(scratch.subspan(off), len, 6, cos_tbl);
    out[2 * k] = scratch[off];
    out[2 * k + 1] = scratch[off + 1];
    fill16(scratch.subspan(off + 6), 0, wsub<i16>(static_cast<i16>(len), 6));
    off += static_cast<usz>(static_cast<i16>(len));
  }
  for (usz k = 0; k < 4; k++) {
    i32 p = wmul<i32>(out[2 * k + 1], 0xb504);
    i32 q = (static_cast<i32>(out[2 * k]) << 16) >> 1;
    out[2 * k] = sat15(wadd<i32>(p, q) >> 15);
    out[2 * k + 1] = sat15(wsub<i32>(q, p) >> 15);
  }
  harmonic_dft(out, 8, 8, cos_tbl);
}

void normalise_to_exp(S16 dst, CS16 src, u16 n, i16& exp) {
  u16 e = static_cast<u16>(norm_exp_absmax(src, n));
  shift_copy16(dst, src, n, static_cast<i16>(wneg<u16>(e)));
  exp = wadd<i16>(exp, static_cast<i16>(e));
}

void suppress_spectral_peaks(S16 cur, S16 prev, i16 frame_kind, i16 sub_mode) {
  i16 l = cur[2];
  usz vn = l > 0 ? static_cast<usz>(l) : 0;
  i16 e = 0;
  i32 energy;
  {
    CS16 mags = CS16(cur).subspan(8, vn);
    energy = dot_norm64(e, mags, mags, static_cast<u16>(l));
  }
  e = wadd<i16>(e, 8);
  i32 ll = static_cast<i32>(l) << 16;
  u16 sh = norm_shift_al2b5(ll);
  i32 den = static_cast<i32>(static_cast<u32>(ll) << (static_cast<u32>(sh) & 0x1f)) >> 16;
  i16 mean = div_norm(energy, e, static_cast<i16>(den), static_cast<i16>(0xf - static_cast<i32>(sh)), e);
  u16 rms = static_cast<u16>(sqrt_to_exp(static_cast<i32>(mean) << 16, e, 4));
  i32 t = prev[0x41];
  i32 s0 = sat_add_al2b5(t << 16, static_cast<i32>(0xf800'0000u), t << 16);
  i32 s1 = sat_sub_al2b5(s0, static_cast<i32>(static_cast<i16>(rms)) << 16, s0);
  if (s1 < 0) {
    prev[0x41] = static_cast<i16>(wadd<i32>(wmul<i32>(t, 0xfae0), wmul<i32>(static_cast<i16>(rms), 0xa3d6) >> 5) >> 16);
  } else {
    prev[0x41] = static_cast<i16>(wsub<i32>(t << 16, 0x0013'0001) >> 16);
  }
  if (frame_kind < 4 && sub_mode < 2) return;
  i16 curve[60] = {};
  spread_hold_alt(S16(curve), 0x3c, prev.subspan(8));
  i32 tr = prev[0x41];
  i32 base = wsub<i32>(0x2c00'0000, (static_cast<i32>(sub_mode) << 27) >> 1);
  u16 floor0 = static_cast<u16>(sat_add_al2b5(tr << 16, base, base) >> 16);
  i32 w1 = static_cast<i32>(cur[6]) << 16;
  u16 n1 = norm_shift_al2b5(w1);
  i32 num = static_cast<i32>(static_cast<u32>(w1) << (static_cast<u32>(n1) & 0x1f));
  i32 w2 = static_cast<i32>(prev[6]) << 16;
  u16 n2 = norm_shift_al2b5(w2);
  i32 den2 = static_cast<i32>(static_cast<u32>(w2) << (static_cast<u32>(n2) & 0x1f)) >> 16;
  i16 ratio_exp = 0;
  i16 ratio = div_norm(num, static_cast<i16>(static_cast<u16>(wsub<u32>(wsub<u32>(0u, n1), 4u))), static_cast<i16>(den2),
                       static_cast<i16>(static_cast<u16>(wsub<u32>(0xffff'fffcu, n2))), ratio_exp);
  if (l <= 0) return;
  i32 floor_q31 = static_cast<i32>(static_cast<i16>(floor0)) << 16;
  i32 ratio_i = ratio;
  u16 i = 0;
  for (;;) {
    i16 k = static_cast<i16>(wadd<u16>(i, 1));
    i32 tilt6 = static_cast<i16>(wmul<i32>(k, 0x60000) >> 16);
    i32 tilt = (wmul<i32>(tilt6, cur[6]) << 8) >> 16;
    tilt = static_cast<u16>(tilt);
    if (static_cast<i16>(static_cast<u16>(tilt)) > 0x1000) tilt = 0x1000;
    i32 fl = wsub<i32>(floor_q31, static_cast<i32>(static_cast<i16>(static_cast<u16>(tilt))) << 16);
    u16 fl16 = static_cast<u16>(fl >> 16);
    i32 warped = shift32_al2b5(wmul<i32>(wmul<i32>(ratio_i, k), 2), ratio_exp);
    i32 idx = static_cast<i16>(static_cast<u16>(warped >> 16));
    i32 frac = static_cast<i16>(static_cast<u16>((wsub<i32>(warped, idx << 16) << 15) >> 16));
    i32 co = static_cast<i16>(0x7fff - frac);
    i32 interp = static_cast<i16>(static_cast<u16>(
        wmul<i32>(wadd<i32>(wmul<i32>(co, curve[static_cast<usz>(idx)]), wmul<i32>(curve[static_cast<usz>(idx) + 1], frac)), 2) >> 16));
    usz p = 8 + static_cast<usz>(static_cast<i16>(i));
    i32 interp_q31 = interp << 16;
    i32 c = cur[p];
    i32 d1 = sat_sub_al2b5(c << 16, interp_q31, c << 16);
    if (sat_add_al2b5(d1, -0x1000'0000, d1) > 0) {
      c = cur[p];
      i32 d3 = sat_sub_al2b5(c << 16, static_cast<i32>(static_cast<i16>(fl16)) << 16, c << 16);
      if (d3 > 0) {
        i16 g = static_cast<i16>(static_cast<u16>(d3 >> 16));
        if (g > 0x400) g = 0x400;
        c = cur[p];
        i16 delta = static_cast<i16>(sat_sub_al2b5(interp_q31, c << 16, interp_q31) >> 16);
        i32 prod = (delta == -0x8000 && g == -0x8000) ? 0x7fff'ffff : wmul<i32>(wmul<i32>(delta, g), 2);
        i32 step = shl_sat(prod, 4);
        c = cur[p];
        i16 ns = static_cast<i16>(sat_add_al2b5(c << 16, step, step) >> 16);
        cur[p] = ns;
        cur[p] = ns > 0x77ff ? static_cast<i16>(0x77ff) : (ns < -0x77ff ? static_cast<i16>(-0x77ff) : ns);
      }
    }
    i = wadd<u16>(i, 1);
    if (static_cast<i16>(i) >= l) break;
  }
}

void synth_harmonic_span(std::span<i32> dst, i16 pos, i16 pos_exp, i16 step, CS16 table, i32 exp, CS16 cos_tbl) {
  i16 start = 0, start_frac = 0;
  ceil_and_residual(start, start_frac, pos, pos_exp);
  i16 end = 0, end_frac = 0;
  add_recip_split(end, end_frac, pos, pos_exp, step);
  i16 e = end;
  if (0xa6 < e) e = 0xa6;
  u16 last;
  if (static_cast<i32>(e) + 0x10 <= 0xa6) {
    last = wadd<u16>(static_cast<u16>(e), 0x10);
    if (static_cast<i16>(last) <= 0) return;
  } else {
    last = 0xa6;
  }
  i32 fall_phase = wadd<i32>((static_cast<i32>(end_frac) << 16) >> 15, 0xf'0000);
  i32 rise_phase = (static_cast<i32>(start_frac) << 16) >> 15;
  i32 stp = step;
  i32 phase = wmul<i32>(wmul<i32>(stp, start_frac), 2) >> 11;
  u16 cur;
  if (start < 0) {
    phase = wadd<i32>(phase, wmul<i32>(wneg<i16>(start), stp) << 5);
    if (start < -0x10) rise_phase = wadd<i32>(rise_phase, 0x10'0000);
    else rise_phase = wsub<i32>(rise_phase, static_cast<i32>(start) << 16);
    if (e < 0) fall_phase = wadd<i32>(fall_phase, static_cast<i32>(wadd<i16>(e, 1)) << 16);
    cur = 0;
  } else {
    cur = static_cast<u16>(start);
  }
  i16 shift = static_cast<i16>(static_cast<u16>(wsub<i32>(exp, 0x10)));
  usz cursor = static_cast<usz>(static_cast<isz>(static_cast<i16>(cur)));
  u16 n_rise = wadd<u16>(wsub<u16>(static_cast<u16>(start), cur), 0x10);
  if (static_cast<i16>(n_rise) > 0) {
    cur = wadd<u16>(cur, n_rise);
    synth_harmonic(phase, table, step, rise_phase, cursor, dst, shift, 1, n_rise, cos_tbl);
  }
  u16 n_body = wadd<u16>(wsub<u16>(static_cast<u16>(e), cur), 1);
  if (static_cast<i16>(n_body) > 0) {
    cur = wadd<u16>(cur, n_body);
    synth_harmonic(phase, table, step, 0, cursor, dst, shift, 0, n_body, cos_tbl);
  }
  u16 n_fall = wadd<u16>(wsub<u16>(last, cur), 1);
  synth_harmonic(phase, table, step, fall_phase, cursor, dst, shift, -1, n_fall, cos_tbl);
}

i16 shifted_hi(i32 v, i16 sh) {
  i32 t;
  if (sh >= 0) t = static_cast<i32>(static_cast<u32>(v) << (static_cast<u32>(static_cast<i32>(sh)) & 0x1f));
  else if (sh < -0x1e) t = v >> 31;
  else t = v >> (static_cast<u32>(-static_cast<i32>(sh)) & 0x1f);
  return static_cast<i16>(t >> 16);
}

void dec_init(S16 state, i16 mode) {
  fill16(state, 0, 0x3f4);
  state[0x7c2 / 2] = -0x1e;
  state[0x7be / 2] = 0x1c94;
  (void)frame_size_class(0x20);
  state[0x79c / 2] = 0x20;
  {
    i16 g = 0;
    (void)gain_for_rate(g, 0x20, -10);
    state[0x79e / 2] = g;
  }
  init_level_tracker(state.subspan(0x470 / 2));
  init_level_tracker(state.subspan(0x4f8 / 2));
  for (usz i = 0; i < 0x54; i++) {
    i16 seed = state[0x648 / 2];
    rand_step(seed);
    state[0x648 / 2] = seed;
    state[0x64a / 2 + i] = seed;
  }
  A13 alloc(state.data() + 0x7a0 / 2, 13);
  (void)init_bit_alloc_and_agc(mode, alloc, state.subspan(0x580 / 2));
}

void quantize_coeffs_six_blocks(S16 payload, i32& pos, S16 vals, CSU16 allocs, i16 total, i16 encode, CS16 weight,
                                CS16 loading) {
  u16 q = static_cast<u16>(div32(total, 6));
  u16 r = static_cast<u16>(wsub<i32>(wmul<i32>(wadd<i32>(static_cast<i32>(q), 1), 6), total));
  quantize_coeff_blocks(payload, pos, static_cast<i16>(r), wsub<u16>(q, 1), vals, allocs.subspan(5), encode, weight, loading);
  usz v_off = static_cast<usz>(static_cast<isz>(wmul<i16>(static_cast<i16>(q), static_cast<i16>(r))));
  usz a_off = 5 + static_cast<usz>(static_cast<isz>(wmul<i16>(static_cast<i16>(wsub<u16>(q, 1)), static_cast<i16>(r))));
  quantize_coeff_blocks(payload, pos, static_cast<i16>(6 - static_cast<i32>(r)), q, vals.subspan(v_off), allocs.subspan(a_off),
                        encode, weight, loading);
}

void expand_prba_to_log_mags(S16 out, u16 l, S16 prba, S16 mags, CS16 block_len, i16 gain, CS16 cos_tbl) {
  harmonic_idct(prba, 8, 8, cos_tbl);
  for (usz i = 0; i < 4; i++) {
    i32 a = wmul<i32>(prba[2 * i], 0x8000);
    i32 b = wmul<i32>(prba[2 * i + 1], 0x8000);
    prba[2 * i] = high16_al2b1(wadd<i32>(b, a));
    i32 d = high16_al2b1(wsub<i32>(a, b));
    prba[2 * i + 1] = high16_al2b1(wmul<i32>(d, 0xb504));
  }
  usz dst = 0;
  for (usz i = 0; i < 4; i++) {
    mags[dst] = prba[2 * i];
    mags[dst + 1] = prba[2 * i + 1];
    i16 n = block_len[i];
    harmonic_idct(mags.subspan(dst), n, 6, cos_tbl);
    dst += static_cast<usz>(static_cast<isz>(n));
  }
  i16 lg = static_cast<i16>(log2_at_q(static_cast<i16>(l), 0xf, 5));
  i32 acc = wmul<i32>(wmul<i32>(wsub<i16>(gain, lg), static_cast<i16>(l)), 2);
  for (usz i = 0; i < 4; i++) {
    i32 t = wmul<i32>(wmul<i32>(block_len[i], prba[2 * i]), 2);
    acc = wsub<i32>(acc, t);
  }
  i16 offset = static_cast<i16>(div32_half(acc, static_cast<i16>(l)));
  if (static_cast<i16>(l) > 0) {
    i32 base = wmul<i32>(offset, 0x8000);
    for (usz i = 0; i < l; i++) {
      i32 x = wmul<i32>(mags[i], 0x8000);
      out[i] = static_cast<i16>(shl_sat(wadd<i32>(x, base), 1) >> 16);
    }
  }
}

void scale_to_target_energy(S16 buf, i16& exp, u16 n, i16 target, i16 target_exp) {
  usz nn = static_cast<i16>(n) > 0 ? static_cast<usz>(n) : 0;
  i16 src[56] = {};
  std::copy_n(buf.begin(), nn, src);
  normalise_to_exp(buf, CS16(src), n, exp);
  i16 e = 0;
  i32 energy = dot_norm64(e, buf, buf, n);
  e = wadd<i16>(e, wmul<i16>(exp, 2));
  i16 den = high16_al2b1(energy);
  if (den == 0) return;
  i16 scratch = 0;
  i16 m = div_norm(wmul<i32>(target, 0x10000), target_exp, den, e, scratch);
  i32 g = sqrt_norm(wmul<i32>(m, 0x10000), scratch);
  std::copy_n(buf.begin(), nn, src);
  scale_vec(buf, CS16(src), static_cast<i16>(n), high16_al2b1(g));
  exp = wadd<i16>(exp, scratch);
  if (exp > 0) {
    std::copy_n(buf.begin(), nn, src);
    scale_block_exp(buf, CS16(src), n, static_cast<u16>(exp), 0);
    exp = 0;
  }
}

i16 build_voiced_spectrum(S16 spec, i16& exp_out, i16 w0, CS16 phase_resid, CS16 tags, CS16 mags, i32 exp_in, i16 kmax,
                          i16 kmin, i16 tag, CS16 cos_tbl) {
  u16 e = static_cast<u16>(norm_exp_max_tagged(mags, tags, kmin, kmax, tag));
  fill16(spec, 0, 0x100);
  i16 cnt = 0;
  if (kmin < kmax) {
    i16 sh = wneg<i16>(static_cast<i16>(e));
    usz idx = static_cast<u16>(kmin);
    u16 kp1 = wadd<u16>(static_cast<u16>(kmin), 1);
    u16 left = static_cast<u16>(wsub<i16>(kmax, kmin));
    for (;;) {
      if (tags[idx] == tag) {
        i32 k = static_cast<i16>(kp1);
        i32 phase = wadd<i32>(wmul<i32>(wadd<i32>(wmul<i32>(w0, k), 0x20), 0x400), (static_cast<i32>(phase_resid[idx]) << 16) >> 6);
        i32 f = high16_al2b1(frac_at_exp(phase, 6));
        u32 r = shift_round16_batch2(wmul<i32>(f, 0x400), 0xf);
        usz ci = static_cast<usz>((static_cast<i32>(r) >> 16) & 0x1ff);
        i32 m = mags[idx];
        i32 p = wmul<i32>(wmul<i32>(cos_tbl[ci], m), 2);
        spec[static_cast<usz>(k) * 2] = shifted_hi(p, sh);
        usz ci2 = static_cast<usz>(static_cast<isz>(wsub<i32>(static_cast<i32>(ci) << 16, 0x7f'0001) >> 16)) & 0x1ff;
        i32 p2 = wmul<i32>(wmul<i32>(cos_tbl[ci2], m), 2);
        cnt = wadd<i16>(cnt, 1);
        spec[static_cast<usz>(k) * 2 + 1] = shifted_hi(p2, sh);
      }
      kp1 = wadd<u16>(kp1, 1);
      idx += 1;
      left = wsub<u16>(left, 1);
      if (left == 0) break;
    }
  }
  i32 ex = wadd<i32>(wadd<i32>(exp_in, 0x17), static_cast<i32>(e));
  exp_out = static_cast<i16>(ex);
  i32 r = cnt < 1 ? -0x20 : inverse_real_fft(spec, ex, 8, 0);
  exp_out = static_cast<i16>(r);
  spec[0x100] = spec[0];
  return cnt;
}

i16 decode_tone_frame(S16 cfg, S16 prev, CS16 bits, i32& pos, CS16 alloc, CS16 tbl) {
  read_field_i16(bits, pos, 6);
  i16 amp_hi = static_cast<i16>(read_field_i16(bits, pos, 6));
  i16 amp2 = wmul<i16>(amp_hi, 2);
  i16 rep[8] = {};
  fill16(S16(rep), 0, 8);
  i16 n = static_cast<i16>((static_cast<i32>(alloc[2]) - 0xd) >> 3);
  if (n > 0) {
    for (usz i = 0; i < static_cast<u16>(n) && i < 8; i++) rep[i] = static_cast<i16>(read_field_i16(bits, pos, 8));
  }
  i16 tone = rep[0];
  i16 errs = 0;
  if (n > 1) {
    for (usz k = 1; k < static_cast<u16>(n); k++) {
      i32 d = static_cast<i32>(rep[k]) ^ static_cast<i32>(rep[0]);
      errs = wadd<i16>(errs, static_cast<i16>(popcount_low(static_cast<u32>(d), 8)));
    }
  }
  if (((static_cast<u8>(rep[1]) ^ static_cast<u8>(rep[0])) & 0xf0) != 0 || errs >= 6) tone = -1;
  i16 lsb = static_cast<i16>(read_field_i16(bits, pos, 1));
  i32 v = static_cast<i32>(lsb) | static_cast<i32>(amp2);
  i16 r = static_cast<i16>(wsub<i32>(v << 17, 0x0010'0001) >> 16);
  i16 level = static_cast<i16>(wmul<i32>(r, 0x0078'f200) >> 16);
  i16 st = static_cast<i16>(static_cast<u16>(pitch_config_init(cfg, tone, level, level, 0, tbl)));
  if (st == 0) copy_params_l1_batch3(prev, cfg);
  return st;
}

usz expand_rows_idct(S16 dst, CS16 src, i16 count, i16 n, CS16 cos_tbl) {
  if (count <= 0) return 0;
  isz stride = n;
  isz off = 0;
  for (usz i = 0; i < static_cast<u16>(count); i++) {
    dst[static_cast<usz>(off)] = src[i];
    harmonic_idct(dst.subspan(static_cast<usz>(off)), n, n, cos_tbl);
    off += stride;
  }
  return static_cast<usz>(off);
}

void pack_imbe_frame(S16 chan, i16 nharm, CS16 hbits, CS16 alloc) { interleave_imbe_frame(chan, nharm, hbits, alloc, true); }

void shape_by_headroom(S16 spec, i16& exp, i16 n, i16 scale, i16 exp_scale, i16 thresh, i16 exp_thresh, CS16 reference) {
  i16 lo = static_cast<i16>(n >> 3);
  usz keep = lo > 0 ? static_cast<usz>(lo) : 0;
  i16 head[7] = {};
  std::copy_n(spec.begin(), keep, head);
  i32 e_blk = static_cast<u16>(exp);
  rebase_block_exp(spec, CS16(head), static_cast<u16>(lo), e_blk, e_blk + 1);
  i16 e = wadd<i16>(exp_thresh > exp_scale ? exp_thresh : exp_scale, 1);
  if (lo >= n) {
    exp = wadd<i16>(exp, 1);
    return;
  }
  i16 sh_scale = wsub<i16>(exp_scale, e);
  i16 sh_thresh = wsub<i16>(exp_thresh, e);
  i32 t = static_cast<i32>(thresh) << 16;
  i32 g = scale;
  u16 count = wsub<u16>(static_cast<u16>(n), static_cast<u16>(lo));
  for (usz k = 0; k < count; k++) {
    usz i = static_cast<usz>(static_cast<u16>(lo)) + k;
    i32 a = shift32_al3b2(wmul<i32>(wmul<i32>(reference[i], g), 2), sh_scale);
    i32 margin = wsub<i32>(shift32_al3b2(t, sh_thresh), a);
    i16 gain;
    if (margin == 0) {
      gain = clamp_gain(0, -1);
    } else {
      i16 ex = e;
      i32 s1 = sqrt_norm(margin, ex);
      ex = wadd<i16>(ex, exp);
      i32 arg = wmul<i32>(wmul<i32>(high16_al2b3(s1), spec[i]), 2);
      i32 s2 = sqrt_norm(arg, ex);
      gain = ex < 2 ? clamp_gain(s2, wsub<i16>(ex, 1)) : static_cast<i16>(0x4ccd);
    }
    spec[i] = high16_al2b3(wmul<i32>(wmul<i32>(spec[i], gain), 2));
  }
  exp = wadd<i16>(exp, 1);
}

i16 clamp_gain(i32 v, i16 sh) {
  i16 h = high16_al2b3(shift32_al3b2(v, sh));
  if (h > 0x4ccd) return 0x4ccd;
  if (h < 0x2000) return 0x2000;
  return h;
}

void synth_harmonic_bank(std::span<i32> dst, i16& last_pos, i16& last_exp, i16 step, CS16 table, i32 exp, i32 x0, i16 k,
                         i16 a, i16 c, u16 count, CS16 cos_tbl) {
  i16 pos[20] = {};
  i16 pexp[20] = {};
  solve_quad_ramp(S16(pos), S16(pexp), x0, a, k, c, count);
  if (static_cast<i16>(count) > 0) {
    for (usz i = 0; i < count; i++) synth_harmonic_span(dst, pos[i], pexp[i], step, table, exp, cos_tbl);
  }
  usz last = static_cast<usz>(static_cast<i16>(count) - 1);
  last_pos = pos[last];
  last_exp = pexp[last];
}

i32 add_sat_sgn_al3b3(i32 a, i32 b, i32 sgn) {
  i32 r = wadd<i32>(a, b);
  if (((r ^ a) & (r ^ b)) < 0) return static_cast<i32>(wadd<u32>(0x7fff'ffffu, sgn < 0 ? 1u : 0u));
  return r;
}
i32 sub_sat(i32 a, i32 b) {
  i32 r = wsub<i32>(a, b);
  if (((~b ^ r) & (r ^ a)) < 0) return static_cast<i32>(wadd<u32>(0x7fff'ffffu, a < 0 ? 1u : 0u));
  return r;
}

isz fill_at(S16 buf, isz base, i16 val, i16 n) {
  if (n > 0) {
    usz a = static_cast<usz>(base);
    fill16(buf.subspan(a), val, n);
  }
  return base + n;
}

void fundamental_from_lag(i16 mode, i16& w0, i16& l, i16 lag, i16 win, i16 allow_high) {
  if (mode != 1) {
    w0 = 0x4027;
    l = 0xf;
    return;
  }
  u32 k = static_cast<u32>(wsub<i32>(win, 6)) & 0x1f;
  u16 base16 = k >= 16 ? static_cast<u16>(0) : static_cast<u16>(0x3cu << k);
  i16 den = static_cast<i16>(wmul<u16>(base16, 2));
  i32 num = wmul<i32>(wadd<i16>(wmul<i16>(lag, 2), 1), 0x2a52);
  i16 q = static_cast<i16>(div32(num, den));
  i16 log_pitch = wsub<i16>(static_cast<i16>(0xbb03u), q);
  fundamental_from_log2(w0, l, log_pitch, allow_high);
}

void unvoiced_envelope(S16 out, i16 n, CS16 mags, i16 l, i16 level, i16 mask) {
  i16 scratch[0x38] = {};
  attenuate_unvoiced_bands(S16(scratch), mags, mask, level, l, 0);
  i32 sum = resample_lerp(out, static_cast<u16>(n), CS16(scratch), l);
  i16 coef;
  if (n < 0x10) coef = 0x3333;
  else if (n < 0x19) coef = high16_al2b3(wsub<i32>(wmul<i32>(n, 0x3d7'0800), 0x666'0000));
  else coef = 0x599a;
  scale_bias_vec(out, sum, n, coef);
}

void average_neighbour_envelopes(S16 cur, CS16 prev, CS16 next) {
  i16 l = cur[2];
  i16 a[0x38] = {}, b[0x38] = {};
  interp_envelope(S16(a), cur[6], prev[6], prev.subspan(8), static_cast<u16>(l));
  interp_envelope(S16(b), cur[6], next[6], next.subspan(8), static_cast<u16>(l));
  for (i16 i = 0; i < cur[2]; i = wadd<i16>(i, 1)) {
    i32 s = wadd<i32>(b[static_cast<usz>(i)], a[static_cast<usz>(i)]) >> 1;
    i16 v = static_cast<i16>(s);
    if (v < -0x77ff) v = -0x77ff;
    else if (v > 0x77ff) v = 0x77ff;
    cur[8 + static_cast<usz>(i)] = v;
  }
  i16 n = cur[2];
  if (n < 0x38) {
    i16 pad = cur[7 + static_cast<usz>(n)];
    u32 k = wsub<u16>(0x38, static_cast<u16>(n));
    usz d = 8 + static_cast<usz>(n);
    while (k != 0) {
      cur[d++] = pad;
      k -= 1;
    }
  }
}

i16 shape_noise_bands(S16 spec, i16& exp_out, i16 n, CS16 st, CS16 flags, i16 zero_head, i16 scale) {
  bool mode2 = st[0] == 2;
  u16 half = static_cast<u16>(n >> 1);
  i16 limit = n == 0x200 ? static_cast<i16>(0xf2) : static_cast<i16>(0x79);
  u16 peak = static_cast<u16>(peak_exp_unflagged(st.subspan(8), flags, st[2]));
  u16 exp = 0x11;
  i16 mant;
  if (mode2) {
    i16 e = static_cast<i16>(exp);
    i32 r = sqrt_norm(wmul<i32>(st[2], 0x86b6), e);
    i32 c = wmul<i32>(static_cast<i16>(r >> 16), 0xaff4);
    exp = wadd<u16>(static_cast<u16>(e), wadd<u16>(static_cast<u16>(exp_out), 1));
    mant = high16_al2b3(c);
  } else {
    i16 e = static_cast<i16>(exp);
    i32 x = wmul<i16>(st[2], scale);
    i32 r = sqrt_norm(wmul<i32>(x, 0x86b6), e);
    exp = static_cast<u16>(e);
    mant = static_cast<i16>(r >> 16);
  }
  u16 total = wadd<u16>(wadd<u16>(exp, peak), 2);
  i32 nn = n;
  i32 step = wmul<i32>(wmul<i32>(st[6], nn), 2) >> 4;
  i32 acc = wadd<i32>(step, 0x10000) >> 1;
  u16 idx = static_cast<u16>(static_cast<i16>(acc >> 16));
  isz ptr = fill_at(spec, 0, 0, static_cast<i16>(wmul<u16>(idx, 2)));
  i16 voiced_bands = 0;
  if (0 < st[2]) {
    i16 sh = wneg<i16>(static_cast<i16>(peak));
    i16 src[112] = {};
    i16 i = 0;
    for (;;) {
      acc = wadd<i32>(acc, step);
      u16 prev = idx;
      i16 cur = static_cast<i16>(acc >> 16);
      if (cur > static_cast<i16>(half)) cur = static_cast<i16>(half);
      idx = static_cast<u16>(cur);
      i16 cnt = static_cast<i16>(wsub<u16>(idx, prev));
      if (flags[static_cast<usz>(i)] != 0) {
        fill_at(spec, ptr, 0, wmul<i16>(cnt, 2));
      } else {
        i16 m, e;
        if (mode2) {
          m = mant;
          e = static_cast<i16>(exp);
        } else {
          i16 rm = 0, re = 0;
          vector_rms(rm, re, CS16(spec).subspan(static_cast<usz>(ptr)), cnt);
          m = high16_al2b3(wmul<i32>(wmul<i32>(rm, mant), 2));
          e = wadd<i16>(re, static_cast<i16>(exp));
        }
        i32 p = shift32_al3b2(wmul<i32>(wmul<i32>(st[8 + static_cast<usz>(i)], m), 2), sh);
        i16 e2 = wadd<i16>(e, static_cast<i16>(peak));
        i16 coef = high16_al2b3(p);
        usz a = static_cast<usz>(ptr);
        usz w = static_cast<usz>(static_cast<isz>(cnt)) * 2;
        if (cnt > 0) std::copy_n(spec.begin() + static_cast<isz>(a), w, src);
        scale_vec_shift(spec.subspan(a), CS16(src), wmul<i16>(cnt, 2), coef, e2, static_cast<i16>(total));
        voiced_bands = wadd<i16>(voiced_bands, 1);
      }
      ptr += static_cast<isz>(cnt) * 2;
      i = wadd<i16>(i, 1);
      if (i >= st[2]) break;
    }
  }
  u16 rem = static_cast<u16>(wsub<i16>(static_cast<i16>(half), static_cast<i16>(idx)));
  if (static_cast<i32>(static_cast<i16>(half)) - static_cast<i32>(limit) > static_cast<i32>(static_cast<i16>(rem))) {
    rem = static_cast<u16>(wsub<i16>(static_cast<i16>(half), limit));
  }
  fill_at(spec, static_cast<isz>(nn - 2 * static_cast<i32>(static_cast<i16>(rem))), 0, static_cast<i16>(wmul<u16>(rem, 2)));
  fill_at(spec, 0, 0, static_cast<i16>((wmul<i32>(wmul<i32>(zero_head, nn), 2) >> 0x14) * 2));
  exp_out = wadd<i16>(st[0x42], static_cast<i16>(total));
  return voiced_bands;
}

void quantise_gain(S16 gain, i16 mode, CS16 mags, i16 nmags, i16& prev, i16 nbits, CS16 book, S16 bits, i32& pos, i16 encode) {
  if (encode != 0) {
    gain[0] = static_cast<i16>(mean_i16(mags, static_cast<u16>(nmags)));
    i16 lg = static_cast<i16>(log2_at_q(nmags, 0xf, 5));
    i32 s1 = add_sat_sgn_al3b3(static_cast<i32>(lg) << 16, static_cast<i32>(gain[0]) << 16, static_cast<i32>(gain[0]) << 16);
    i32 s2 = sub_sat(s1, static_cast<i32>(prev) << 15);
    i32 s3 = shl_sat(s2, 0);
    gain[0] = high16_al2b3(add_sat_sgn_al3b3(s3, 0x8000, s3));
    if (mode == 2) {
      i32 v = static_cast<i32>(gain[0]) << 16;
      gain[0] = high16_al2b3(add_sat_sgn_al3b3(v, 0x1000'0000, v));
    }
  }
  codebook_field(gain, 1, nbits, book, nbits, 1, bits, pos, encode);
  if (mode == 2) gain[0] = high16_al2b3(wsub<i32>(static_cast<i32>(gain[0]) << 16, 0x0fff'0001));
  i16 g = high16_al2b3(static_cast<i32>(wshl<u32>(static_cast<u32>(wadd<i32>(prev, wmul<i32>(gain[0], 2))), 15)));
  if (g > 0x67ff) g = 0x6800;
  gain[0] = g;
  if (mode != 2) prev = g;
}

u16 norm_shift_al4b1(i32 x) {
  if (x == 0) return 0;
  u32 mag = static_cast<u32>(x < 0 ? ~x : x);
  u32 msb_ = mag == 0 ? 0u : static_cast<u32>(31 - __builtin_clz(mag));
  return static_cast<u16>(wsub<u32>(0x1eu, msb_));
}
i16 shl_hi16(i32 x, u16 sh) { return static_cast<i16>(static_cast<i32>(static_cast<u32>(x) << (static_cast<u32>(sh) & 0x1f)) >> 16); }
u32 get32_al4b1(CS16 w, usz i) { return static_cast<u32>(static_cast<u16>(w[i])) | (static_cast<u32>(static_cast<u16>(w[i + 1])) << 16); }

void put32_al4b1(S16 w, usz i, u32 v) {
  w[i] = static_cast<i16>(static_cast<u16>(v));
  w[i + 1] = static_cast<i16>(static_cast<u16>(v >> 16));
}

void select_repeat_params(S16 dst, CS16 cur, CS16 prev) {
  if (static_cast<u16>(cur[1]) != 0xff || static_cast<u16>(prev[1]) != 0xff) {
    copy_params_l1_batch3(dst, cur);
    return;
  }
  if (prev[0] == 2) {
    u32 mask = get32_al4b1(dst, 4);
    i16 lvl = dst[7];
    copy_params_l1_batch3(dst, cur);
    put32_al4b1(dst, 4, mask);
    dst[7] = lvl;
    if (cur[0] != 2) {
      for (usz i = 8; i < 8 + 0x38 && i < dst.size(); i++) dst[i] = static_cast<i16>(dst[i] >> 1);
    }
    return;
  }
  if (cur[0] != 2) {
    u16 w0 = merge_voiced_amplitude(get32_al4b1(dst, 4), cur, prev);
    dst[6] = static_cast<i16>(w0);
    dst[2] = harmonic_count(static_cast<i16>(w0));
    dst[1] = cur[1];
    average_neighbour_envelopes(dst, cur, prev);
    return;
  }
  CS16 src = (prev[0x43] < 2 && prev[0] != 3) ? prev : cur;
  copy_params_l1_batch3(dst, src);
  if (dst[7] > 0) {
    dst[7] = wsub<i16>(dst[7], 0x5000);
  } else {
    put32_al4b1(dst, 4, get32_al4b1(dst, 4) & 0x5555'5555u);
    dst[7] = 0;
  }
}

void shape_spectrum_by_voicing(S16 amps, i16& exp, i16& level, i16& level_exp, i16 omega, i16 n, CS16 cos_tbl) {
  i16 cosines[56] = {};
  i16 e_m = 0, e_e = 0, c_m = 0, c_e = 0;
  harmonic_energy_and_corr(e_m, e_e, c_m, c_e, S16(cosines), omega, amps, exp, static_cast<u16>(n), cos_tbl);
  i16 energy = e_m;
  i16 energy_exp = e_e;
  track_level(level, level_exp, e_m, e_e);
  if (energy > 0) {
    i16 a0 = e_m, b0 = e_e, c0 = c_m, d0 = c_e;
    i16 gain = 0;
    voicing_gains(gain, e_m, c_m, c_e, a0, b0, c0, d0, omega);
    shape_by_headroom(amps, exp, n, gain, e_m, c_m, c_e, CS16(cosines));
    scale_to_target_energy(amps, exp, static_cast<u16>(n), energy, energy_exp);
  }
}

void synthesise_noise_subframe(std::span<i32> acc, i16 n, S16 st, CS16 params, CS16 flags, i16 zero_head, CS16 sqrt_tbl) {
  i32 log2n = 8;
  if (n > 0x80) {
    i32 k = 8;
    for (;;) {
      log2n = k + 1;
      i32 probe = 1 << (k & 0x1f);
      k = log2n;
      if (probe >= static_cast<i32>(n)) break;
    }
  }
  i16 fftlen = (log2n & 0x1f) >= 16 ? static_cast<i16>(0) : static_cast<i16>(static_cast<u16>(1u << (log2n & 0x1f)));
  i16 win[86] = {};
  resample_linear(S16(win), n, sqrt_tbl, 0x50);
  i16 spec[256] = {};
  i16 exp_out;
  if (params[0] == 3) {
    fill16(st.subspan(1), 0, 0x54);
    fill16(S16(spec), 0, 0x100);
    exp_out = 0;
  } else {
    window_overlap(S16(spec), st[0], st.subspan(1), CS16(win), static_cast<u16>(n), fftlen);
    i16 e = static_cast<i16>(static_cast<u16>(rfft_forward(S16(spec), 0, static_cast<i16>(log2n), 0)));
    shape_noise_bands(S16(spec), e, fftlen, params, flags, zero_head, n);
    exp_out = static_cast<i16>(static_cast<u16>(inverse_real_fft(S16(spec), static_cast<u16>(e), log2n, 0)));
  }
  correlate_and_shift(acc, CS16(spec), exp_out, st.subspan(0x55, 0xa9 - 0x55), st[0xa9], CS16(win), n);
}

void code_amplitudes(CS16 prev, S16 cur, i16 gain, CS16 cfg, S16 bits, i32& pos, i16 encode, CS16 block_lens, CS16 prba_book9,
                     CS16 prba_book7, CS16 coef_books, CS16 cos_tbl) {
  u16 l = static_cast<u16>(cur[2]);
  i16 pred[56] = {};
  i32 sum = resample_lerp(S16(pred), l, prev.subspan(8), prev[2]);
  scale_bias_vec(S16(pred), sum, static_cast<i16>(l), 0x5333);
  u16 packed = static_cast<u16>(block_lens[static_cast<usz>(static_cast<isz>(wsub<i16>(static_cast<i16>(l), 9)))]);
  i16 lens[4] = {};
  for (auto& slot : lens) {
    slot = wadd<i16>(static_cast<i16>(packed & 0xf), 2);
    packed = static_cast<u16>(static_cast<i16>(packed) >> 4);
  }
  i16 prba[8] = {};
  if (encode != 0) {
    i16 a[0x38];
    std::copy_n(cur.begin() + 8, 0x38, a);
    u16 lens_u[4] = {static_cast<u16>(lens[0]), static_cast<u16>(lens[1]), static_cast<u16>(lens[2]), static_cast<u16>(lens[3])};
    prba_transform(S16(prba), cur.subspan(8), CS16(pred), CS16(a), l, CSU16(lens_u), cos_tbl);
  }
  prba[0] = 0;
  codebook_field(S16(prba).subspan(1), 3, cfg[7], prba_book9, 9, 3, bits, pos, encode);
  codebook_field(S16(prba).subspan(4), 4, cfg[8], prba_book7, 7, 4, bits, pos, encode);
  usz base = 0;
  for (usz k = 0; k < 4; k++) {
    i16 coeffs = wsub<i16>(lens[k], 2);
    i16 avail = cfg[9 + k];
    if (coeffs < 1) {
      if (encode == 0) {
        read_field_i16(bits, pos, avail);
      } else {
        usz cur_pos = static_cast<usz>(pos);
        emit_bits(cur_pos, bits, avail, 0);
        pos = static_cast<i32>(cur_pos);
      }
    } else {
      u16 nwords = static_cast<u16>(coeffs > 4 ? 4 : coeffs);
      codebook_field(cur.subspan(8 + base + 2), nwords, avail, coef_books.subspan(dec_tbl::COEF_BOOK_OFF[k]), dec_tbl::COEF_BOOK_BITS[k], 4,
                     bits, pos, encode);
    }
    base += static_cast<usz>(static_cast<isz>(lens[k]));
  }
  i16 out[0x38];
  std::copy_n(cur.begin() + 8, 0x38, out);
  i16 mags[0x38];
  std::copy_n(out, 0x38, mags);
  expand_prba_to_log_mags(S16(out), l, S16(prba), S16(mags), CS16(lens), gain, cos_tbl);
  for (usz i = 0; i < 0x38; i++) cur[8 + i] = i < l ? out[i] : mags[i];
  avg_and_extend(cur.subspan(8), CS16(pred), l);
}

void code_log_mag_residual(S16 bits, i32& pos, S16 prev, S16 cur, CSU16 alloc, i16 encode, CS16 gain_book, CS16 weight,
                           CS16 loading, CS16 band_tbl, CS16 cos_tbl) {
  i16 l = cur[2];
  u16 q = static_cast<u16>(div32(l, 6));
  u16 split = static_cast<u16>(wsub<u32>(wmul<u32>(wadd<u32>(wmul<u32>(q, 3u), 3u), 2u), static_cast<u16>(l)));
  i16 pred[56] = {};
  unvoiced_envelope(S16(pred), l, prev.subspan(8), prev[2], prev[6], prev[0x45]);
  i16 means[6] = {};
  i16 gain = 0;
  if (encode != 0) {
    vsub_sat(cur.subspan(8), CS16(pred), static_cast<u16>(l));
    if (cur[0] == 3) gain = static_cast<i16>(band_pair_mean(cur.subspan(8), cur[1], band_tbl));
    i32 off = block_dft_dc(S16(means), cur.subspan(8), split, q, cos_tbl);
    block_dft_dc(S16(means).subspan(split), cur.subspan(8 + static_cast<usz>(off)), wsub<u16>(6, split), wadd<u32>(q, 1u), cos_tbl);
    harmonic_dft(S16(means), 6, 6, cos_tbl);
    if (cur[0] != 3) gain = means[0];
  }
  means[0] = 0;
  u16 nbits[5] = {alloc[0], alloc[1], alloc[2], alloc[3], alloc[4]};
  code_higher_order_coeffs(bits, pos, nbits, S16(means), encode != 0);
  quantize_coeffs_six_blocks(bits, pos, cur.subspan(8), alloc, l, encode, weight, loading);
  harmonic_idct(S16(means), 6, 6, cos_tbl);
  usz off = expand_rows_idct(cur.subspan(8), CS16(means), static_cast<i16>(split), static_cast<i16>(q), cos_tbl);
  expand_rows_idct(cur.subspan(8 + off), CS16(means).subspan(split), static_cast<i16>(6 - static_cast<i32>(split)),
                   wadd<i16>(static_cast<i16>(q), 1), cos_tbl);
  if (encode == 0) {
    i16 idx = static_cast<i16>(read_field_i16(bits, pos, 6));
    gain = gain_book[static_cast<usz>(idx)];
  } else {
    if (cur[0] == 3) gain = wsub<i16>(gain, static_cast<i16>(band_pair_mean(cur.subspan(8), cur[1], band_tbl)));
    i16 idx = quantize_codebook64(gain);
    usz cursor = static_cast<usz>(pos);
    emit_bits(cursor, bits, 6, idx);
    pos = static_cast<i32>(cursor);
  }
  vec_add3_clamp(cur.subspan(8), CS16(pred), static_cast<u16>(l), gain);
  if (encode == 0 && l < 0x38) {
    usz cnt = wsub<u16>(0x38, static_cast<u16>(l));
    for (usz i = 0; i < cnt; i++) cur[8 + static_cast<usz>(l) + i] = cur[7 + static_cast<usz>(l) + i];
  }
  prev[0x42] = gain;
}

i32 rd32_al4b1(CS16 w, usz i) { return static_cast<i32>(get32_al4b1(w, i)); }
void wr32_al4b1(S16 w, usz i, i32 v) { put32_al4b1(w, i, static_cast<u32>(v)); }

void synthesise_voiced_subframe(std::span<i32> acc, i16 n, S16 state, CS16 cur, CS16 cur_tags, CS16 prev, i16 p6, CS16 cos_tbl) {
  i16 last_pos = state[0xaa];
  i16 last_exp = state[0xab];
  i16 prev_voiced = (get32_al4b1(prev, 4) & 0x5555'5555u) != 0 ? 1 : 0;
  i16 cur_voiced = (get32_al4b1(cur, 4) & 0x5555'5555u) != 0 ? 1 : 0;
  i32 nsig = n;
  i32 nq = nsig << 16;
  u16 sh = norm_shift_al4b1(nq);
  i16 fade_exp = static_cast<i16>(wsub<u16>(0x0f, sh));
  i16 fade_step = static_cast<i16>(static_cast<u16>(div_norm(0x4000'0000, 1, shl_hi16(nq, sh), fade_exp, fade_exp)));
  i32 pitch = rd32_al4b1(state, 0xa8);
  i32 pitch_a = pitch;
  i16 ra = 0, rb = 0, f_half = 0, f_dbl = 0;
  i32 refined = refine_pitch(pitch, ra, rb, f_half, f_dbl, prev[6], cur[6], prev_voiced, cur_voiced, n);
  i16 count = static_cast<i16>(refined >> 16);
  i32 saved = refined;
  i32 acc_b = refined;
  i16 ca = ra, cb = rb;
  i32 buf[168] = {};
  for (usz i = 0; i < 84; i++) buf[i] = rd32_al4b1(state, 2 * i);
  std::span<i32> bufs(buf);
  if (prev_voiced != 0 && static_cast<i16>(refined >> 16) > 0) {
    i32 exp = static_cast<u16>(state[0xe5]);
    synth_harmonic_bank(bufs, last_pos, last_exp, prev[6], state.subspan(0xe6), exp, pitch, rb, ra, n, static_cast<u16>(count),
                        cos_tbl);
  }
  ramp_mix(acc, bufs, fade_step, fade_exp, n);
  std::fill(std::begin(buf), std::end(buf), 0);
  i32 w_prev = wneg<i32>(static_cast<i32>(cur[6]) << 5);
  i16 kmin = static_cast<i16>(static_cast<u16>(ratio_q15(p6, cur[6])));
  i16 voiced_cnt = build_voiced_spectrum(state.subspan(0xe6), state[0xe5], static_cast<i16>((w_prev << 0xb) >> 0x10),
                                         state.subspan(0xad), cur_tags, cur.subspan(8), static_cast<u16>(cur[0x42]), cur[2], kmin, 2,
                                         cos_tbl);
  if (voiced_cnt != 0) {
    i32 v = wadd<i32>((static_cast<i32>(cur[7]) << 16) >> 8, (nsig - 0x10) << 16);
    u16 sh2 = norm_shift_al4b1(v);
    i32 exp = static_cast<u16>(state[0xe5]);
    synth_harmonic_span(bufs, shl_hi16(v, sh2), static_cast<i16>(wsub<u16>(0x0f, sh2)), cur[6], state.subspan(0xe6), exp, cos_tbl);
  }
  i32 out_acc = saved;
  if (cur_voiced != 0) {
    if (f_half == 0) {
      if (f_dbl != 0) {
        cb = static_cast<i16>((static_cast<i32>(rb) << 16) >> 0x11);
        i32 half = rd32_al4b1(state, 0xa8) >> 1;
        ca = static_cast<i16>((static_cast<i32>(ra) << 16) >> 0x11);
        out_acc = wadd<i32>(wsub<i32>(saved, rd32_al4b1(state, 0xa8)) >> 1, half);
        wr32_al4b1(state, 0xa8, half);
        pitch = half;
      }
    } else {
      pitch_advance(pitch_a, acc_b, ra, rb, prev[6], state);
      pitch = pitch_a;
      ca = ra;
      cb = rb;
      out_acc = acc_b;
    }
    count = static_cast<i16>(out_acc >> 16);
    i32 w_cur = wneg<i32>(static_cast<i32>(cur[6]) << 5);
    build_voiced_spectrum(state.subspan(0xe6), state[0xe5], static_cast<i16>((w_cur << 0xb) >> 0x10), state.subspan(0xad), cur_tags,
                          cur.subspan(8), static_cast<u16>(cur[0x42]), cur[2], 0, 1, cos_tbl);
    i32 exp = static_cast<u16>(state[0xe5]);
    synth_harmonic_span(bufs, state[0xaa], state[0xab], cur[6], state.subspan(0xe6), exp, cos_tbl);
    if (count > 0) {
      synth_harmonic_bank(bufs, last_pos, last_exp, cur[6], state.subspan(0xe6), exp, pitch, cb, ca, n, static_cast<u16>(count),
                          cos_tbl);
    }
  }
  wr32_al4b1(state, 0xa8, wsub<i32>(out_acc, static_cast<i32>(count) << 16));
  mac_ramp(acc, bufs, fade_step, fade_exp, n);
  for (usz i = 0; i < 84; i++) wr32_al4b1(state, 2 * i, buf[static_cast<usz>(n) + i]);
  if (prev_voiced == 0 && cur_voiced == 0) {
    put32_al4b1(state, 0xaa, 0);
    return;
  }
  u16 sh3 = norm_shift_al4b1(nq);
  i16 m = sub_norm(shl_hi16(nq, sh3), static_cast<i16>(wsub<u16>(0x0f, sh3)), last_pos, last_exp, state[0xab]);
  state[0xaa] = m;
}

i32 get32_al5b1(CS16 w, usz i) {
  return static_cast<i32>(static_cast<u32>(static_cast<u16>(w[i])) | (static_cast<u32>(static_cast<u16>(w[i + 1])) << 16));
}
void put32_al5b1(S16 w, usz i, i32 v) {
  w[i] = static_cast<i16>(static_cast<u16>(v));
  w[i + 1] = static_cast<i16>(static_cast<u16>(static_cast<u32>(v) >> 16));
}

void dec_synth_subframe(S16 params, S16 out, i16 n, i16 half, S16 state, CS16 hilbert, CS16 cos_tbl, CS16 sqrt_tbl) {
  using namespace dec_tbl;
  if ((static_cast<u16>(state[S_BYPASS]) & 1) == 0) {
    if (params[0] == 1) spectral_slope(state.subspan(S_SLOPE), params.subspan(8), params[2], hilbert);
    else fill16(state.subspan(S_SLOPE), 0, 0x38);
    {
      u16 count = static_cast<u16>(params[2]);
      expand_log_mags(params.subspan(8, 0x42 - 8), params[0x42], count);
    }
    if (params[0] == 1) {
      i16 omega = params[6];
      i16 count = params[2];
      shape_spectrum_by_voicing(params.subspan(8, 0x42 - 8), params[0x42], state[S_SHAPE_LEVEL], state[S_SHAPE_EXP], omega, count,
                                cos_tbl);
    }
    if (params[0] == 2) copy_params_reset_phase(state.subspan(S_TONE), params);
  }
  i16 tags[TAG_WORDS] = {};
  voicing_setup(params, S16(tags));
  if (params[1] == 0xff) {
    if (half == 0) {
      u32 flags = static_cast<u32>(get32_al5b1(params, 4));
      state[S_LEVEL] = static_cast<i16>(smooth_level_if_voiced(static_cast<u16>(state[S_LEVEL]), params[6], flags));
    }
    weight_harmonics_by_freq_sq(params, state[S_LEVEL]);
    refine_mags(params);
  } else if (state[S_PITCH_MSG] == 1) {
    retune_from_pitch_msg(params, S16(tags), state.subspan(S_PREV_AL5B1));
  } else {
    tone_frame_fixup(params, cos_tbl);
  }
  i32 acc[84] = {};
  std::span<i32> accs(acc);
  i16 level = state[S_LEVEL];
  synthesise_noise_subframe(accs, n, state.subspan(S_NOISE), params, CS16(tags), level, sqrt_tbl);
  synthesise_voiced_subframe(accs, n, state.subspan(S_VOICED, S_PREV_AL5B1 - S_VOICED), params, CS16(tags),
                             state.subspan(S_PREV_AL5B1), level, cos_tbl);
  i32 reson[6] = {};
  for (usz k = 0; k < 6; k++) reson[k] = get32_al5b1(state, S_RESON + 2 * k);
  i32 src[84];
  std::copy_n(acc, 84, src);
  resonator3(accs, std::span<const i32>(src), std::span<i32>(reson), n);
  for (usz k = 0; k < 6; k++) put32_al5b1(state, S_RESON + 2 * k, reson[k]);
  round_block_i32(out, 0xf, std::span<const i32>(acc), 0x11, static_cast<u16>(n));
  params[0x40] = 0;
  params[0x41] = 0;
  copy_params_reset_phase(state.subspan(S_PREV_AL5B1), params);
}

void code_gain_and_amplitudes(S16 prev, S16 cur, CS16 cfg, S16 bits, i32& pos, i16 encode, CS16 gain_book, CS16 block_lens,
                              CS16 prba_book9, CS16 prba_book7, CS16 coef_books, CS16 cos_tbl) {
  i16 gain[1] = {0};
  {
    i16 mode = cur[0];
    i16 nmags = cur[2];
    quantise_gain(S16(gain), mode, CS16(cur).subspan(8), nmags, prev[0x42], cfg[6], gain_book, bits, pos, encode);
  }
  code_amplitudes(prev, cur, gain[0], cfg, bits, pos, encode, block_lens, prba_book9, prba_book7, coef_books, cos_tbl);
}

void attenuate_in_place(S16 cur, i16 mask, i16 negate) {
  i16 level = cur[6];
  i16 nbits = cur[2];
  i16 src[dec_tbl::TAG_WORDS];
  std::copy_n(cur.begin() + 8, dec_tbl::TAG_WORDS, src);
  attenuate_unvoiced_bands(cur.subspan(8), CS16(src), mask, level, nbits, negate);
}

void code_log_mags_with_voicing(S16 prev, S16 cur, i16 mask, CSU16 alloc, S16 bits, i32& pos, i16 encode, CS16 gain_book,
                                CS16 weight, CS16 loading, CS16 band_tbl, CS16 cos_tbl) {
  if (encode != 0) attenuate_in_place(cur, mask, 0);
  code_log_mag_residual(bits, pos, prev, cur, alloc, encode, gain_book, weight, loading, band_tbl, cos_tbl);
  attenuate_in_place(cur, mask, 1);
  if (encode != 0) {
    usz cursor = static_cast<usz>(pos);
    emit_bits(cursor, bits, 1, static_cast<i16>(prev[0x44] & 1));
    pos = static_cast<i32>(cursor);
    prev[0x44] = wadd<i16>(prev[0x44], 1);
  }
}

u32 get32_al6b1(CS16 w, usz i) {
  return static_cast<u32>(static_cast<u16>(w[i])) | (static_cast<u32>(static_cast<u16>(w[i + 1])) << 16);
}
u16 shl16_al6b1(u16 v, i32 count) { return static_cast<u16>(static_cast<u32>(v) << (static_cast<u32>(count) & 0x1f)); }

i16 decode_frame_ambe(S16 cur, S16 synth, S16 prev, S16 bits, i16 rate, CS16 ctrl, A13 alloc, i16 allow_high, CS16 submode_tbl,
                      std::span<const u32> voicing_tbl, CS16 gain_book, CS16 block_lens, CS16 prba_book9, CS16 prba_book7,
                      CS16 coef_books, CS16 blob) {
  using namespace dec_tbl;
  if (alloc[0] == 0) {
    imbe_bit_alloc(alloc);
  } else {
    i32 sub = wsub<i32>(0x31, static_cast<u16>(submode_tbl[static_cast<usz>(static_cast<i32>(rate))]));
    ambe_bit_alloc(alloc, static_cast<i16>(sub));
  }
  i32 pos = 0;
  i16 code = static_cast<i16>(pack_bits_i16(bits, 6));
  i16 cls;
  if (static_cast<u16>(code) == 0x3f) cls = 3;
  else if (static_cast<u16>(code) == 0x3e) cls = 2;
  else cls = code < 0x3c ? 1 : 0;
  if (cls <= 0) {
    prev[S_TRACK] = -3;
    return 4;
  }
  if (cls <= 2) {
    synth[P_CLASS] = cls;
    cur[P_CLASS] = cls;
    synth[P_TAG] = 0xff;
    cur[P_TAG] = 0xff;
    frame_bit_permute(bits, CS16(alloc.data(), 13), 1);
    i16 lag = static_cast<i16>(read_field_i16(bits, pos, alloc[5]));
    i16 vidx = static_cast<i16>(read_field_i16(bits, pos, alloc[4]));
    {
      u32 m_synth = get32_al6b1(synth, P_VOICE);
      u32 m_cur = get32_al6b1(cur, P_VOICE);
      voicing_from_table(cls, m_synth, m_cur, vidx, static_cast<u16>(alloc[4]), voicing_tbl);
      put32_al4b1(synth, P_VOICE, m_synth);
      put32_al4b1(cur, P_VOICE, m_cur);
    }
    i32 split = split_voiced_freq_code(lag, synth, cur, 0, 0, static_cast<u16>(alloc[5]), 0);
    if (split != 0) {
      synth[P_W0] = 0x1079;
      cur[P_W0] = 0x1079;
      synth[P_L] = 0x38;
      cur[P_L] = 0x38;
    } else {
      i16 mode = cur[P_CLASS];
      fundamental_from_lag(mode, cur[P_W0], cur[P_L], lag, alloc[5], allow_high);
    }
    code_gain_and_amplitudes(prev, cur, CS16(alloc.data(), 13), bits, pos, 0, gain_book, block_lens, prba_book9, prba_book7,
                             coef_books, blob);
    suppress_spectral_peaks(cur, prev, ctrl[C_FRAMES], ctrl[C_SUBMODE]);
  } else {
    i16 status = decode_tone_frame(cur, synth, bits, pos, CS16(alloc.data(), 13), blob);
    if (status != 0) {
      prev[S_TRACK] = -3;
      return status;
    }
  }
  select_repeat_params(synth, cur, prev);
  copy_params_batch2(prev, cur, 0);
  prev[S_TRACK] = -3;
  return 0;
}

i16 decode_frame_imbe(S16 cur, S16 synth, S16 prev, S16 bits, i16 voffs, CS16 ctrl, A13 alloc, i16 mant, i16 exp, CS16 nib_tbl,
                      CS16 gain_book, CS16 weight, CS16 loading, CSU16 codewords, CS16 blob) {
  using namespace dec_tbl;
  if (alloc[0] == 0) imbe_bit_alloc(alloc);
  else ambe_bit_alloc(alloc, 0);
  i32 pos = 0;
  u16 coarse = pack_bits_i16(bits, 6);
  {
    i16 src[2] = {prev[S_TONE_HIST + 1], prev[S_TONE_HIST + 2]};
    copy16_batch3(prev.subspan(S_TONE_HIST), CS16(src), 2);
  }
  if (static_cast<i16>(coarse) >= 0x34) {
    prev[S_STATUS_AL6B1] = -1;
    return 4;
  }
  synth[P_CLASS] = 1;
  cur[P_CLASS] = 1;
  synth[P_TAG] = 0xff;
  cur[P_TAG] = 0xff;
  i32 widen = wsub<i32>(alloc[5], 6);
  set_pitch_params(cur, static_cast<i16>(shl16_al6b1(coarse, widen)));
  u16 hbits[56] = {};
  unpack_nibbles(SU16(hbits), cur[P_L], CS16(nib_tbl));
  i16 hbits_i[56];
  for (usz i = 0; i < 56; i++) hbits_i[i] = static_cast<i16>(hbits[i]);
  pack_imbe_frame(bits, cur[P_L], CS16(hbits_i), CS16(alloc.data(), 13));
  u32 pitch = read_field_i16(bits, pos, alloc[5]);
  set_pitch_params(cur, static_cast<i16>(pitch));
  i16 triads = triads_capped(cur[P_L]);
  u16 vfield = read_field_i16(bits, pos, triads);
  i32 vnarrow = static_cast<i16>(vfield);
  if (triads > 8) vnarrow >>= (static_cast<u32>(static_cast<i32>(triads) - 8) & 0x1f);
  prev[S_STATUS_AL6B1] = static_cast<i16>(static_cast<u16>(wadd<i32>(static_cast<i32>(pitch << 8), vnarrow)));
  u16 mask = vfield;
  i32 idx;
  i16 nb = triads;
  if (vfield == 0) {
    i16 track = prev[S_TRACK];
    u16 below = wsub<u16>(static_cast<u16>(pitch), 0x80);
    idx = (track > 0 || below > 0x37) ? 0x38 : static_cast<i32>(below);
    bool suspicious = ((static_cast<u8>(pitch) & 0x7c) == 0x18) ||
                      (static_cast<i16>(static_cast<u16>(pitch)) >= 0x80 && static_cast<i16>(static_cast<u16>(pitch)) < 0xb8);
    if (suspicious) {
      if (track > -3) prev[S_TRACK] = wsub<i16>(track, 1);
    } else if (track < 10) {
      prev[S_TRACK] = wadd<i16>(track, 1);
    }
  } else if (triads > 8) {
    u32 folded = map_voicing_bits(static_cast<i16>(vfield), triads, cur[P_L]);
    idx = wadd<i32>(static_cast<u16>(folded), 0x38);
    nb = 8;
  } else {
    idx = wadd<i32>(static_cast<i32>(vfield), 0x38);
  }
  u32 dibits = static_cast<u16>(gather_dibits(static_cast<i32>(get32_al6b1(prev, P_VOICE))));
  u32 pattern = voicing_pattern(static_cast<i16>(idx), dibits, nb);
  {
    u32 m_synth = get32_al6b1(synth, P_VOICE);
    u32 m_cur = get32_al6b1(cur, P_VOICE);
    expand_dibits(m_synth, m_cur, pattern);
    put32_al4b1(synth, P_VOICE, m_synth);
    put32_al4b1(cur, P_VOICE, m_cur);
  }
  u32 synth_mask = get32_al6b1(synth, P_VOICE);
  voiced_offset(synth[P_OFFS], synth_mask, prev[P_OFFS], get32_al6b1(prev, P_VOICE), voffs);
  i16 synth_offs = synth[P_OFFS];
  u32 cur_mask = get32_al6b1(cur, P_VOICE);
  voiced_offset(cur[P_OFFS], cur_mask, synth_offs, synth_mask, voffs);
  code_log_mags_with_voicing(prev, cur, static_cast<i16>(mask), CSU16(hbits), bits, pos, 0, gain_book, weight, loading, blob, blob);
  voicing_decision(cur, synth, prev, mask, ctrl[C_THRESH], ctrl[C_FRAMES], ctrl.subspan(C_CFG), mant, exp);
  suppress_spectral_peaks(cur, prev, ctrl[C_FRAMES], ctrl[C_SUBMODE]);
  {
    i16 prev_code = prev[P_TAG];
    i16 hist[3];
    std::copy_n(prev.begin() + S_TONE_HIST, 3, hist);
    detect_tone_frame(cur, prev_code, CS16(hist), blob, codewords);
  }
  cur[P_CLASS] = 1;
  select_repeat_params(synth, cur, prev);
  copy_params_batch2(prev, cur, static_cast<i16>(mask));
  return 0;
}

i16 decode_frame(S16 cur, S16 synth, S16 prev, S16 bits, i16 rate, i16 voffs, CS16 ctrl, A13 alloc, i16 mant, i16 exp,
                 i16 allow_high, const FrameTables& t) {
  if (alloc[0] == 0) {
    return decode_frame_imbe(cur, synth, prev, bits, voffs, ctrl, alloc, mant, exp, t.nibbles, t.gain64, t.hoc_weight, t.hoc_loading,
                             t.codewords, t.blob);
  }
  return decode_frame_ambe(cur, synth, prev, bits, rate, ctrl, alloc, allow_high, t.submode, t.voicing_patterns, t.gain32,
                           t.block_lens, t.prba9, t.prba7, t.coef_books, t.blob);
}

void state_copy(S16 cur, S16 synth, S16 state, usz at) {
  copy_params_reset_phase(cur, state.subspan(at));
  copy_params_reset_phase(synth, cur);
  state[dec_tbl::S_STATUS_AL8B1] = static_cast<i16>(state[dec_tbl::S_STATUS_AL8B1] | 1);
}

i16 finish_decoded(S16 state, i16 ret) {
  state[dec_tbl::S_CLEARED] = 0;
  bump_frame_counter(state.subspan(dec_tbl::S_PREV_AL8B1));
  return ret;
}

i16 finish_undecoded(S16 state) {
  bump_frame_counter(state.subspan(dec_tbl::S_PREV_AL8B1));
  return dec_tbl::R_REPEAT;
}

i16 decode_frame_params(S16 bits, i16 rate, u16 flags, i16 voffs, S16 cur, S16 synth, S16 state, const FrameTables& t) {
  using namespace dec_tbl;
  flags = static_cast<u16>(flags | static_cast<u16>(state[S_CTRL_FLAGS]));
  state[S_STATUS_AL8B1] = wmul<i16>(state[S_STATUS_AL8B1], 2);
  bool hard_fail = false;
  bool soft_fail = false;
  if ((flags & 0xc) == 0) {
    i16 alloc_a[13];
    std::copy_n(state.begin() + S_ALLOC, 13, alloc_a);
    i16 mant = state[S_MANT];
    i16 exp = state[S_EXP];
    u16 status = static_cast<u16>(decode_frame(cur, synth, state.subspan(S_PREV_AL8B1, S_CTRL - S_PREV_AL8B1), bits, rate, voffs,
                                               CS16(state).subspan(S_CTRL, CTRL_WORDS), A13(alloc_a), mant, exp,
                                               static_cast<i16>(flags & 0x80), t));
    std::copy_n(alloc_a, 13, state.begin() + S_ALLOC);
    if (status != 0) {
      if ((status & 8) == 0) soft_fail = true;
      else hard_fail = true;
    }
  }
  i16 sym = cur[1];
  i32 substituted = tone_seq_track(sym, state.subspan(S_TONE_SEQ), 0);
  bool have_cfg = false;
  i16 unit = 0, level = 0;
  if ((flags & 0x4000) != 0) {
    have_cfg = true;
    unit = state[S_TONE_RATE];
    level = state[S_TONE_LEVEL];
  } else if (substituted != 0) {
    unit = sym;
    i32 cls = frame_size_class(unit);
    i16 hi = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, unit, 1, t.blob)));
    i16 lo = static_cast<i16>(static_cast<u16>(band_count_for_mode(cls, unit, 0, t.blob)));
    i32 a = cur[static_cast<usz>(8 + static_cast<i32>(hi))];
    i32 b = cur[static_cast<usz>(8 + static_cast<i32>(lo))];
    have_cfg = true;
    level = static_cast<i16>((b + a) >> 1);
  }
  if (have_cfg) {
    if (pitch_config_init(cur, unit, level, level, 0, t.blob) == 0) {
      copy_params_reset_phase(synth, cur);
      return finish_decoded(state, R_TONE);
    }
  }
  usz tmpl;
  if ((flags & 8) != 0) {
    tmpl = S_TEMPLATE_GIVEUP;
  } else {
    if ((flags & 4) == 0) {
      if (hard_fail) {
        state_copy(cur, synth, state, S_TEMPLATE_GIVEUP);
        return finish_undecoded(state);
      }
      if (!soft_fail) {
        switch (cur[0]) {
          case 1: {
            i16 r = (state[S_ALLOC] == 0 && cur[1] != 0xff) ? R_TONE : R_SPEECH;
            return finish_decoded(state, r);
          }
          case 2: return finish_decoded(state, R_SILENCE);
          case 3: return finish_decoded(state, R_TONE);
          default: break;
        }
      }
    }
    tmpl = ((static_cast<u8>(static_cast<u16>(state[S_STATUS_AL8B1])) & 0xe) == 0xe) ? S_TEMPLATE_GIVEUP : S_TEMPLATE_HOLD;
  }
  state_copy(cur, synth, state, tmpl);
  return finish_undecoded(state);
}

i16 clamp_half_frame(i16 n) { return n > 0x54 ? static_cast<i16>(0x54) : (n < 0x4c ? static_cast<i16>(0x4c) : n); }

i16 dec_synth(S16 pcm, i16 n, S16 bits, i16 rate, u16 flags, i16 half, S16 state, const DecTables& t) {
  using namespace dec_tbl;
  n = clamp_half_frame(n);
  state[S_NOMINAL_N] = 0x50;
  i16 frame[PARAM_WORDS];
  std::copy_n(state.begin() + S_FRAME_PARAMS, PARAM_WORDS, frame);
  if (half == 0) {
    i16 scratch[PARAM_WORDS] = {};
    i16 ret = decode_frame_params(bits, rate, flags, n, S16(frame), S16(scratch), state, t.frame);
    std::copy_n(frame, PARAM_WORDS, state.begin() + S_FRAME_PARAMS);
    dec_synth_subframe(S16(scratch), pcm, n, 0, state, t.hilbert, t.frame.blob, t.sqrt_tbl);
    return ret;
  }
  dec_synth_subframe(S16(frame), pcm, n, half, state, t.hilbert, t.frame.blob, t.sqrt_tbl);
  std::copy_n(frame, PARAM_WORDS, state.begin() + S_FRAME_PARAMS);
  return 0;
}

}  // namespace b25::eng
