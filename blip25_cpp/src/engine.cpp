// Port of the Decoder (and FrameStatus / Frame) of blip25-vocoder
// src/engine/mod.rs.
#include "b25/engine.h"

#include <algorithm>
#include <cassert>

#include "b25/decode.h"
#include "b25/encode.h"
#include "b25/encode_tables.h"
#include "b25/engine_tables.h"

namespace b25::eng {

using namespace rs;

Frame Frame::from_code(i16 c) {
  u16 u = static_cast<u16>(c);
  if (u & 0x8000) return {Kind::Tone, c};
  if (u & 0x0020) return {Kind::Repeat, c};
  if (u & 0x0002) return {Kind::Speech, c};
  if (u == 0) return {Kind::Silence, 0};
  return {Kind::Other, c};
}

FrameStatus FrameStatus::make(u32 raw_error_rate, bool mute) {
  FrameStatus s;
  s.error_rate = static_cast<u8>(std::min<u32>(raw_error_rate, 15));
  s.lost = false;
  s.mute = mute;
  return s;
}

std::array<i16, 26> FrameStatus::to_ctrl() const {
  std::array<i16, 26> c{};
  c[0x24 / 2] = static_cast<i16>(std::min<u8>(error_rate, 15));
  // Forcing word: bit 2 repeat, bit 3 mute.
  i16 force = 0;
  if (lost) force |= 4;
  if (mute) force |= 8;
  c[0x32 / 2] = force;
  return c;
}

static DecTables dec_tables() {
  DecTables t;
  t.frame.submode = tables::SUBMODE;
  t.frame.voicing_patterns = tables::VOICING_PATTERNS;
  t.frame.gain32 = tables::GAIN32;
  t.frame.block_lens = tables::BLOCK_LENS;
  t.frame.prba9 = tables::PRBA9;
  t.frame.prba7 = tables::PRBA7;
  t.frame.coef_books = tables::HOC_A;
  t.frame.nibbles = tables::NIBBLES;
  t.frame.gain64 = tables::GAIN64;
  t.frame.hoc_weight = tables::HOC_WEIGHT;
  t.frame.hoc_loading = tables::HOC_LOADING;
  t.frame.codewords = tables::CODEWORDS;
  t.frame.blob = tables::BLOB;
  t.hilbert = tables::HILBERT;
  t.sqrt_tbl = tables::SQRT80;
  return t;
}

Decoder::Decoder(Mode mode) : mode_(mode) { dec_init(S16(state_), static_cast<i16>(mode_)); }

void Decoder::reset() {
  state_.fill(0);
  dec_init(S16(state_), static_cast<i16>(mode_));
}

Frame Decoder::decode_with_status(CS16 payload, FrameStatus status, S16 pcm) {
  usz nb = payload_bits();
  assert(payload.size() >= nb && "payload too short");
  assert(pcm.size() >= FRAME_SAMPLES && "pcm buffer too short");

  bits_.fill(0);
  std::copy_n(payload.begin(), nb, bits_.begin());

  auto ctrl = status.to_ctrl();
  dec_ctrl(S16(state_), CS16(ctrl));

  static const DecTables t = dec_tables();
  i16 disposition = dec_synth(pcm.first(HALF), static_cast<i16>(HALF), S16(bits_), 0, 0, 0, S16(state_), t);
  dec_synth(pcm.subspan(HALF, HALF), static_cast<i16>(HALF), S16(bits_), 0, 0, 1, S16(state_), t);
  return Frame::from_code(disposition);
}


static enc::FrontEndTables enc_front_tables() {
  enc::FrontEndTables t;
  t.hamming = tables::HAMMING100;
  t.wide_win = tables::WIDE_RAMP;
  t.taper = tables::TAPER32;
  t.interp = tables::INTERP7X16;
  t.subband_win = tables::SUBBAND_WINDOW;
  t.cos = tables::COS512;
  t.fb_win = tables::FB_WINDOW;
  t.fb_tbl = tables::DCT160;
  CS16 r(tables::TONE_TABLES);
  t.tone.class_ = r.subspan(0, 16);
  t.tone.lin_lo_a = r.subspan(16, 4);
  t.tone.bin_lo_a = r.subspan(20, 4);
  t.tone.lin_lo_b = r.subspan(24, 4);
  t.tone.bin_lo_b = r.subspan(28, 4);
  t.tone.lin_hi_a = r.subspan(32, 4);
  t.tone.bin_hi_a = r.subspan(36, 4);
  t.tone.lin_hi_b = r.subspan(40, 4);
  t.tone.bin_hi_b = r.subspan(44, 4);
  return t;
}

static enc::PackTables enc_pack_tables() {
  enc::PackTables t;
  t.blob = tables::BLOB;
  t.partitions = tables::PARTITIONS;
  t.voicing_book = tables::VOICING_PATTERNS;
  t.nibbles = tables::NIBBLES;
  t.gain64 = tables::GAIN64;
  t.gain32 = tables::GAIN32;
  t.hoc_weight = tables::HOC_WEIGHT;
  t.hoc_loading = tables::HOC_LOADING;
  t.block_lens = tables::BLOCK_LENS;
  t.prba9 = tables::PRBA9;
  t.prba7 = tables::PRBA7;
  t.coef_books = tables::HOC_A;
  return t;
}

static_assert(ENC_STATE_WORDS == enc_tbl::ENC_STATE_WORDS, "encoder state size");
constexpr usz BLK_WORDS = 0x48;
constexpr i16 ENC_UNIT = 0x2000;

Encoder::Encoder(Mode mode) : mode_(mode) { enc::enc_init(S16(state_), static_cast<i16>(mode_), 1); }

void Encoder::reset() {
  state_.fill(0);
  enc::enc_init(S16(state_), static_cast<i16>(mode_), 1);
}

Frame Encoder::encode(CS16 pcm, S16 payload) {
  usz nb = payload_bits();
  assert(pcm.size() >= FRAME_SAMPLES && "pcm too short");
  assert(payload.size() >= nb && "payload buffer too short");

  bits_.fill(0);
  static const enc::FrontEndTables fe = enc_front_tables();
  static const enc::PackTables pk = enc_pack_tables();
  enc::Bases bases{0, 0};
  // The frame tail's two parameter blocks are locals of the reference's frame
  // function; a fresh (zeroed) pair every frame is what the reference produces.
  i16 blk0[BLK_WORDS] = {};
  i16 blk1[BLK_WORDS] = {};

  for (i16 half = 0; half < 2; half++) {
    usz at = HALF * static_cast<usz>(half);
    auto r = enc::enc_frame(S16(bits_), 0, pcm.subspan(at, HALF), static_cast<i16>(HALF), 0, half, ENC_UNIT, S16(state_), S16(blk0), S16(blk1),
                            bases, fe, pk);
    if (half == 1) {
      std::copy_n(bits_.begin(), nb, payload.begin());
      return Frame::from_code(static_cast<i16>(r.first));
    }
  }
  return {};
}

}  // namespace b25::eng
