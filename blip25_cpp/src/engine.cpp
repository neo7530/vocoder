// Port of the Decoder (and FrameStatus / Frame) of blip25-vocoder
// src/engine/mod.rs.
#include "b25/engine.h"

#include <algorithm>
#include <cassert>

#include "b25/decode.h"
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

}  // namespace b25::eng
