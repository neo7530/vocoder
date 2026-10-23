// Public face of the codec core: Decoder (and, once ported, Encoder).
// Port of the types in blip25-vocoder src/engine/mod.rs.
#pragma once

#include <array>

#include "b25/rs.h"

namespace b25::eng {

constexpr usz FRAME_SAMPLES = 160;  // 20 ms of 8 kHz PCM
constexpr usz HALF = 80;            // samples per DSP call: a frame is two

enum class Mode : int {
  FullRate = 0,  // 88 payload bits per frame (P25 Phase 1 / IMBE)
  HalfRate = 1,  // 49 payload bits per frame (AMBE+2: P25 Phase 2, DMR, NXDN)
};

constexpr usz payload_bits(Mode m) { return m == Mode::FullRate ? 88 : 49; }

// What the frame decoder made of the payload it was given.
struct Frame {
  enum class Kind { Speech, Silence, Tone, Repeat, Other };
  Kind kind = Kind::Silence;
  i16 code = 0;  // raw disposition word
  static Frame from_code(i16 c);
};

// Transport-layer status handed to the decoder alongside the payload. The
// decoder conceals with these: error_rate is live over 0..=15 (clamped), `lost`
// holds and fades the previous frame, `mute` forces silence and wins over lost.
struct FrameStatus {
  u8 error_rate = 0;
  bool lost = false;
  bool mute = false;

  static FrameStatus clean() { return {}; }
  static FrameStatus lost_frame() { return {15, true, false}; }
  static FrameStatus make(u32 raw_error_rate, bool mute);
  FrameStatus with_lost(bool v) const { auto s = *this; s.lost = v; return s; }
  FrameStatus with_mute(bool v) const { auto s = *this; s.mute = v; return s; }
  std::array<i16, 26> to_ctrl() const;
};

constexpr usz DEC_STATE_WORDS = 0x400;
constexpr usz BITS_WORDS = 256;

class Decoder {
 public:
  explicit Decoder(Mode mode);
  Mode mode() const { return mode_; }
  usz payload_bits() const { return eng::payload_bits(mode_); }
  void reset();

  // payload: one payload bit per i16 (0/1); pcm: at least FRAME_SAMPLES.
  Frame decode(CS16 payload, S16 pcm) { return decode_with_status(payload, FrameStatus::clean(), pcm); }
  Frame decode_with_status(CS16 payload, FrameStatus status, S16 pcm);

  // Test hook: the raw state words (used to localise divergences).
  CS16 state_words() const { return CS16(state_); }

 private:
  std::array<i16, DEC_STATE_WORDS> state_{};
  std::array<i16, BITS_WORDS> bits_{};
  Mode mode_;
};

constexpr usz ENC_STATE_WORDS = 0xb80;

class Encoder {
 public:
  explicit Encoder(Mode mode);
  Mode mode() const { return mode_; }
  usz payload_bits() const { return eng::payload_bits(mode_); }
  void reset();

  // pcm: FRAME_SAMPLES samples; payload: at least payload_bits() entries, one bit per i16.
  // Output lags the input by one frame of encoder delay.
  Frame encode(CS16 pcm, S16 payload);

  CS16 state_words() const { return CS16(state_); }

 private:
  std::array<i16, ENC_STATE_WORDS> state_{};
  std::array<i16, BITS_WORDS> bits_{};
  Mode mode_;
};

}  // namespace b25::eng
