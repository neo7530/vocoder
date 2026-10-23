// Vocoder façade: PCM <-> wire frames at the four P25 rates.
// Port of the `Vocoder` type in blip25-vocoder src/vocoder.rs (enhancement
// chain, transcoder and live/stream helpers are not ported).
#pragma once

#include <vector>

#include "b25/engine.h"
#include "b25/rs.h"

namespace b25 {

enum class Rate : int {
  FullRate7200x4400,  // IMBE, 18-byte frame with Golay/Hamming FEC (P25 Phase 1)
  FullRate4400x4400,  // IMBE, 11-byte info-only frame
  HalfRate3600x2450,  // AMBE+2, 9-byte frame with Golay FEC (P25 Phase 2, DMR, NXDN)
  HalfRate2450x2450,  // AMBE+2, 7-byte info-only frame
};

constexpr usz fec_frame_bytes(Rate r) {
  switch (r) {
    case Rate::FullRate7200x4400: return 18;
    case Rate::FullRate4400x4400: return 11;
    case Rate::HalfRate3600x2450: return 9;
    default: return 7;
  }
}
// Soft-decision bits per frame, 0 for the rates without FEC.
constexpr usz soft_frame_bits(Rate r) {
  return r == Rate::FullRate7200x4400 ? 144 : r == Rate::HalfRate3600x2450 ? 72 : 0;
}
constexpr usz info_vector_count(Rate r) {
  return (r == Rate::HalfRate3600x2450 || r == Rate::HalfRate2450x2450) ? 4 : 8;
}
constexpr eng::Mode engine_mode(Rate r) {
  return (r == Rate::HalfRate3600x2450 || r == Rate::HalfRate2450x2450) ? eng::Mode::HalfRate : eng::Mode::FullRate;
}
constexpr usz FRAME_SAMPLES = eng::FRAME_SAMPLES;

enum class Error {
  Ok = 0,
  WrongPcmLength,
  WrongBitsLength,
  WrongInfoLength,
  WrongSoftLength,
  SoftUnsupported,
};
const char* error_string(Error e);

// What the decoder did with the last frame.
struct DecodeStats {
  u8 epsilon_0 = 0;  // errors corrected in the first (most important) code vector; 255 = unrecoverable
  u8 epsilon_t = 0;  // total corrected errors, saturated
  i16 disposition = 0;
  bool is_speech() const { return disposition & 0x0002; }
  bool is_concealed() const { return disposition & 0x0020; }
  bool is_tone() const { return disposition & i16(-0x8000); }
};

class Vocoder {
 public:
  explicit Vocoder(Rate rate);
  Rate rate() const { return rate_; }
  void reset();

  // Encode exactly FRAME_SAMPLES samples. The output lags the input by one frame.
  Error encode_pcm(CS16 pcm, std::vector<u8>& wire);
  // As above but stops before the wire format: info_vector_count(rate) values.
  Error encode_info(CS16 pcm, std::vector<u16>& info);

  // Decode one wire frame (fec_frame_bytes(rate) bytes) to FRAME_SAMPLES samples.
  Error decode_bits(const u8* bits, usz len, std::vector<i16>& pcm);
  Error decode_bits_with_status(const u8* bits, usz len, eng::FrameStatus status, std::vector<i16>& pcm);
  // Decode from info vectors; `status` carries the transport-layer hint.
  Error decode_info(const u16* info, usz len, eng::FrameStatus status, std::vector<i16>& pcm);
  // Soft decision (-128..127, positive = 1) for the FEC rates only.
  Error decode_soft(const i8* llrs, usz len, std::vector<i16>& pcm);

  // A wire frame the decoder treats as an erasure.
  std::vector<u8> erasure_frame() const;

  const DecodeStats& last_decode() const { return stats_; }
  i16 last_encode_disposition() const { return enc_disposition_; }

 private:
  Error decode_inner(const u8* bits, usz len, const eng::FrameStatus* upstream, std::vector<i16>& pcm);
  void synth(const std::array<i16, 88>& payload, usz n, eng::FrameStatus st, u8 e0, u8 et, std::vector<i16>& pcm);

  Rate rate_;
  eng::Encoder enc_;
  eng::Decoder dec_;
  DecodeStats stats_{};
  i16 enc_disposition_ = 0;
};

}  // namespace b25
