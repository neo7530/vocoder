#include "b25/vocoder.h"

#include <algorithm>

#include "b25/wire.h"

namespace b25 {

namespace {

using Payload = std::array<i16, 88>;

template <usz N>
usz info_to_payload(const std::array<u16, N>& info, const std::array<u8, N>& widths, Payload& out) {
  usz at = 0;
  for (usz i = 0; i < N; ++i)
    for (int k = widths[i] - 1; k >= 0; --k) out[at++] = i16((info[i] >> k) & 1);
  return at;
}

template <usz N>
std::array<u16, N> payload_to_info(const Payload& p, const std::array<u8, N>& widths) {
  std::array<u16, N> out{};
  usz at = 0;
  for (usz i = 0; i < N; ++i) {
    u16 v = 0;
    for (int k = 0; k < widths[i]; ++k) v = u16((v << 1) | (p[at++] & 1));
    out[i] = v;
  }
  return out;
}

u8 saturate(u16 total) { return u8(std::min<u16>(total, 255)); }

using IMBE_INFO = std::array<u16, 8>;
using AMBE_INFO = std::array<u16, 4>;

std::vector<u8> imbe_info_to_wire(Rate r, const IMBE_INFO& info) {
  if (r == Rate::FullRate7200x4400) {
    auto b = wire::pack_dibits<72, 18>(wire::full::encode_frame(info));
    return {b.begin(), b.end()};
  }
  std::vector<u8> out(11, 0);
  wire::pack_info_bits(info, wire::full::INFO_WIDTHS, out.data());
  return out;
}

std::vector<u8> ambe_info_to_wire(Rate r, const AMBE_INFO& info) {
  if (r == Rate::HalfRate3600x2450) {
    auto b = wire::pack_dibits<36, 9>(wire::half::encode_frame(info));
    return {b.begin(), b.end()};
  }
  auto b = wire::half::pack_natural(info);
  return {b.begin(), b.end()};
}

AMBE_INFO ambe_erasure_info() {
  std::array<u16, wire::half::PARAM_COUNT> b{};
  b[0] = 127;
  return wire::half::prioritize(b);
}

constexpr IMBE_INFO IMBE_ERASURE_INFO = {0xFFF, 0, 0, 0, 0, 0, 0, 0};

}  // namespace

const char* error_string(Error e) {
  switch (e) {
    case Error::Ok: return "ok";
    case Error::WrongPcmLength: return "wrong PCM length (need exactly 160 samples)";
    case Error::WrongBitsLength: return "wrong wire frame length for this rate";
    case Error::WrongInfoLength: return "wrong number of info vectors for this rate";
    case Error::WrongSoftLength: return "wrong number of soft bits for this rate";
    case Error::SoftUnsupported: return "this rate has no FEC, so no soft decoding";
  }
  return "?";
}

Vocoder::Vocoder(Rate rate) : rate_(rate), enc_(engine_mode(rate)), dec_(engine_mode(rate)) {}

void Vocoder::reset() {
  enc_.reset();
  dec_.reset();
  stats_ = {};
  enc_disposition_ = 0;
}

Error Vocoder::encode_info(CS16 pcm, std::vector<u16>& info) {
  if (pcm.size() != FRAME_SAMPLES) return Error::WrongPcmLength;
  Payload payload{};
  auto frame = enc_.encode(pcm, S16(payload).first(enc_.payload_bits()));
  enc_disposition_ = frame.code;
  if (engine_mode(rate_) == eng::Mode::HalfRate) {
    auto v = payload_to_info(payload, wire::half::INFO_WIDTHS);
    info.assign(v.begin(), v.end());
  } else {
    auto v = payload_to_info(payload, wire::full::INFO_WIDTHS);
    info.assign(v.begin(), v.end());
  }
  return Error::Ok;
}

Error Vocoder::encode_pcm(CS16 pcm, std::vector<u8>& out) {
  std::vector<u16> info;
  if (Error e = encode_info(pcm, info); e != Error::Ok) return e;
  if (engine_mode(rate_) == eng::Mode::HalfRate) {
    AMBE_INFO v;
    std::copy(info.begin(), info.end(), v.begin());
    out = ambe_info_to_wire(rate_, v);
  } else {
    IMBE_INFO v;
    std::copy(info.begin(), info.end(), v.begin());
    out = imbe_info_to_wire(rate_, v);
  }
  return Error::Ok;
}

void Vocoder::synth(const Payload& payload, usz n, eng::FrameStatus st, u8 e0, u8 et, std::vector<i16>& pcm) {
  pcm.assign(FRAME_SAMPLES, 0);
  auto frame = dec_.decode_with_status(CS16(payload).first(n), st, pcm);
  stats_ = {e0, et, frame.code};
}

Error Vocoder::decode_inner(const u8* bits, usz len, const eng::FrameStatus* upstream, std::vector<i16>& pcm) {
  if (len != fec_frame_bytes(rate_)) return Error::WrongBitsLength;
  Payload payload{};
  usz n = 0;
  u8 e0 = 0, et = 0;
  switch (rate_) {
    case Rate::HalfRate3600x2450: {
      auto f = wire::half::decode_frame(wire::unpack_dibits<36>(bits));
      n = info_to_payload(f.info, wire::half::INFO_WIDTHS, payload);
      e0 = f.errors[0];
      et = saturate(f.error_total());
      break;
    }
    case Rate::HalfRate2450x2450:
      n = info_to_payload(wire::half::unpack_natural(bits), wire::half::INFO_WIDTHS, payload);
      break;
    case Rate::FullRate7200x4400: {
      auto f = wire::full::decode_frame(wire::unpack_dibits<72>(bits));
      n = info_to_payload(f.info, wire::full::INFO_WIDTHS, payload);
      e0 = f.errors[0];
      et = saturate(f.error_total());
      break;
    }
    case Rate::FullRate4400x4400:
      n = info_to_payload(wire::unpack_info_bits<8>(bits, wire::full::INFO_WIDTHS), wire::full::INFO_WIDTHS, payload);
      break;
  }
  eng::FrameStatus st = upstream ? *upstream
                        : e0 == 255 ? eng::FrameStatus::lost_frame()
                                    : eng::FrameStatus::make(et, false);
  synth(payload, n, st, e0, et, pcm);
  return Error::Ok;
}

Error Vocoder::decode_bits(const u8* bits, usz len, std::vector<i16>& pcm) {
  return decode_inner(bits, len, nullptr, pcm);
}

Error Vocoder::decode_bits_with_status(const u8* bits, usz len, eng::FrameStatus status, std::vector<i16>& pcm) {
  return decode_inner(bits, len, &status, pcm);
}

Error Vocoder::decode_info(const u16* info, usz len, eng::FrameStatus status, std::vector<i16>& pcm) {
  if (len != info_vector_count(rate_)) return Error::WrongInfoLength;
  Payload payload{};
  usz n;
  if (engine_mode(rate_) == eng::Mode::HalfRate) {
    AMBE_INFO v;
    std::copy(info, info + 4, v.begin());
    n = info_to_payload(v, wire::half::INFO_WIDTHS, payload);
  } else {
    IMBE_INFO v;
    std::copy(info, info + 8, v.begin());
    n = info_to_payload(v, wire::full::INFO_WIDTHS, payload);
  }
  synth(payload, n, status, 0, 0, pcm);
  return Error::Ok;
}

Error Vocoder::decode_soft(const i8* llrs, usz len, std::vector<i16>& pcm) {
  usz expected = soft_frame_bits(rate_);
  if (expected == 0) return Error::SoftUnsupported;
  if (len != expected) return Error::WrongSoftLength;
  // Soft-decode the FEC, re-encode a clean frame and run the hard path on it
  // (same as the Rust crate), so concealment and state handling are shared.
  std::vector<u8> fec;
  if (rate_ == Rate::FullRate7200x4400) {
    std::array<i8, 144> s;
    std::copy(llrs, llrs + 144, s.begin());
    fec = imbe_info_to_wire(rate_, wire::full::decode_frame_soft(s).info);
  } else {
    std::array<i8, 72> s;
    std::copy(llrs, llrs + 72, s.begin());
    fec = ambe_info_to_wire(rate_, wire::half::decode_frame_soft(s).info);
  }
  return decode_bits(fec.data(), fec.size(), pcm);
}

std::vector<u8> Vocoder::erasure_frame() const {
  switch (rate_) {
    case Rate::HalfRate3600x2450:
    case Rate::HalfRate2450x2450: return ambe_info_to_wire(rate_, ambe_erasure_info());
    default: return imbe_info_to_wire(rate_, IMBE_ERASURE_INFO);
  }
}

}  // namespace b25
