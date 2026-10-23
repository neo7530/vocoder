// End-to-end check against the Rust crate's public regression corpus: encode every
// source at all four rates, decode the stored wire frames at both codecs, and
// replay the concealment severity ramp. Usage: golden_corpus_test <corpus file>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "b25/vocoder.h"
#include "b25/wire.h"

using namespace b25;

namespace {

struct Reader {
  const std::vector<u8>& b;
  usz at = 0;
  u16 u16_() { u16 v = u16(b.at(at) | (b.at(at + 1) << 8)); at += 2; return v; }
  u32 u32_() { u32 v = 0; for (int i = 0; i < 4; ++i) v |= u32(b.at(at + i)) << (8 * i); at += 4; return v; }
  std::vector<u8> bytes(usz n) { if (at + n > b.size()) throw std::runtime_error("truncated"); std::vector<u8> v(b.begin() + at, b.begin() + at + n); at += n; return v; }
  std::vector<i16> pcm(usz n) { auto r = bytes(n * 2); std::vector<i16> v(n); for (usz i = 0; i < n; ++i) v[i] = i16(r[2 * i] | (r[2 * i + 1] << 8)); return v; }
  std::string name() { auto r = bytes(u32_()); return {r.begin(), r.end()}; }
};

struct Source {
  std::string name;
  std::vector<i16> pcm;
  std::vector<u8> wire[4];
  std::vector<i16> decoded[2];
};

constexpr Rate RATES[4] = {Rate::FullRate7200x4400, Rate::FullRate4400x4400, Rate::HalfRate3600x2450,
                           Rate::HalfRate2450x2450};
const char* RATE_NAMES[4] = {"7200x4400", "4400x4400", "3600x2450", "2450x2450"};

int failures = 0;
void fail(const std::string& m) { std::printf("FAIL %s\n", m.c_str()); ++failures; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s corpus\n", argv[0]); return 2; }
  std::ifstream f(argv[1], std::ios::binary);
  std::vector<u8> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.size() < 8 || std::memcmp(data.data(), "B25CORP\x01", 8) != 0) { std::fprintf(stderr, "bad magic\n"); return 2; }
  Reader r{data, 8};
  r.bytes(r.u32_());  // credit
  usz n = r.u32_();
  std::vector<Source> sources(n);
  for (auto& s : sources) {
    s.name = r.name();
    usz frames = r.u32_();
    s.pcm = r.pcm(frames * FRAME_SAMPLES);
    for (auto& w : s.wire) { r.u16_(); w = r.bytes(r.u32_()); }
    for (auto& d : s.decoded) d = r.pcm(r.u32_());
  }
  auto conceal = r.pcm(r.u32_());
  if (r.at != data.size()) { std::fprintf(stderr, "trailing bytes\n"); return 2; }

  usz checks = 0;
  for (auto& s : sources) {
    for (int ri = 0; ri < 4; ++ri) {  // encoder
      Vocoder tx(RATES[ri]);
      std::vector<u8> got, w;
      for (usz o = 0; o < s.pcm.size(); o += FRAME_SAMPLES) {
        if (tx.encode_pcm(CS16(s.pcm).subspan(o, FRAME_SAMPLES), w) != Error::Ok) { fail("encode error"); return 1; }
        got.insert(got.end(), w.begin(), w.end());
      }
      ++checks;
      if (got != s.wire[ri]) {
        usz at = 0;
        while (at < got.size() && at < s.wire[ri].size() && got[at] == s.wire[ri][at]) ++at;
        fail(s.name + " @ " + RATE_NAMES[ri] + ": encoder output differs at byte " + std::to_string(at));
      }
    }
    for (int ci = 0; ci < 2; ++ci) {  // decoder
      int ri = ci == 0 ? 0 : 2;
      Vocoder rx(RATES[ri]);
      usz fb = fec_frame_bytes(RATES[ri]);
      std::vector<i16> got, pcm;
      for (usz o = 0; o + fb <= s.wire[ri].size(); o += fb) {
        if (rx.decode_bits(s.wire[ri].data() + o, fb, pcm) != Error::Ok) { fail("decode error"); return 1; }
        got.insert(got.end(), pcm.begin(), pcm.end());
      }
      ++checks;
      if (got != s.decoded[ci]) {
        usz at = 0;
        while (at < got.size() && at < s.decoded[ci].size() && got[at] == s.decoded[ci][at]) ++at;
        fail(s.name + " @ " + RATE_NAMES[ri] + ": decoder output differs at sample " + std::to_string(at));
      }
    }
  }

  {  // concealment severity ramp
    auto& probe = sources.at(0);
    Vocoder tx(Rate::HalfRate3600x2450), rx(Rate::HalfRate3600x2450);
    std::vector<i16> got, pcm;
    std::vector<u8> bits;
    for (usz i = 0, o = 0; o + FRAME_SAMPLES <= probe.pcm.size(); o += FRAME_SAMPLES, ++i) {
      tx.encode_pcm(CS16(probe.pcm).subspan(o, FRAME_SAMPLES), bits);
      auto info = wire::half::decode_frame(wire::unpack_dibits<36>(bits.data())).info;
      auto st = i % 17 == 16 ? eng::FrameStatus::lost_frame() : eng::FrameStatus::make(u32(i % 16), false);
      rx.decode_info(info.data(), 4, st, pcm);
      got.insert(got.end(), pcm.begin(), pcm.end());
    }
    ++checks;
    if (got != conceal) fail("concealment ramp differs");
  }

  // Soft decoding of clean LLRs must equal hard decoding.
  {
    auto& s = sources.at(0);
    for (int ri : {0, 2}) {
      usz fb = fec_frame_bytes(RATES[ri]);
      Vocoder hard(RATES[ri]), soft(RATES[ri]);
      std::vector<i16> a, b;
      for (usz o = 0; o + fb <= s.wire[ri].size() && o < 40 * fb; o += fb) {
        const u8* p = s.wire[ri].data() + o;
        hard.decode_bits(p, fb, a);
        std::vector<i8> llr(soft_frame_bits(RATES[ri]));
        for (usz j = 0; j < llr.size(); ++j) llr[j] = ((p[j / 8] >> (7 - j % 8)) & 1) ? 100 : -100;
        soft.decode_soft(llr.data(), llr.size(), b);
        if (a != b) { fail(std::string("soft != hard @ ") + RATE_NAMES[ri]); break; }
      }
      ++checks;
    }
  }

  std::printf("%zu sources, %zu checks, %d failing\n", sources.size(), checks, failures);
  return failures ? 1 : 0;
}
