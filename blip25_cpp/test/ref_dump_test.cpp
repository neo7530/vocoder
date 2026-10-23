// Replays the engine-level reference dump produced from the Rust crate
// (src/engine/dump_ref.rs) through the C++ port and reports the first
// divergence: frame, field, and - for state divergence - the state word.
//
//   ref_dump_test <ref.b25dump> [--encoder]
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "b25/engine.h"

using namespace b25;
using namespace b25::eng;

constexpr size_t ENC_STATE_WORDS_REF = 0xb80;

struct Reader {
  std::vector<uint8_t> b;
  size_t at = 0;
  template <class T> T get() {
    T v;
    std::memcpy(&v, &b[at], sizeof v);
    at += sizeof v;
    return v;
  }
  void get_i16s(i16* d, size_t n) { std::memcpy(d, &b[at], n * 2); at += n * 2; }
  void skip(size_t n) { at += n; }
};

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <ref.b25dump>\n", argv[0]); return 2; }
  Reader r;
  {
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::perror("open"); return 2; }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    r.b.resize(static_cast<size_t>(n));
    if (std::fread(r.b.data(), 1, r.b.size(), f) != r.b.size()) return 2;
    std::fclose(f);
  }
  if (std::memcmp(r.b.data(), "B25DUMP1", 8) != 0) { std::fprintf(stderr, "bad magic\n"); return 2; }
  r.at = 8;
  uint32_t nsrc = r.get<uint32_t>();
  size_t frames = 0, bad = 0, bad_pcm = 0, bad_state = 0, bad_code = 0, bad_enc = 0;
  const size_t max_report = 8;
  for (uint32_t s = 0; s < nsrc; s++) {
    uint32_t nl = r.get<uint32_t>();
    std::string name(reinterpret_cast<const char*>(&r.b[r.at]), nl);
    r.skip(nl);
    uint32_t nf = r.get<uint32_t>();
    std::vector<i16> in_pcm(static_cast<size_t>(nf) * FRAME_SAMPLES);
    r.get_i16s(in_pcm.data(), in_pcm.size());
    for (int m = 0; m < 2; m++) {
      Mode mode = m == 0 ? Mode::FullRate : Mode::HalfRate;
      uint32_t nb = r.get<uint32_t>();
      Decoder clean(mode), stat(mode);
      Encoder enc(mode);
      std::vector<i16> enc_payload(nb), enc_state_ref(ENC_STATE_WORDS_REF);
      std::vector<i16> payload(nb), noisy(nb), pcm_ref(FRAME_SAMPLES), pcm(FRAME_SAMPLES), st_ref(DEC_STATE_WORDS);
      for (uint32_t f = 0; f < nf; f++) {
        i16 ecode_ref = r.get<i16>();
        r.get_i16s(payload.data(), nb);
        r.get_i16s(enc_state_ref.data(), ENC_STATE_WORDS_REF);
        {
          // encoder: same input, must reproduce payload bits, disposition and all state words
          Frame ef = enc.encode(CS16(in_pcm).subspan(static_cast<size_t>(f) * FRAME_SAMPLES, FRAME_SAMPLES), S16(enc_payload));
          frames++;
          bool ok = ef.code == ecode_ref || static_cast<u16>(ef.code) == static_cast<u16>(ecode_ref);
          size_t first_bit = nb;
          for (size_t i = 0; i < nb; i++) if (enc_payload[i] != payload[i]) { first_bit = i; break; }
          size_t first_st = ENC_STATE_WORDS_REF;
          auto sw = enc.state_words();
          for (size_t i = 0; i < ENC_STATE_WORDS_REF; i++) if (sw[i] != enc_state_ref[i]) { first_st = i; break; }
          if (!ok || first_bit != nb || first_st != ENC_STATE_WORDS_REF) {
            bad_enc++;
            if (bad++ < max_report) {
              std::printf("MISMATCH encode src=%s mode=%d frame=%u: code %d vs %d, first payload bit diff @%zu, first state diff @0x%zx", name.c_str(), m, f,
                          ef.code, ecode_ref, first_bit, first_st);
              if (first_st != ENC_STATE_WORDS_REF) std::printf(" (cpp %d, ref %d)", sw[first_st], enc_state_ref[first_st]);
              std::printf("\n");
            }
          }
        }
        i16 code_ref = r.get<i16>();
        r.get_i16s(pcm_ref.data(), FRAME_SAMPLES);
        r.get_i16s(st_ref.data(), DEC_STATE_WORDS);
        Frame fr = clean.decode(CS16(payload), S16(pcm));
        auto check = [&](const char* what, i16 code_ref_, Frame fr_, const std::vector<i16>& pr, const Decoder& d,
                         const std::vector<i16>& sr) {
          frames++;
          bool ok = true;
          if (fr_.code != code_ref_) { ok = false; bad_code++; }
          size_t first_pcm = FRAME_SAMPLES;
          for (size_t i = 0; i < FRAME_SAMPLES; i++) if (pcm[i] != pr[i]) { first_pcm = i; break; }
          if (first_pcm != FRAME_SAMPLES) { ok = false; bad_pcm++; }
          size_t first_st = DEC_STATE_WORDS;
          auto sw = d.state_words();
          for (size_t i = 0; i < DEC_STATE_WORDS; i++) if (sw[i] != sr[i]) { first_st = i; break; }
          if (first_st != DEC_STATE_WORDS) { ok = false; bad_state++; }
          if (!ok) {
            if (bad++ < max_report) {
              std::printf("MISMATCH %s src=%s mode=%d frame=%u: code %d vs %d, first pcm diff @%zu, first state diff @0x%zx", what,
                          name.c_str(), m, f, fr_.code, code_ref_, first_pcm, first_st);
              if (first_st != DEC_STATE_WORDS) std::printf(" (cpp %d, ref %d)", sw[first_st], sr[first_st]);
              std::printf("\n");
            }
          }
        };
        check("clean ", code_ref, fr, pcm_ref, clean, st_ref);
        // status-driven pass
        r.get_i16s(noisy.data(), nb);
        FrameStatus st;
        st.error_rate = r.get<uint8_t>();
        st.lost = r.get<uint8_t>() != 0;
        st.mute = r.get<uint8_t>() != 0;
        i16 code2 = r.get<i16>();
        r.get_i16s(pcm_ref.data(), FRAME_SAMPLES);
        r.get_i16s(st_ref.data(), DEC_STATE_WORDS);
        Frame fr2 = stat.decode_with_status(CS16(noisy), st, S16(pcm));
        check("status", code2, fr2, pcm_ref, stat, st_ref);
      }
    }
  }
  std::printf("%zu frame checks: %zu mismatching (decoder: code %zu, pcm %zu, state %zu; encoder %zu)\n", frames, bad, bad_code, bad_pcm, bad_state, bad_enc);
  return bad == 0 ? 0 : 1;
}
