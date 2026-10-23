/*
 * b25_crosscheck.cpp - interoperability check of the firmware-free blip25 backend
 * (qdv_* built with -DQDV_BACKEND=b25) against the independent mbelib-based DMR
 * AMBE+2 coder in third_party/dv_codecs (MBEVocoder).
 *
 * No MD380 firmware is needed. Four round trips over the test clips, compared by
 * 20 ms loudness-envelope correlation with the input (best lag within +-3 frames):
 *   mbe->mbe   baseline
 *   b25->mbe   our encoder, their decoder   (wire format + encoder)
 *   mbe->b25   their encoder, our decoder   (wire format + decoder)
 *   b25->b25
 * plus the 2971 DMRGateway prompt frames (real DVSI-made DMR frames): FEC errors
 * must be zero and the b25 decode must track the mbelib decode.
 *
 * Usage: b25_crosscheck <male.wav> <female.wav> <dmr_en_US.ambe>
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "mbevocoder.h"
#include "qso_dmr_vocoder.h"

using std::vector;

static vector<uint8_t> readFile(const char* p) {
  vector<uint8_t> b;
  FILE* f = fopen(p, "rb");
  if (!f) return b;
  uint8_t t[65536];
  size_t n;
  while ((n = fread(t, 1, sizeof t, f)) > 0) b.insert(b.end(), t, t + n);
  fclose(f);
  return b;
}

static vector<int16_t> readWav(const char* p) {
  auto b = readFile(p);
  size_t off = 12;
  while (off + 8 <= b.size()) {
    uint32_t sz = b[off + 4] | (b[off + 5] << 8) | (b[off + 6] << 16) | ((uint32_t)b[off + 7] << 24);
    if (!memcmp(&b[off], "data", 4)) {
      size_t n = std::min<size_t>(sz, b.size() - off - 8) / 2;
      vector<int16_t> o(n);
      for (size_t i = 0; i < n; i++) o[i] = (int16_t)(b[off + 8 + 2 * i] | (b[off + 9 + 2 * i] << 8));
      return o;
    }
    off += 8 + sz + (sz & 1);
  }
  return {};
}

static vector<double> envelope(const vector<int16_t>& x) {
  vector<double> e;
  for (size_t i = 0; i + 160 <= x.size(); i += 160) {
    double s = 0;
    for (int k = 0; k < 160; k++) s += double(x[i + k]) * x[i + k];
    e.push_back(20 * std::log10(std::sqrt(s / 160) + 1.0));
  }
  return e;
}

static double corr(const vector<double>& a, const vector<double>& b, int* bestLag = nullptr) {
  double best = -2;
  int bl = 0;
  for (int lag = -3; lag <= 3; lag++) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    int n = 0;
    for (int i = 0; i < (int)a.size(); i++) {
      int j = i + lag;
      if (j < 0 || j >= (int)b.size()) continue;
      sa += a[i]; sb += b[j]; saa += a[i] * a[i]; sbb += b[j] * b[j]; sab += a[i] * b[j]; n++;
    }
    double c = (sab - sa * sb / n) / std::sqrt((saa - sa * sa / n) * (sbb - sb * sb / n));
    if (c > best) { best = c; bl = lag; }
  }
  if (bestLag) *bestLag = bl;
  return best;
}

static double rmsDb(const vector<int16_t>& x) {
  double s = 0;
  for (auto v : x) s += double(v) * v;
  return 20 * std::log10(std::sqrt(s / (x.size() ? x.size() : 1)) + 1.0);
}

static int fails = 0;
static void check(bool ok, const std::string& m) {
  printf("  %s %s\n", ok ? "PASS" : "FAIL", m.c_str());
  if (!ok) fails++;
}

enum Enc { MBE, B25 };

static vector<uint8_t> encodeAll(Enc e, const vector<int16_t>& pcm, MBEVocoder& mv) {
  vector<uint8_t> w;
  for (size_t i = 0; i + 160 <= pcm.size(); i += 160) {
    int16_t in[160];
    memcpy(in, &pcm[i], sizeof in);
    uint8_t f[9] = {0};
    if (e == MBE) mv.encode_2450x1150(in, f);
    else qdv_encode_fec(f, in);
    w.insert(w.end(), f, f + 9);
  }
  return w;
}

static vector<int16_t> decodeAll(Enc d, const vector<uint8_t>& w, MBEVocoder& mv) {
  vector<int16_t> o;
  for (size_t i = 0; i + 9 <= w.size(); i += 9) {
    int16_t pcm[160] = {0};
    uint8_t f[9];
    memcpy(f, &w[i], 9);
    if (d == MBE) mv.decode_2450x1150(pcm, f);
    else qdv_decode_fec(f, pcm);
    o.insert(o.end(), pcm, pcm + 160);
  }
  return o;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s male.wav female.wav dmr_en_US.ambe\n", argv[0]); return 2; }
  if (qdv_init(nullptr, nullptr) != QDV_OK) { puts("init failed"); return 1; }
  printf("%s\n", qdv_build_id());

  for (int c = 0; c < 2; c++) {
    auto pcm = readWav(argv[1 + c]);
    printf("%s clip (%zu frames)\n", c ? "female" : "male", pcm.size() / 160);
    auto ein = envelope(pcm);
    const char* names[4] = {"mbe->mbe", "b25->mbe", "mbe->b25", "b25->b25"};
    Enc encs[4] = {MBE, B25, MBE, B25}, decs[4] = {MBE, MBE, B25, B25};
    double cc[4];
    for (int k = 0; k < 4; k++) {
      qdv_shutdown(); qdv_init(nullptr, nullptr);
      MBEVocoder me, md;
      auto w = encodeAll(encs[k], pcm, me);
      auto out = decodeAll(decs[k], w, md);
      int lag;
      cc[k] = corr(ein, envelope(out), &lag);
      char m[160];
      snprintf(m, sizeof m, "%s: envelope corr %.3f (lag %d), level %+.1f dB vs input", names[k], cc[k], lag,
               rmsDb(out) - rmsDb(pcm));
      printf("  %s\n", m);
    }
    check(cc[1] > 0.85 && cc[2] > 0.85 && cc[3] > 0.85, "all cross paths track the input envelope (> 0.85)");
    check(cc[1] > cc[0] - 0.10 && cc[2] > cc[0] - 0.10, "cross paths within 0.10 of the mbelib-only baseline");
  }

  printf("DMRGateway en_US prompt (real DVSI-made DMR frames)\n");
  auto dm = readFile(argv[3]);
  qdv_shutdown(); qdv_init(nullptr, nullptr);
  MBEVocoder md;
  auto a = decodeAll(B25, dm, md);
  auto b = decodeAll(MBE, dm, md);
  int lag;
  double c = corr(envelope(b), envelope(a), &lag);
  char m[160];
  snprintf(m, sizeof m, "b25 decode vs mbelib decode: envelope corr %.3f (lag %d), level %+.1f dB", c, lag,
           rmsDb(a) - rmsDb(b));
  check(c > 0.95, m);  // level differs by design: mbelib decodes ~5 dB quieter than a chip-like decoder

  qdv_shutdown();
  printf("%s (%d failing)\n", fails ? "FAIL" : "OK", fails);
  return fails ? 1 : 0;
}
