/*
 * qdv_b25.cpp - the qdv_* C ABI over the portable C++ blip25 codec (blip25_cpp/)
 * instead of the MD380 firmware under dynarmic.
 *
 * Same surface and data shapes as qdv_abi.cpp (one 20 ms frame: 160 int16 PCM <->
 * 7-byte 49-bit AMBE+2 MSB-first, or the 9-byte 72-bit DMR on-air frame), but no
 * firmware: qdv_init() takes its two path arguments for compatibility, ignores
 * them (NULL is fine) and cannot fail. qdv_unwrap_md380 is not supported.
 *
 * Notes against the firmware backend:
 *  - the 49-bit frame is the same bit order (a[12] b[12] c[25]); the firmware
 *    path reads only the first 48 bits on decode, this one reads all 49.
 *  - qdv_decode_fec runs the real Golay(24,12)/(23,12) error correction and
 *    feeds the error count to the decoder's concealment; the firmware path
 *    does not correct errors.
 *
 * GPL: part of the qso_dmr_vocoder module (it implements the GPL-side header).
 * The codec it calls (blip25_cpp) is MIT; see blip25_cpp/LICENSE.upstream-MIT and
 * PATENT_NOTICE.md.
 */

#define QDV_BUILDING 1
#include "qso_dmr_vocoder.h"

#include <cstring>
#include <memory>
#include <vector>

#include "b25/vocoder.h"
#include "b25/wire.h"

namespace {
std::unique_ptr<b25::Vocoder> g_voc;

const char kBuildId[] =
    "qso_dmr_vocoder/1.2.0 abi2 (blip25_cpp portable C++ backend, firmware-free; "
    "blip25-vocoder MIT)";
}  // namespace

extern "C" {

int qdv_abi_version(void) { return QDV_ABI_VERSION; }

const char* qdv_build_id(void) { return kBuildId; }

int qdv_unwrap_md380(const char*, const char*) { return QDV_ERR_UNSUPPORTED; }

int qdv_init(const char* /*firmware_path*/, const char* /*core_path*/) {
  if (g_voc) return QDV_ERR_ALREADY_INIT;
  g_voc = std::make_unique<b25::Vocoder>(b25::Rate::HalfRate3600x2450);
  return QDV_OK;
}

void qdv_encode(uint8_t* ambe_out, const int16_t* pcm_in) {
  if (!g_voc || !ambe_out || !pcm_in) return;
  std::vector<b25::u16> info;
  if (g_voc->encode_info(b25::CS16(pcm_in, 160), info) != b25::Error::Ok) return;
  std::array<b25::u16, 4> v;
  std::copy(info.begin(), info.end(), v.begin());
  auto bytes = b25::wire::half::pack_natural(v);
  std::memcpy(ambe_out, bytes.data(), 7);
}

void qdv_decode(const uint8_t* ambe_in, int16_t* pcm_out) {
  if (!g_voc || !ambe_in || !pcm_out) return;
  auto info = b25::wire::half::unpack_natural(ambe_in);
  std::vector<b25::i16> pcm;
  if (g_voc->decode_info(info.data(), 4, b25::eng::FrameStatus::clean(), pcm) != b25::Error::Ok) return;
  std::memcpy(pcm_out, pcm.data(), 160 * sizeof(int16_t));
}

void qdv_encode_fec(uint8_t* ambe_out, const int16_t* pcm_in) {
  if (!g_voc || !ambe_out || !pcm_in) return;
  std::vector<b25::u8> wire;
  if (g_voc->encode_pcm(b25::CS16(pcm_in, 160), wire) != b25::Error::Ok) return;
  std::memcpy(ambe_out, wire.data(), 9);
}

void qdv_decode_fec(const uint8_t* ambe_in, int16_t* pcm_out) {
  if (!g_voc || !ambe_in || !pcm_out) return;
  std::vector<b25::i16> pcm;
  if (g_voc->decode_bits(ambe_in, 9, pcm) != b25::Error::Ok) return;
  std::memcpy(pcm_out, pcm.data(), 160 * sizeof(int16_t));
}

void qdv_shutdown(void) { g_voc.reset(); }

}  // extern "C"
