// Encoder-side parameter blocks (non-owning views over the state / tables).
struct ToneTables {
  CS16 class_;  // `class` in the Rust source
  CS16 lin_lo_a, bin_lo_a, lin_lo_b, bin_lo_b;
  CS16 lin_hi_a, bin_hi_a, lin_hi_b, bin_hi_b;
};

struct SlotCtx {
  CS16 bands, mags, pitch_cand, pitch_err;
  i16 pitch_target = 0;
  CS16 energies, energy_exp;
  i32 active_mask = 0;
};

struct SlotContext {
  u32 spec_off = 0, exps_off = 0, sink_a = 0, sink_b = 0;
  u16 sink_arg0 = 0;
  i16 sink_enable = 0;
  u32 sink_c = 0, sink_d = 0, sink_e = 0, sink_fn = 0;
  i16 level_bias = 0;
  u16 sink_arg9 = 0;
};

// The call the reference makes into its host sink; recorded, never executed.
struct SinkCall {
  u32 args[10] = {};
};

struct FrameCtx {
  SlotCtx slot;
  SlotContext amp;
  CS16 env, spec, exps;
};

struct PackTables {
  CS16 blob, partitions;
  std::span<const u32> voicing_book;
  CS16 nibbles, gain64, gain32, hoc_weight, hoc_loading, block_lens, prba9, prba7, coef_books;
};

struct FrontEndTables {
  CS16 hamming, wide_win, taper, interp, subband_win, cos, fb_win, fb_tbl;
  ToneTables tone;
};

struct Bases {
  u32 state = 0;
  u32 sink = 0;
};

namespace disposition {
constexpr u16 TONE = 0x8000;
constexpr u16 REARMED = 0x0400;
constexpr u16 ACTIVE = 0x0002;
}  // namespace disposition

