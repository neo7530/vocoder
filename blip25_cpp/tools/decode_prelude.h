// The per-codec quantiser tables a frame decode reads (non-owning views).
struct FrameTables {
  CS16 submode;
  std::span<const u32> voicing_patterns;
  CS16 gain32;
  CS16 block_lens;
  CS16 prba9;
  CS16 prba7;
  CS16 coef_books;
  CS16 nibbles;
  CS16 gain64;
  CS16 hoc_weight;
  CS16 hoc_loading;
  CSU16 codewords;
  CS16 blob;
};

struct DecTables {
  FrameTables frame;
  CS16 hilbert;
  CS16 sqrt_tbl;
};

