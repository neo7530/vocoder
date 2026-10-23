# Firmware-free backend (QDV_BACKEND=b25): Stand, Bemerkungen, Testplan

Stand: 2026-10-23, Branch `claude/blip25-vocoder-analysis-lptbg9`.

## Was es ist
`qdv_*`-ABI (unveraendert, ABI 2) ueber den portablen C++-Codec in `blip25_cpp/`
(Port von blip25-vocoder, MIT) statt MD380-Firmware unter dynarmic.
Bauen: `cmake -S . -B build -DQDV_BACKEND=b25 && cmake --build build`.
Standard bleibt `dynarmic`. `qdv_init(NULL, NULL)` klappt immer, keine Firmware noetig.

## Was geprueft ist (ohne Firmware)
- blip25_cpp: 10 800 Frames bitgenau gegen das Rust-Original (Payload, PCM, alle
  Zustandsworte), 219 Checks im Golden-Corpus (alle 4 Raten, Concealment, Soft).
- Wire-Format: alle 2971 Frames aus `test/refframes/dmr_en_US_dmrgateway.ambe`
  dekodieren mit 0 Golay-Fehlern -> es ist das echte DMR-Format.
- `b25_crosscheck`: Interoperabilitaet mit dem mbelib-Coder in beide Richtungen,
  Huellkurven-Korrelation 0,94-0,98 (mbelib-Basis 0,91-0,97).
- Pegel Round-Trip +0,1 dB (m) / +0,7 dB (w); mbelib dekodiert ca. 5 dB leiser.
- ca. 0,2 ms CPU je 20-ms-Frame (Encode+Decode, x86-64); nur `qdv_*` exportiert.

## NICHT geprueft / offen
1. **Kein Vergleich mit der MD380-Firmware** (nicht redistributierbar). Es ist unbekannt,
   ob die Ausgabe sample-genau oder auch nur im Pegel gleich ist. Das Corpus-Ergebnis
   beweist nur Gleichheit mit dem Rust-Crate, nicht mit einer Hardware-Referenz.
2. Unterschiede zum alten Modul: `qdv_unwrap_md380` -> `QDV_ERR_UNSUPPORTED` (12, neu);
   `qdv_decode_fec` korrigiert Golay-Fehler (Firmware nicht); `qdv_decode` liest 49 Bits
   (Firmware 48).
3. MSVC: Kern setzt `-fwrapv` voraus, MSVC kennt es nicht (Windows-Build + Corpus-Test offen).
4. Android-Builds, `dart:ffi`-Aufruf, Helper-Prozess (`helper_main.cpp`) mit b25 nicht ausprobiert.
5. Die `dynarmic`-Variante wurde nach dem CMake-Umbau nicht neu gebaut.
6. Patente: AMBE+2 (US8359197 bis 2028) betrifft auch diese Variante; MIT gibt keine
   Patentrechte. Siehe `blip25_cpp/PATENT_NOTICE.md`.
7. Nicht portiert: Enhancement-Filter, Transcoder, Stream-Wrapper, SpectrumProbe.
   Der Soft-Decoder entscheidet Gleichstand nach Index (Rust: instabile Sortierung).

## Testplan (am PC, mit Firmware)
1. Beide Varianten bauen: `-DQDV_BACKEND=dynarmic` (Firmware) und `-DQDV_BACKEND=b25`.
2. `qdv_test <D002.032.bin> <d02032-core.img> test/clips/male_harvard_8k.wav out/` mit
   beiden Varianten; Pegel, Huellkurve, Latenz vergleichen.
3. Gleiche PCM-Datei durch `qdv_encode_fec` beider Varianten, die 9-Byte-Frames vergleichen
   (Anteil identischer Frames, Bitdifferenzen), dann jeweils kreuzweise dekodieren.
4. Pegel/Pitch-Unterschied messen; falls die App auf Firmware-Pegel abgestimmt ist,
   Verstaerkung im Shim (`src/qdv_b25.cpp`) anpassen.
5. `build/b25_crosscheck test/clips/male_harvard_8k.wav test/clips/female_harvard_8k.wav test/refframes/dmr_en_US_dmrgateway.ambe`
6. Windows (MSVC) und Android bauen; Corpus-Test (`blip25_cpp/test/golden_corpus_test`) dort laufen lassen.
7. Echtes Funk-/Gateway-Material (DMR) durch Decoder, auf Artefakte hoeren.
