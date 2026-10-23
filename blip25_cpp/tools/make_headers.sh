#!/bin/sh
# Regenerate the prototype headers from the .cpp files (run from blip25_cpp/).
set -e
python3 tools/gen_header.py src/dsp.cpp include/b25/dsp.h b25::eng --prelude tools/dsp_prelude.h b25/rs.h
python3 tools/gen_header.py src/decode.cpp include/b25/decode.h b25::eng --prelude tools/decode_prelude.h b25/rs.h b25/dsp.h b25/decode_tables.h
python3 tools/gen_header.py src/encode.cpp include/b25/encode.h b25::eng::enc --prelude tools/encode_prelude.h b25/rs.h b25/dsp.h b25/decode.h b25/encode_tables.h
