cmake_minimum_required(VERSION 3.20)
project(blip25_cpp CXX)

# Portable C++20 port of the blip25 vocoder engine (see README.md).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(b25 STATIC
  src/engine_tables.cpp
  src/dsp_tables.cpp
  src/decode_tables.cpp
  src/encode_tables.cpp
  src/dsp.cpp
  src/decode.cpp
  src/encode.cpp
  src/engine.cpp
  src/fec.cpp
  src/wire_tables.cpp
  src/wire.cpp
  src/vocoder.cpp
)
target_include_directories(b25 PUBLIC include)
# The engine relies on two's-complement wrapping for plain + - * on signed
# types in a few places (the Rust original wraps in release builds).
if(MSVC)
  target_compile_options(b25 PRIVATE /W4)
else()
  target_compile_options(b25 PRIVATE -fwrapv -Wall -Wextra)
endif()

option(B25_BOUNDS_CHECK "bounds-checked spans (libstdc++ assertions) in tests" ON)

add_executable(ref_dump_test test/ref_dump_test.cpp)
target_link_libraries(ref_dump_test PRIVATE b25)
add_executable(golden_corpus_test test/golden_corpus_test.cpp)
target_link_libraries(golden_corpus_test PRIVATE b25)
if(B25_BOUNDS_CHECK AND NOT MSVC)
  target_compile_definitions(b25 PRIVATE _GLIBCXX_ASSERTIONS)
  target_compile_definitions(ref_dump_test PRIVATE _GLIBCXX_ASSERTIONS)
  target_compile_definitions(golden_corpus_test PRIVATE _GLIBCXX_ASSERTIONS)
endif()
