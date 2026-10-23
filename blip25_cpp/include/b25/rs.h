// Rust-semantics helpers for the blip25 engine port.
//
// The engine is a transliteration of fixed-point reference code and relies on
// Rust's exact integer behaviour: `wrapping_*` arithmetic, `as` casts that
// truncate / sign-extend, arithmetic right shift on signed values. C++20 defines
// the casts and the arithmetic shift; signed overflow is the one trap, so
// everything that can wrap goes through the helpers below (unsigned math), and
// the build also passes -fwrapv as a belt-and-braces guard for plain `+ - *`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace b25 {

using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using isz = ptrdiff_t;
using usz = size_t;

using S16 = std::span<i16>;
using CS16 = std::span<const i16>;
using SU16 = std::span<u16>;
using CSU16 = std::span<const u16>;
using A13 = std::span<i16, 13>;  // the 13-word bit-allocation descriptor

namespace rs {

// All of these compute in u32 (or u64 for 64-bit) so that no intermediate is a
// signed overflow, then convert back to T (modular in C++20).
template <class T>
using wide_u = std::conditional_t<(sizeof(T) > 4), u64, u32>;

template <class T>
constexpr T wadd(T a, std::type_identity_t<T> b) {
  return static_cast<T>(static_cast<wide_u<T>>(a) + static_cast<wide_u<T>>(b));
}
template <class T>
constexpr T wsub(T a, std::type_identity_t<T> b) {
  return static_cast<T>(static_cast<wide_u<T>>(a) - static_cast<wide_u<T>>(b));
}
template <class T>
constexpr T wmul(T a, std::type_identity_t<T> b) {
  return static_cast<T>(static_cast<wide_u<T>>(a) * static_cast<wide_u<T>>(b));
}
template <class T>
constexpr T wneg(T a) {
  return static_cast<T>(wide_u<T>(0) - static_cast<wide_u<T>>(a));
}
// wrapping_shl: the shift amount is masked to the type width.
template <class T>
constexpr T wshl(T a, u32 n) {
  return static_cast<T>(static_cast<wide_u<T>>(a) << (n & (sizeof(T) * 8 - 1)));
}
// wrapping_shr (arithmetic for signed T, logical for unsigned).
template <class T>
constexpr T wshr(T a, u32 n) {
  return static_cast<T>(a >> (n & (sizeof(T) * 8 - 1)));
}

}  // namespace rs
}  // namespace b25
