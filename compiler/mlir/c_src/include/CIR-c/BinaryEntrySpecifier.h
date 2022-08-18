#pragma once

#include <cstdlib>

#ifdef __cplusplus
namespace mlir {
namespace cir {
enum class BinaryEntrySpecifierType : uint32_t {
  Integer = 0,
  Float = 1,
  Bytes = 2,
  Utf8 = 3,
  Utf16 = 4,
  Utf32 = 5
};

enum class Endianness : uint8_t { Big = 0, Little = 1, Native = 2 };

extern "C" {
#endif

struct BinaryEntrySpecifierInteger {
  uint8_t isSigned;
  uint8_t endianness;
  uint8_t unit;
  uint8_t _padding[1];
};

struct BinaryEntrySpecifierFloat {
  uint8_t endianness;
  uint8_t unit;
  uint8_t _padding[2];
};

struct BinaryEntrySpecifierBytes {
  uint8_t unit;
  uint8_t _padding[3];
};

struct BinaryEntrySpecifierUtfWide {
  uint8_t endianness;
  uint8_t _padding[3];
};

union BinaryEntrySpecifierData {
  BinaryEntrySpecifierInteger integer;
  BinaryEntrySpecifierFloat flt;
  BinaryEntrySpecifierBytes bytes;
  BinaryEntrySpecifierUtfWide utfWide;
  uint32_t raw;
};

struct BinaryEntrySpecifier {
  uint32_t tag;
  BinaryEntrySpecifierData data;
};

#ifdef __cplusplus
} // extern "C"
} // namespace mlir
} // namespace cir
#endif
