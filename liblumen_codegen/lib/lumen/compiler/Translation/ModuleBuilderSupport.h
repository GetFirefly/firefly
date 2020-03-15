#ifndef LUMEN_MODULEBUILDER_SUPPORT_H
#define LUMEN_MODULEBUILDER_SUPPORT_H

#include <stdint.h>

#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using ::mlir::Block;
using ::mlir::Type;
using ::mlir::Value;

typedef struct MLIROpaqueValue *MLIRValueRef;
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;

namespace lumen {
namespace eir {

struct Closure {
  MLIRAttributeRef module;
  char *name;
  uint8_t arity;
  uint32_t index;
  uint32_t oldUnique;
  char unique[16];
  MLIRValueRef *env;
  unsigned envLen;
};

//===----------------------------------------------------------------------===//
// Binary Support Types
//===----------------------------------------------------------------------===//

struct MapEntry {
  MLIRValueRef key;
  MLIRValueRef value;
};

enum class Endianness : uint32_t {
  Big,
  Little,
  Native,
};

enum class BinarySpecifierType : uint32_t {
  Integer,
  Float,
  Bytes,
  Bits,
  Utf8,
  Utf16,
  Utf32,
};

struct IntegerSpecifier {
  bool isSigned;
  Endianness endianness;
  int64_t unit;
};

struct FloatSpecifier {
  Endianness endianness;
  int64_t unit;
};

struct BytesSpecifier {
  int64_t unit;
};

struct BitsSpecifier {
  int64_t unit;
};

struct Utf16Specifier {
  Endianness endianness;
};

struct Utf32Specifier {
  Endianness endianness;
};

union BinarySpecifierPayload {
  IntegerSpecifier i;
  FloatSpecifier f;
  BytesSpecifier bytes;
  BitsSpecifier bits;
  Utf16Specifier utf16;
  Utf32Specifier utf32;
};

struct BinarySpecifier {
  BinarySpecifierType tag;
  BinarySpecifierPayload payload;
};

//===----------------------------------------------------------------------===//
// MatchOp Support Types
//===----------------------------------------------------------------------===//

enum class MatchPatternType : uint32_t {
  Any,
  Cons,
  Tuple,
  MapItem,
  IsType,
  Value,
  Binary,
};

class MatchPattern {
 public:
  MatchPattern(MatchPatternType tag) : tag(tag) {}

  MatchPatternType getKind() const { return tag; }

 private:
  MatchPatternType tag;
};

class AnyPattern : public MatchPattern {
 public:
  AnyPattern() : MatchPattern(MatchPatternType::Any) {}

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Any;
  }
};

class ConsPattern : public MatchPattern {
 public:
  ConsPattern() : MatchPattern(MatchPatternType::Cons) {}

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Cons;
  }
};

class TuplePattern : public MatchPattern {
 public:
  TuplePattern(unsigned arity)
      : MatchPattern(MatchPatternType::Tuple), arity(arity) {}

  unsigned getArity() const { return arity; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Tuple;
  }

 private:
  unsigned arity;
};

class MapPattern : public MatchPattern {
 public:
  MapPattern(Value key) : MatchPattern(MatchPatternType::MapItem), key(key) {}

  Value getKey() { return key; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::MapItem;
  }

 private:
  Value key;
};

class IsTypePattern : public MatchPattern {
 public:
  IsTypePattern(Type ty)
      : MatchPattern(MatchPatternType::IsType), expectedType(ty) {}

  Type getExpectedType() { return expectedType; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::IsType;
  }

 private:
  Type expectedType;
};

class ValuePattern : public MatchPattern {
 public:
  ValuePattern(Value value)
      : MatchPattern(MatchPatternType::Value), value(value) {}

  Value getValue() { return value; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Value;
  }

 private:
  Value value;
};

class BinaryPattern : public MatchPattern {
 public:
  BinaryPattern(BinarySpecifier spec, llvm::Optional<Value> size = llvm::None)
      : MatchPattern(MatchPatternType::Binary), spec(spec), size(size) {}

  llvm::Optional<Value> getSize() { return size; }
  BinarySpecifier &getSpec() { return spec; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Binary;
  }

 private:
  llvm::Optional<Value> size;
  BinarySpecifier spec;
};

}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_MODULEBUILDER_SUPPORT_H
