#ifndef EIR_SUPPORT_TYPES_H
#define EIR_SUPPORT_TYPES_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/Casting.h"

#include <stdint.h>

namespace mlir {
class Value;
class Block;
}

typedef struct MLIROpaqueValue *MLIRValueRef;

namespace eir {

//===----------------------------------------------------------------------===//
// Binary Support Types
//===----------------------------------------------------------------------===//

struct MapEntry {
  MLIRValueRef key;
  MLIRValueRef value;
};

enum class Endianness: uint32_t {
    Big,
    Little,
    Native,
};

enum class BinarySpecifierType: uint32_t {
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

enum class MatchPatternType: uint32_t {
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
  MapPattern(M::Value key)
    : MatchPattern(MatchPatternType::MapItem), key(key) {}

  M::Value getKey() { return key; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::MapItem;
  }

private:
  M::Value key;
};

class IsTypePattern : public MatchPattern {
public:
  IsTypePattern(M::Type ty)
    : MatchPattern(MatchPatternType::IsType), expectedType(ty) {}

  M::Type getExpectedType() { return expectedType; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::IsType;
  }

private:
  M::Type expectedType;
};

class ValuePattern : public MatchPattern {
public:
  ValuePattern(M::Value value)
    : MatchPattern(MatchPatternType::Value), value(value) {}

  M::Value getValue() { return value; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Value;
  }

private:
  M::Value value;
};

class BinaryPattern : public MatchPattern {
public:
  BinaryPattern(BinarySpecifier spec, llvm::Optional<M::Value> size = llvm::None)
    : MatchPattern(MatchPatternType::Binary), spec(spec), size(size) {}

  llvm::Optional<M::Value> getSize() { return size; }
  BinarySpecifier &getSpec() { return spec; }

  static bool classof(const MatchPattern *pattern) {
    return pattern->getKind() == MatchPatternType::Binary;
  }

private:
  llvm::Optional<M::Value> size;
  BinarySpecifier spec;
};

} // namespace eir

#endif
