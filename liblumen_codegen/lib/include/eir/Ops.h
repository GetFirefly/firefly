#ifndef EIR_OPS_H_
#define EIR_OPS_H_

#include "eir/Attributes.h"
#include "eir/SupportTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace M = mlir;
namespace L = llvm;

using llvm::ArrayRef;
using llvm::StringRef;

namespace eir {

class FloatType;

class MatchBranch {
public:
  MatchBranch(M::Block *dest, ArrayRef<M::Value> destArgs, std::unique_ptr<MatchPattern> pattern)
     : dest(dest), destArgs(destArgs.begin(), destArgs.end()), pattern(std::move(pattern)) {}

  M::Block *getDest() const { return dest; }
  ArrayRef<M::Value> getDestArgs() const { return destArgs; }
  MatchPatternType getPatternType() const { return pattern->getKind(); }
  bool isCatchAll() const { return getPatternType() == MatchPatternType::Any; }

  MatchPattern *getPattern() const { return pattern.get(); }

  template <typename T>
  T *getPatternTypeOrNull() const {
    T *result = dyn_cast<T>(getPattern());
    return result;
  }

private:
  M::Block *dest;
  L::SmallVector<M::Value, 3> destArgs;
  std::unique_ptr<MatchPattern> pattern;
};

//===----------------------------------------------------------------------===//
// TableGen
//===----------------------------------------------------------------------===//

/// All operations are declared in this auto-generated header
#define GET_OP_CLASSES
#include "eir/Ops.h.inc"

//===----------------------------------------------------------------------===//
// Operation Sub-Types
//===----------------------------------------------------------------------===//

/// This is a refinement of the "constant" op for the case where it is
/// returning a float value of eir::FloatType or eir::PackedFloatType.
///
///   %1 = "std.constant"(){value: 42.0} : !eir.float
///
class ConstantFloatOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Builds a constant float op producing a float of the specified type.
  static void build(M::Builder *builder, M::OperationState &result,
                    const L::APFloat &value, M::Type type);

  M::APFloat getValue() { return getAttrOfType<M::FloatAttr>("value").getValue(); }

  static bool classof(M::Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of eir::FixnumType.
///
///   %1 = "std.constant"(){value: 42, encoding: 1} : !eir.fixnum
///
/// The 'encoding' attribute contains the integer value of the 'EncodingType' enum,
/// which is used during lowering, to encode the term correctly for the target
///
class ConstantIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  /// Build a constant int op producing an integer of the specified width.
  static void build(M::Builder *builder, M::OperationState &result, int64_t value,
                    unsigned width);

  int64_t getValue() { return getAttrOfType<M::IntegerAttr>("value").getInt(); }

  unsigned getEncoding() {
    return (unsigned) getAttrOfType<M::IntegerAttr>("encoding")
        .getValue()
        .getLimitedValue();
  }

  static bool classof(M::Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value of eir::BigIntType.
///
///   %1 = "std.constant"(){value: 42} : !eir.bigint
///
class ConstantBigIntOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  static void build(M::Builder *builder, M::OperationState &result, const L::APInt &value);

  L::APInt getValue() { return getAttrOfType<M::IntegerAttr>("value").getValue(); }

  static bool classof(M::Operation *op);
};


/// This is a refinement of the "constant" op for the case where it is
/// returning an atom value, i.e. eir::AtomType.
///
///   %1 = "std.constant"(){value: "foo"} : !eir.atom
///
class ConstantAtomOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  static void build(M::Builder *builder, M::OperationState &result, StringRef value, L::APInt id);

  L::APInt getValue() { return getAttrOfType<AtomAttr>("value").getValue(); }
  StringRef getStringValue() { return getAttrOfType<AtomAttr>("value").getStringValue(); }

  static bool classof(M::Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning a binary literal value.
///
///   %1 = "std.constant"(){value: "foo"} : !eir.binary
///
class ConstantBinaryOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  static void build(M::Builder *builder, M::OperationState &result, ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width);

  ArrayRef<char> getValue() { return getAttrOfType<BinaryAttr>("value").getValue(); }
  L::APInt &getHeader() { return getAttrOfType<BinaryAttr>("value").getHeader(); }
  L::APInt &getFlags() { return getAttrOfType<BinaryAttr>("value").getFlags(); }

  static bool classof(M::Operation *op);
};

/// This is a refinement of the "constant" op for the case where it is
/// returning a nil literal value.
///
///   %1 = "std.constant"(){value: 0} : !eir.nil
///
class ConstantNilOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  static void build(M::Builder *builder, M::OperationState &result, int64_t value,
                    unsigned width);

  static bool classof(M::Operation *op);
};


/// This is a refinement of the "constant" op for the case where it is
/// returning a literal list or tuple value.
///
///   %1 = "std.constant"(){value: [1, 2, 3]} : !eir.cons
///
///   %1 = "std.constant"(){value: [1, 2, 3]} : !eir.tuple
///
class ConstantSeqOp : public ConstantOp {
public:
  using ConstantOp::ConstantOp;

  static void build(M::Builder *builder, M::OperationState &result, ArrayRef<M::Attribute> elements, M::Type type);

  ArrayRef<M::Attribute> getValue() { return getAttrOfType<SeqAttr>("value").getValue(); }
  M::Type getValueType() { return getAttrOfType<SeqAttr>("value").getType(); }

  //size_t getNumElements() const { return getAttrOfType<SeqAttr>("value").getValue().size(); }

  static bool classof(M::Operation *op);
};

} // namespace eir

#endif // EIR_OPS_H_
