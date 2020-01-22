#ifndef EIR_BUILDER_H
#define EIR_BUILDER_H

#include "lumen/LLVM.h"

#include "eir/Context.h"
#include "eir/Types.h"
#include "eir/SupportTypes.h"

#include "llvm-c/Types.h"
#include "llvm/Support/CBindingWrapping.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

#include "llvm-c/Core.h"

namespace mlir {
class Builder;
class Location;
class FuncOp;
class Attribute;
class Type;
} // namespace mlir

namespace L = llvm;
namespace M = mlir;

typedef struct MLIROpaqueBuilder *MLIRBuilderRef;
typedef struct MLIROpaqueLocation *MLIRLocationRef;
typedef struct MLIROpaqueFuncOp *MLIRFunctionOpRef;
typedef struct MLIROpaqueBlock *MLIRBlockRef;
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Builder, MLIRBuilderRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Location, MLIRLocationRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::FuncOp, MLIRFunctionOpRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(M::Block, MLIRBlockRef);

namespace eir {

extern "C" {

/// A source location in EIR
struct Span {
  // The starting byte index of a span
  uint32_t start;
  // The end byte index of a span
  uint32_t end;
};

struct FunctionDeclResult {
  MLIRFunctionOpRef function;
  MLIRBlockRef entryBlock;
};

enum class EirTypeTag : uint32_t {
  Unknown = 0,
  Void,
  Term,
  AnyList,
  AnyNumber,
  AnyInteger,
  AnyFloat,
  AnyBinary,
  Atom,
  Boolean,
  Fixnum,
  BigInt,
  Float,
  FloatPacked,
  Nil,
  Cons,
  Tuple,
  Map,
  Closure,
  HeapBin,
  Box,
  Ref,
};

struct EirTypeAny { EirTypeTag tag; };
struct EirTypeTuple { EirTypeTag tag; unsigned arity; };

union EirType {
  EirTypeAny any;
  EirTypeTuple tuple;
};

struct Arg {
  EirType ty;
  Span span;
  bool isImplicit;
};

enum class MapActionType : uint32_t {
  Unknown = 0,
  Insert,
  Update
};

struct MapAction {
  MapActionType action;
  MLIRValueRef key;
  MLIRValueRef value;
};

struct MapUpdate {
  MLIRValueRef map;
  MLIRBlockRef ok;
  MLIRBlockRef err;
  MapAction *actionsv;
  size_t actionsc;
};

struct KeyValuePair {
  MLIRAttributeRef key;
  MLIRAttributeRef value;
};

struct MLIRBinaryPayload { MLIRValueRef size; BinarySpecifier spec; };

union MLIRMatchPatternPayload {
  unsigned i;
  MLIRValueRef v;
  EirType t;
  MLIRBinaryPayload b;
};

struct MLIRMatchPattern {
  MatchPatternType tag;
  MLIRMatchPatternPayload payload;
};

struct MLIRMatchBranch {
  MLIRBlockRef dest;
  MLIRValueRef *destArgv;
  unsigned destArgc;
  MLIRMatchPattern pattern;
};

struct Match {
  MLIRValueRef selector;
  MLIRMatchBranch *branches;
  unsigned numBranches;
};

} // extern "C"

class ModuleBuilder {
public:
  ModuleBuilder(M::MLIRContext &context, L::StringRef name);
  ~ModuleBuilder();
  M::ModuleOp finish();

  //===----------------------------------------------------------------------===//
  // Functions
  //===----------------------------------------------------------------------===//

  M::FuncOp declare_function(L::StringRef functionName,
                             L::SmallVectorImpl<Arg> &functionArgs,
                             EirType *resultType);
  //===----------------------------------------------------------------------===//
  // Blocks
  //===----------------------------------------------------------------------===//

  M::Block *add_block(M::FuncOp &f);
  M::Block *getBlock();
  void position_at_end(M::Block *block);
  //===----------------------------------------------------------------------===//
  // Control Flow
  //===----------------------------------------------------------------------===//

  void build_br(M::Block *dest, M::ValueRange destArgs = {});
  void build_if(M::Value value,
                M::Block *yes,
                M::Block *no,
                M::Block *other,
                L::SmallVectorImpl<M::Value> &yesArgs,
                L::SmallVectorImpl<M::Value> &noArgs,
                L::SmallVectorImpl<M::Value> &otherArgs);
  void build_unreachable();
  void build_return(M::Value value);

  void build_static_call(
    L::StringRef target,
    L::ArrayRef<M::Value> args,
    bool isTail,
    M::Block *ok,
    L::ArrayRef<M::Value> okArgs,
    M::Block *err,
    L::ArrayRef<M::Value> errArgs);

  //===----------------------------------------------------------------------===//
  // Operations
  //===----------------------------------------------------------------------===//

  void build_match(Match op);
  void build_map_update(MapUpdate op);
  M::Value build_is_type_op(M::Value value, M::Type matchType);
  M::Value build_is_equal(M::Value lhs, M::Value rhs, bool isExact);
  M::Value build_is_not_equal(M::Value lhs, M::Value rhs, bool isExact);
  M::Value build_is_less_than_or_equal(M::Value lhs, M::Value rhs);
  M::Value build_is_less_than(M::Value lhs, M::Value rhs);
  M::Value build_is_greater_than_or_equal(M::Value lhs, M::Value rhs);
  M::Value build_is_greater_than(M::Value lhs, M::Value rhs);
  M::Value build_logical_and(M::Value lhs, M::Value rhs);
  M::Value build_logical_or(M::Value lhs, M::Value rhs);
  M::Value build_cons(M::Value head, M::Value tail);
  M::Value build_tuple(L::ArrayRef<M::Value> elements);
  M::Value build_map(L::ArrayRef<MapEntry> entries);

  //===----------------------------------------------------------------------===//
  // Constants
  //===----------------------------------------------------------------------===//

  M::Attribute build_float_attr(M::Type type, double value);
  M::Value build_constant_float(double value, bool isPacked);
  M::Value build_constant_int(int64_t value, unsigned width);
  M::Attribute build_int_attr(int64_t value, unsigned width, bool isSigned = true);
  M::Value build_constant_bigint(L::StringRef value, unsigned width);
  M::Attribute build_bigint_attr(L::StringRef value, unsigned width);
  M::Value build_constant_atom(L::StringRef value, uint64_t valueId, unsigned width);
  M::Attribute build_atom_attr(L::StringRef value, uint64_t valueId, unsigned width);
  M::Attribute build_string_attr(L::StringRef value);
  M::Value build_constant_binary(L::ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width);
  M::Attribute build_binary_attr(L::ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width);
  M::Value build_constant_nil(uint64_t value, unsigned width);
  M::Value build_constant_seq(L::ArrayRef<M::Attribute> elements, M::Type type);
  M::Attribute build_seq_attr(L::ArrayRef<M::Attribute> elements, M::Type type);

  template <typename Ty, typename... Args> Ty getType(Args... args) {
    return builder.getType<Ty>(args...);
  }

  M::Type getArgType(const Arg *arg);

protected:

  std::unique_ptr<MatchPattern> convertMatchPattern(MLIRMatchPattern &inPattern);

  void build_map_insert_op(M::Value map, M::Value key, M::Value val, M::Block *ok, M::Block *err);
  void build_map_update_op(M::Value map, M::Value key, M::Value val, M::Block *ok, M::Block *err);

private:
  /// The module we're building, essentially equivalent to the EIR module
  M::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  M::OpBuilder builder;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<M::FuncOp> functionMap;

  M::Location loc(Span span);
};

} // namespace eir

typedef struct OpaqueModuleBuilder *MLIRModuleBuilderRef;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(eir::ModuleBuilder, MLIRModuleBuilderRef);

extern "C" {

MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                             const char *name);

MLIRModuleRef MLIRFinalizeModuleBuilder(MLIRModuleBuilderRef builder);

eir::FunctionDeclResult MLIRCreateFunction(MLIRModuleBuilderRef builder,
                                           const char *name, const eir::Arg *argv,
                                           int argc, eir::EirType *type);

//===----------------------------------------------------------------------===//
// Blocks
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRGetCurrentBlockArgument(MLIRModuleBuilderRef builder, unsigned id);
MLIRValueRef MLIRGetBlockArgument(MLIRBlockRef block, unsigned id);

MLIRBlockRef MLIRAppendBasicBlock(MLIRModuleBuilderRef builder,
                                  MLIRFunctionOpRef f,
                                  const eir::Arg *argv,
                                  unsigned argc);

void MLIRBlockPositionAtEnd(MLIRModuleBuilderRef builder, MLIRBlockRef block);

//===----------------------------------------------------------------------===//
// Control Flow
//===----------------------------------------------------------------------===//

void MLIRBuildBr(MLIRModuleBuilderRef builder, MLIRBlockRef dest, MLIRValueRef *argv, unsigned argc);

void MLIRBuildIf(MLIRModuleBuilderRef builder,
                 MLIRValueRef value,
                 MLIRBlockRef yes,
                 MLIRValueRef *yesArgv,
                 unsigned yesArgc,
                 MLIRBlockRef no,
                 MLIRValueRef *noArgv,
                 unsigned noArgc,
                 MLIRBlockRef other,
                 MLIRValueRef *otherArgv,
                 unsigned otherArgc);

void MLIRBuildUnreachable(MLIRModuleBuilderRef builder);

void MLIRBuildReturn(MLIRModuleBuilderRef builder, MLIRValueRef value);

void MLIRBuildStaticCall(
  MLIRModuleBuilderRef builder,
  const char *name,
  MLIRValueRef *argv,
  unsigned argc,
  bool isTail,
  MLIRBlockRef okBlock,
  MLIRValueRef *okArgv,
  unsigned okArgc,
  MLIRBlockRef errBlock,
  MLIRValueRef *errArgv,
  unsigned errArgc);

//===----------------------------------------------------------------------===//
// Locations
//===----------------------------------------------------------------------===//

MLIRLocationRef EIRSpanToMLIRLocation(unsigned start, unsigned end);

MLIRLocationRef MLIRCreateLocation(MLIRContextRef context, const char *filename,
                                   unsigned line, unsigned column);

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

void MLIRBuildMatchOp(MLIRModuleBuilderRef builder, eir::Match op);

void MLIRBuildTraceCaptureOp(MLIRModuleBuilderRef builder, MLIRBlockRef dest, MLIRValueRef *argv, unsigned argc);
MLIRValueRef MLIRBuildTraceConstructOp(MLIRModuleBuilderRef builder, MLIRValueRef trace);

void MLIRBuildMapOp(MLIRModuleBuilderRef builder, eir::MapUpdate op);
MLIRValueRef MLIRBuildIsEqualOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs, bool isExact);
MLIRValueRef MLIRBuildIsNotEqualOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs, bool isExact);
MLIRValueRef MLIRBuildLessThanOrEqualOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);
MLIRValueRef MLIRBuildLessThanOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);
MLIRValueRef MLIRBuildGreaterThanOrEqualOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);
MLIRValueRef MLIRBuildGreaterThanOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);
MLIRValueRef MLIRBuildLogicalAndOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);
MLIRValueRef MLIRBuildLogicalOrOp(MLIRModuleBuilderRef, MLIRValueRef lhs, MLIRValueRef rhs);

MLIRValueRef MLIRCons(MLIRModuleBuilderRef, MLIRValueRef head, MLIRValueRef tail);
MLIRValueRef MLIRConstructTuple(MLIRModuleBuilderRef, MLIRValueRef *elements, unsigned num_elements);
MLIRValueRef MLIRConstructMap(MLIRModuleBuilderRef, eir::MapEntry *entries, unsigned num_entries);

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantFloat(MLIRModuleBuilderRef b, double value, bool isPacked);
MLIRAttributeRef MLIRBuildFloatAttr(MLIRModuleBuilderRef b, double value, bool isPacked);
MLIRValueRef MLIRBuildConstantInt(MLIRModuleBuilderRef b, int64_t value, unsigned width);
MLIRAttributeRef MLIRBuildIntAttr(MLIRModuleBuilderRef b, int64_t value, unsigned width);
MLIRValueRef MLIRBuildConstantBigInt(MLIRModuleBuilderRef b, const char *str, unsigned width);
MLIRAttributeRef MLIRBuildBigIntAttr(MLIRModuleBuilderRef b, const char *str, unsigned width);
MLIRValueRef MLIRBuildConstantAtom(MLIRModuleBuilderRef b, const char *str, uint64_t id, unsigned width);
MLIRAttributeRef MLIRBuildAtomAttr(MLIRModuleBuilderRef b, const char *str, uint64_t id, unsigned width);
MLIRValueRef MLIRBuildConstantBinary(MLIRModuleBuilderRef b, const char *str, unsigned size, uint64_t header, uint64_t flags, unsigned width);
MLIRAttributeRef MLIRBuildBinaryAttr(MLIRModuleBuilderRef b, const char *str, unsigned size, uint64_t header, uint64_t flags, unsigned width);
MLIRValueRef MLIRBuildConstantNil(MLIRModuleBuilderRef b, int64_t value, unsigned width);
MLIRAttributeRef MLIRBuildNilAttr(MLIRModuleBuilderRef b, int64_t value, unsigned width);
MLIRValueRef MLIRBuildConstantList(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements);
MLIRAttributeRef MLIRBuildListAttr(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements);
MLIRValueRef MLIRBuildConstantTuple(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements);
MLIRAttributeRef MLIRBuildTupleAttr(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements);
MLIRValueRef MLIRBuildConstantMap(MLIRModuleBuilderRef b, const eir::KeyValuePair *elements, int num_elements);
MLIRAttributeRef MLIRBuildMapAttr(MLIRModuleBuilderRef b, const eir::KeyValuePair *elements, int num_elements);

//===----------------------------------------------------------------------===//
// Type Checking
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildIsTypeTupleWithArity(MLIRModuleBuilderRef builder, MLIRValueRef value, unsigned arity);
MLIRValueRef MLIRBuildIsTypeList(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeNonEmptyList(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeNil(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeMap(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeNumber(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeFloat(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeInteger(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeFixnum(MLIRModuleBuilderRef builder, MLIRValueRef value);
MLIRValueRef MLIRBuildIsTypeBigInt(MLIRModuleBuilderRef builder, MLIRValueRef value);
}

#endif // EIR_BUILDER_H
