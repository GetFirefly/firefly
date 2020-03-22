#ifndef LUMEN_MODULEBUILDER_H
#define LUMEN_MODULEBUILDER_H

#include "llvm-c/Core.h"
#include "llvm-c/Types.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CBindingWrapping.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Support/LLVM.h"
#include "lumen/compiler/Support/MLIR.h"
#include "lumen/compiler/Translation/ModuleBuilderSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

namespace mlir {
class Builder;
class Location;
class Attribute;
class Type;
class FuncOp;
}  // namespace mlir

namespace llvm {
class TargetMachine;
}  // namespace llvm

namespace lumen {
namespace eir {
class FuncOp;
}  // namespace eir
}  // namespace lumen

using ::mlir::Attribute;
using ::mlir::Block;
using ::mlir::Builder;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::Region;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::SmallVectorImpl;
using ::llvm::StringRef;
using ::llvm::TargetMachine;

typedef struct MLIROpaqueBuilder *MLIRBuilderRef;
typedef struct MLIROpaqueLocation *MLIRLocationRef;
typedef struct MLIROpaqueFuncOp *MLIRFunctionOpRef;
typedef struct MLIROpaqueBlock *MLIRBlockRef;
typedef struct MLIROpaqueAttribute *MLIRAttributeRef;
typedef struct LLVMOpaqueTargetMachine *LLVMTargetMachineRef;
typedef struct OpaqueModuleBuilder *MLIRModuleBuilderRef;

namespace lumen {
namespace eir {

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

namespace EirTypeTag {
enum TypeTag {
#define EIR_TERM_KIND(Name, Val) Name = Val,
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/compiler/Dialect/EIR/IR/EIREncoding.h.inc"
};
}  // namespace EirTypeTag

struct EirTypeAny {
  EirTypeTag::TypeTag tag;
};
struct EirTypeTuple {
  EirTypeTag::TypeTag tag;
  unsigned arity;
};

union EirType {
  EirTypeAny any;
  EirTypeTuple tuple;
};

struct Arg {
  EirType ty;
  Span span;
  bool isImplicit;
};

enum class MapActionType : uint32_t { Unknown = 0, Insert, Update };

struct MapAction {
  MapActionType action;
  MLIRValueRef key;
  MLIRValueRef value;
};

struct MapUpdate {
  MLIRLocationRef loc;
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

struct MLIRBinaryPayload {
  MLIRValueRef size;
  BinarySpecifier spec;
};

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
  MLIRLocationRef loc;
  MLIRBlockRef dest;
  MLIRValueRef *destArgv;
  unsigned destArgc;
  MLIRMatchPattern pattern;
};

struct Match {
  MLIRLocationRef loc;
  MLIRValueRef selector;
  MLIRMatchBranch *branches;
  unsigned numBranches;
};

class ModuleBuilder {
 public:
  ModuleBuilder(MLIRContext &context, StringRef name, Location loc,
                const TargetMachine *tm);
  ~ModuleBuilder();

  void dump();

  mlir::ModuleOp finish();

  //===----------------------------------------------------------------------===//
  // Functions
  //===----------------------------------------------------------------------===//

  FuncOp create_function(Location loc, StringRef functionName,
                         SmallVectorImpl<Arg> &functionArgs,
                         EirType *resultType);

  void add_function(FuncOp f);

  Value build_closure(Closure *closure);
  Value build_unpack_op(Location loc, Value env, unsigned index);

  //===----------------------------------------------------------------------===//
  // Blocks
  //===----------------------------------------------------------------------===//

  Block *add_block(FuncOp &f);
  Block *getBlock();
  void position_at_end(Block *block);
  //===----------------------------------------------------------------------===//
  // Control Flow
  //===----------------------------------------------------------------------===//

  void build_br(Location loc, Block *dest, ValueRange destArgs = {});
  void build_if(Location loc, Value value, Block *yes, Block *no, Block *other,
                SmallVectorImpl<Value> &yesArgs, SmallVectorImpl<Value> &noArgs,
                SmallVectorImpl<Value> &otherArgs);
  void build_unreachable(Location loc);
  void build_return(Location loc, Value value);

  void build_static_call(Location loc, StringRef target, ArrayRef<Value> args,
                         bool isTail, Block *ok, ArrayRef<Value> okArgs,
                         Block *err, ArrayRef<Value> errArgs);

  void build_closure_call(Location loc, Value closure, ArrayRef<Value> args,
                          bool isTail, Block *ok, ArrayRef<Value> okArgs,
                          Block *err, ArrayRef<Value> errArgs);

  void build_call_landing_pad(Location loc, Value result, Block *ok,
                              ArrayRef<Value> okArgs, Block *err,
                              ArrayRef<Value> errArgs);

  //===----------------------------------------------------------------------===//
  // Operations
  //===----------------------------------------------------------------------===//

  void build_match(Match op);
  std::unique_ptr<MatchPattern> convertMatchPattern(
      const MLIRMatchPattern &inPattern);

  void build_map_update(MapUpdate op);
  void build_map_insert_op(Location loc, Value map, Value key, Value val,
                           Block *ok, Block *err);
  void build_map_update_op(Location loc, Value map, Value key, Value val,
                           Block *ok, Block *err);

  Value build_is_type_op(Location loc, Value value, Type matchType);
  Value build_is_equal(Location loc, Value lhs, Value rhs, bool isExact);
  Value build_is_not_equal(Location loc, Value lhs, Value rhs, bool isExact);
  Value build_is_less_than_or_equal(Location loc, Value lhs, Value rhs);
  Value build_is_less_than(Location loc, Value lhs, Value rhs);
  Value build_is_greater_than_or_equal(Location loc, Value lhs, Value rhs);
  Value build_is_greater_than(Location loc, Value lhs, Value rhs);
  Value build_logical_and(Location loc, Value lhs, Value rhs);
  Value build_logical_or(Location loc, Value lhs, Value rhs);
  Value build_cons(Location loc, Value head, Value tail);
  Value build_tuple(Location loc, ArrayRef<Value> elements);
  Value build_map(Location loc, ArrayRef<MapEntry> entries);
  void build_binary_push(Location loc, Value head, Value tail, Value size,
                         BinarySpecifier *spec, Block *ok, Block *err);

  void build_trace_capture_op(Location loc, Block *dest,
                              ArrayRef<MLIRValueRef> destArgs = {});

  //===----------------------------------------------------------------------===//
  // Constants
  //===----------------------------------------------------------------------===//

  Attribute build_float_attr(Type type, double value);
  Value build_constant_float(Location loc, double value);
  Value build_constant_int(Location loc, int64_t value);
  Attribute build_int_attr(int64_t value, bool isSigned = true);
  Value build_constant_bigint(Location loc, StringRef value, unsigned width);
  Attribute build_bigint_attr(StringRef value, unsigned width);
  Value build_constant_atom(Location loc, StringRef value, uint64_t valueId);
  Attribute build_atom_attr(StringRef value, uint64_t valueId);
  Attribute build_string_attr(StringRef value);
  Value build_constant_binary(Location loc, StringRef value, uint64_t header,
                              uint64_t flags);
  Attribute build_binary_attr(StringRef value, uint64_t header, uint64_t flags);
  Value build_constant_nil(Location loc);
  Attribute build_nil_attr();
  Value build_constant_list(Location loc, ArrayRef<Attribute> elements);
  Value build_constant_tuple(Location loc, ArrayRef<Attribute> elements);
  Value build_constant_map(Location loc, ArrayRef<Attribute> elements);
  Attribute build_seq_attr(ArrayRef<Attribute> elements, Type type);

  template <typename Ty, typename... Args>
  Ty getType(Args... args) {
    return builder.getType<Ty>(args...);
  }

  Type getArgType(const Arg *arg);

  mlir::OpBuilder &getBuilder() { return builder; }

  mlir::MLIRContext *getContext() { return builder.getContext(); }

  Location getLocation(SourceLocation sloc);
  Location getFusedLocation(ArrayRef<Location> locs);

 private:
  const TargetMachine *targetMachine;

  /// The module we're building, essentially equivalent to the EIR module
  mlir::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  mlir::OpBuilder builder;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::FunctionType> calledSymbols;

  Location loc(Span span);
};

}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_MODULEBUILDER_H
