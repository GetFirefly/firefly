#ifndef LUMEN_MODULEBUILDER_H
#define LUMEN_MODULEBUILDER_H

#include "lumen/compiler/Support/LLVM.h"
#include "lumen/compiler/Support/MLIR.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Translation/ModuleBuilderSupport.h"

#include "llvm-c/Types.h"
#include "llvm-c/Core.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/ADT/StringMap.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"

namespace mlir {
class Builder;
class Location;
class Attribute;
class Type;
class FuncOp;
} // namespace mlir

namespace llvm {
class TargetMachine;
} // namespace llvm

namespace lumen {
namespace eir {
class FuncOp; 
} // namespace eir
} // namespace lumen

using ::mlir::MLIRContext;
using ::mlir::Builder;
using ::mlir::Location;
using ::mlir::Attribute;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;
using ::mlir::Block;
using ::mlir::Region;

using ::llvm::TargetMachine;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::SmallVectorImpl;
using ::llvm::StringRef;
using ::llvm::APInt;
using ::llvm::APFloat;

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
}

struct EirTypeAny { EirTypeTag::TypeTag tag; };
struct EirTypeTuple { EirTypeTag::TypeTag tag; unsigned arity; };

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

class ModuleBuilder {
public:
  ModuleBuilder(MLIRContext &context, StringRef name, const TargetMachine *tm);
  ~ModuleBuilder();

  void dump();
    
  mlir::ModuleOp finish();

  //===----------------------------------------------------------------------===//
  // Functions
  //===----------------------------------------------------------------------===//

  FuncOp create_function(StringRef functionName,
                         SmallVectorImpl<Arg> &functionArgs,
                         EirType *resultType);

  void declare_function(StringRef functionName, mlir::FunctionType fnType);

  void add_function(FuncOp f);

  //===----------------------------------------------------------------------===//
  // Blocks
  //===----------------------------------------------------------------------===//

  Block *add_block(FuncOp &f);
  Block *getBlock();
  void position_at_end(Block *block);
  //===----------------------------------------------------------------------===//
  // Control Flow
  //===----------------------------------------------------------------------===//

  void build_br(Block *dest, ValueRange destArgs = {});
  void build_if(Value value,
                Block *yes,
                Block *no,
                Block *other,
                SmallVectorImpl<Value> &yesArgs,
                SmallVectorImpl<Value> &noArgs,
                SmallVectorImpl<Value> &otherArgs);
  void build_unreachable();
  void build_return(Value value);

  void translate_call_to_intrinsic(
    StringRef target,
    ArrayRef<Value> args,
    bool isTail,
    Block *ok,
    ArrayRef<Value> okArgs,
    Block *err,
    ArrayRef<Value> errArgs);   

  void build_static_call(
    StringRef target,
    ArrayRef<Value> args,
    bool isTail,
    Block *ok,
    ArrayRef<Value> okArgs,
    Block *err,
    ArrayRef<Value> errArgs);

  //===----------------------------------------------------------------------===//
  // Operations
  //===----------------------------------------------------------------------===//

  void build_match(Match op);
  std::unique_ptr<MatchPattern> convertMatchPattern(const MLIRMatchPattern &inPattern);

  void build_map_update(MapUpdate op);
  void build_map_insert_op(Value map, Value key, Value val, Block *ok, Block *err);
  void build_map_update_op(Value map, Value key, Value val, Block *ok, Block *err);

  Value build_is_type_op(Value value, Type matchType);
  Value build_is_equal(Value lhs, Value rhs, bool isExact);
  Value build_is_not_equal(Value lhs, Value rhs, bool isExact);
  Value build_is_less_than_or_equal(Value lhs, Value rhs);
  Value build_is_less_than(Value lhs, Value rhs);
  Value build_is_greater_than_or_equal(Value lhs, Value rhs);
  Value build_is_greater_than(Value lhs, Value rhs);
  Value build_logical_and(Value lhs, Value rhs);
  Value build_logical_or(Value lhs, Value rhs);
  Value build_cons(Value head, Value tail);
  Value build_tuple(ArrayRef<Value> elements);
  Value build_map(ArrayRef<MapEntry> entries);

  Value build_print_op(ArrayRef<Value> args);
  void build_trace_capture_op(Block *dest, ArrayRef<MLIRValueRef> destArgs = {});

  //===----------------------------------------------------------------------===//
  // Constants
  //===----------------------------------------------------------------------===//

  Attribute build_float_attr(Type type, double value);
  Value build_constant_float(double value);
  Value build_constant_int(int64_t value);
  Attribute build_int_attr(int64_t value, bool isSigned = true);
  Value build_constant_bigint(StringRef value, unsigned width);
  Attribute build_bigint_attr(StringRef value, unsigned width);
  Value build_constant_atom(StringRef value, uint64_t valueId);
  Attribute build_atom_attr(StringRef value, uint64_t valueId);
  Attribute build_string_attr(StringRef value);
  Value build_constant_binary(StringRef value, uint64_t header, uint64_t flags);
  Attribute build_binary_attr(StringRef value, uint64_t header, uint64_t flags);
  Value build_constant_nil();
  Attribute build_nil_attr();
  Value build_constant_list(ArrayRef<Attribute> elements);
  Value build_constant_tuple(ArrayRef<Attribute> elements);
  Value build_constant_map(ArrayRef<Attribute> elements);
  Attribute build_seq_attr(ArrayRef<Attribute> elements, Type type);

  template <typename Ty, typename... Args> Ty getType(Args... args) {
    return builder.getType<Ty>(args...);
  }

  Type getArgType(const Arg *arg);

private:
  const TargetMachine* targetMachine;
    
  /// The module we're building, essentially equivalent to the EIR module
  mlir::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  mlir::OpBuilder builder;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::FunctionType> calledSymbols;

  Location loc(Span span);
};

} // namespace eir
} // namespace lumen

#endif // LUMEN_MODULEBUILDER_H
