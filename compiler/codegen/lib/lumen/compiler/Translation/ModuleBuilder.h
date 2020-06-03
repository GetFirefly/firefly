#ifndef LUMEN_MODULEBUILDER_H
#define LUMEN_MODULEBUILDER_H

#include "lumen/compiler/Translation/ModuleBuilderSupport.h"
#include "lumen/llvm/Target.h"
#include "lumen/mlir/IR.h"
#include "lumen/mlir/MLIR.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using ::llvm::APFloat;
using ::llvm::APInt;
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::SmallVectorImpl;
using ::llvm::StringRef;
using ::mlir::LLVM::LLVMType;
using ::mlir::MLIRContext;

namespace LLVM = ::mlir::LLVM;

typedef struct OpaqueModuleBuilder *MLIRModuleBuilderRef;

namespace lumen {
namespace eir {

class FuncOp;

class ModuleBuilder {
 public:
  ModuleBuilder(MLIRContext &context, StringRef name, Location loc,
                const llvm::TargetMachine *tm);
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

  Block *build_landing_pad(Location loc, ArrayRef<Value> catchClauses, Block *err);

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
  void build_binary_start(Location loc, Block *cont);
  void build_binary_push(Location loc, Value head, Value tail, Value size,
                         BinarySpecifier *spec, Block *ok, Block *err);
  void build_binary_finish(Location loc, Block *cont, Value bin);
  void build_receive_start(Location loc, Block *cont, Value timeout);
  void build_receive_wait(Location loc, Block *timeout, Block *check, Value receive_ref);
  void build_receive_done(Location loc, Block *cont, Value receive_ref, ArrayRef<Value> args);

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

  FuncOp getOrDeclareFunction(StringRef symbol, Type resultTy, bool isVarArg,
                              ArrayRef<Type> argTypes = {});

  Location getLocation(SourceLocation sloc);
  Location getFusedLocation(ArrayRef<Location> locs);

  bool isLikeMsvc();

 private:
  const llvm::TargetMachine *targetMachine;

  /// The module we're building, essentially equivalent to the EIR module
  mlir::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  mlir::OpBuilder builder;

  mlir::LLVM::LLVMDialect *llvmDialect;

  Location loc(Span span);
};

}  // namespace eir
}  // namespace lumen

#endif  // LUMEN_MODULEBUILDER_H
