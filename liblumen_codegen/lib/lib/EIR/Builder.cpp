#include "eir/Builder.h"
#include "eir/Context.h"
#include "eir/Ops.h"
#include "eir/Types.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"

namespace M = mlir;
namespace L = llvm;

using llvm::StringRef;

namespace eir {

class ModuleBuilder {
public:
  ModuleBuilder(M::MLIRContext &context, StringRef name) : builder(&context) {
    // Create an empty module into which we can codegen functions
    theModule = M::ModuleOp::create(builder.getUnknownLoc(), name);
  }

  ~ModuleBuilder() {
    if (theModule)
      theModule.erase();
  }

  M::ModuleOp finish() {
    M::ModuleOp finished;
    std::swap(finished, theModule);
    return finished;
  }

  M::FuncOp declare_function(StringRef functionName,
                             L::SmallVectorImpl<Arg> &functionArgs,
                             EirTypes resultType) {

    L::SmallVector<M::Type, 2> argTypes;
    argTypes.reserve(functionArgs.size());
    for (auto it = functionArgs.begin(); it + 1 != functionArgs.end(); ++it) {
      M::Type type = getArgType(*it);
      if (!type)
        return nullptr;
      argTypes.push_back(type);
    }
    M::Type retType = getTypeFromKind(resultType);
    if (!retType) {
      auto fnType = builder.getFunctionType(argTypes, L::None);
      return M::FuncOp::create(builder.getUnknownLoc(), functionName, fnType);
    } else {
      auto fnType = builder.getFunctionType(argTypes, retType);
      return M::FuncOp::create(builder.getUnknownLoc(), functionName, fnType);
    }
  }

  M::Block *add_entry_block(M::FuncOp &f) {
    auto entryBlock = f.addEntryBlock();
    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToEnd(entryBlock);
    return entryBlock;
  }

  M::Block *add_block(M::FuncOp &f) { return f.addBlock(); }

  void position_at_end(M::Block *block) {
    builder.setInsertionPointToEnd(block);
  }

  void build_br(M::Block *dest) {
    builder.create<M::BranchOp>(builder.getUnknownLoc(), dest);
  }

  void build_unreachable() {
    builder.create<eir::UnreachableOp>(builder.getUnknownLoc());
  }

protected:
  M::Type getArgType(Arg &arg) { return getTypeFromKind(fromRust(arg.ty)); }

  M::Type getTypeFromKind(EirTypes kind) {
    auto *ctx = builder.getContext();
    switch (kind) {
    case EirTypes::Term:
      return TermType::get(ctx);
    case EirTypes::Atom:
      return AtomType::get(ctx);
    case EirTypes::Boolean:
      return BooleanType::get(ctx);
    case EirTypes::Fixnum:
      return FixnumType::get(ctx);
    case EirTypes::BigInt:
      return BigIntType::get(ctx);
    case EirTypes::Float:
      return FloatType::get(ctx);
    case EirTypes::FloatPacked:
      return PackedFloatType::get(ctx);
    case EirTypes::Nil:
      return NilType::get(ctx);
    case EirTypes::Cons:
      return ConsType::get(ctx);
    case EirTypes::Map:
      return MapType::get(ctx);
    case EirTypes::HeapBin:
      return HeapBinType::get(ctx);
    default:
      llvm::report_fatal_error(
          "tried to construct MLIR type from invalid EIR type kind");
    }
  }

private:
  /// The module we're building, essentially equivalent to the EIR module
  M::ModuleOp theModule;

  /// The builder is used for generating IR inside of functions in the module,
  /// it is very similar to the LLVM builder
  M::OpBuilder builder;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<M::FuncOp> functionMap;

  M::Location loc(Span span) {
    MLIRLocationRef fileLocRef = EIRSpanToMLIRLocation(span.start, span.end);
    M::Location *fileLoc = unwrap(fileLocRef);
    return *fileLoc;
  }
};

EirTypes fromRust(EirType t) {
  switch (t) {
  case EirType::Term:
    return EirTypes::Term;
  case EirType::Atom:
    return EirTypes::Atom;
  case EirType::Boolean:
    return EirTypes::Boolean;
  case EirType::Fixnum:
    return EirTypes::Fixnum;
  case EirType::BigInt:
    return EirTypes::BigInt;
  case EirType::Float:
    return EirTypes::Float;
  case EirType::FloatPacked:
    return EirTypes::FloatPacked;
  case EirType::Nil:
    return EirTypes::Nil;
  case EirType::Cons:
    return EirTypes::Cons;
  case EirType::Tuple:
    return EirTypes::Tuple;
  case EirType::Map:
    return EirTypes::Map;
  case EirType::Closure:
    return EirTypes::Closure;
  case EirType::HeapBin:
    return EirTypes::HeapBin;
  case EirType::Box:
    return EirTypes::Box;
  case EirType::Ref:
    return EirTypes::Ref;
  default:
    llvm::report_fatal_error("Bad EirType.");
  }
}

} // namespace eir

MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                             const char *name) {
  M::MLIRContext *ctx = unwrap(context);
  StringRef moduleName(name);
  return wrap(new eir::ModuleBuilder(*ctx, moduleName));
}

MLIRModuleRef MLIRFinalizeModuleBuilder(MLIRModuleBuilderRef b) {
  eir::ModuleBuilder *builder = unwrap(b);
  M::ModuleOp finished = builder->finish();
  delete builder;
  if (failed(mlir::verify(finished))) {
    finished.emitError("module verification error");
    return nullptr;
  }

  // Move to the heap
  return wrap(new M::ModuleOp(finished));
}

MLIRFunctionOpRef MLIRCreateFunction(MLIRModuleBuilderRef b, const char *name,
                                     const eir::Arg *argv, int argc,
                                     eir::EirType type) {
  eir::ModuleBuilder *builder = unwrap(b);
  StringRef functionName(name);
  L::SmallVector<eir::Arg, 2> functionArgs(argv, argv + argc);
  eir::EirTypes resultType = fromRust(type);
  auto fun = builder->declare_function(functionName, functionArgs, resultType);
  if (!fun)
    return nullptr;

  return wrap(new M::FuncOp(fun));
}

MLIRBlockRef MLIRAppendEntryBlock(MLIRModuleBuilderRef b, MLIRFunctionOpRef f) {
  eir::ModuleBuilder *builder = unwrap(b);
  M::FuncOp *fun = unwrap(f);
  auto block = builder->add_entry_block(*fun);
  if (!block)
    return nullptr;
  return wrap(block);
}

MLIRBlockRef MLIRAppendBasicBlock(MLIRModuleBuilderRef b, MLIRFunctionOpRef f) {
  eir::ModuleBuilder *builder = unwrap(b);
  M::FuncOp *fun = unwrap(f);
  auto block = builder->add_block(*fun);
  if (!block)
    return nullptr;
  return wrap(block);
}

void MLIRBlockPositionAtEnd(MLIRModuleBuilderRef b, MLIRBlockRef blk) {
  eir::ModuleBuilder *builder = unwrap(b);
  M::Block *block = unwrap(blk);
  builder->position_at_end(block);
}

void MLIRBuildBr(MLIRModuleBuilderRef b, MLIRBlockRef destBlk) {
  eir::ModuleBuilder *builder = unwrap(b);
  M::Block *dest = unwrap(destBlk);
  builder->build_br(dest);
}

void MLIRBuildUnreachable(MLIRModuleBuilderRef b) {
  eir::ModuleBuilder *builder = unwrap(b);
  builder->build_unreachable();
}

MLIRLocationRef MLIRCreateLocation(MLIRContextRef context, const char *filename,
                                   unsigned line, unsigned column) {
  M::MLIRContext *ctx = unwrap(context);
  StringRef FileName(filename);
  M::Location loc = M::FileLineColLoc::get(FileName, line, column, ctx);
  return wrap(&loc);
}
