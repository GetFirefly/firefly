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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"

namespace M = mlir;
namespace L = llvm;

using llvm::StringRef;
using llvm::ArrayRef;

using namespace eir;

inline M::Attribute unwrap(const void *P) {
  return M::Attribute::getFromOpaquePointer(P);
}

inline MLIRAttributeRef wrap(const M::Attribute &attr) {
  auto ptr = attr.getAsOpaquePointer();
  return reinterpret_cast<MLIRAttributeRef>(const_cast<void*>(ptr));
}

inline M::Value unwrap(MLIRValueRef v) {
  return M::Value::getFromOpaquePointer(v);
}

inline MLIRValueRef wrap(const M::Value &attr) {
  auto ptr = attr.getAsOpaquePointer();
  return reinterpret_cast<MLIRValueRef>(const_cast<void*>(ptr));
}

inline raw_ostream &operator<<(raw_ostream &os, EirTypeTag tag) {
  auto i = static_cast<uint32_t>(tag);
  os << "EirTypeTag(raw=" << i << ", val=";
  switch (tag) {
    case EirTypeTag::Unknown:
      os << "Unknown";
      break;
    case EirTypeTag::Void:
      os << "Void";
      break;
    case EirTypeTag::Term:
      os << "Term";
      break;
    case EirTypeTag::AnyList:
      os << "AnyList";
      break;
    case EirTypeTag::AnyNumber:
      os << "AnyNumber";
      break;
    case EirTypeTag::AnyInteger:
      os << "AnyInteger";
      break;
    case EirTypeTag::AnyFloat:
      os << "AnyFloat";
      break;
    case EirTypeTag::AnyBinary:
      os << "AnyBinary";
      break;
    case EirTypeTag::Atom:
      os << "Atom";
      break;
    case EirTypeTag::Boolean:
      os << "Boolean";
      break;
    case EirTypeTag::Fixnum:
      os << "Fixnum";
      break;
    case EirTypeTag::BigInt:
      os << "BigInt";
      break;
    case EirTypeTag::Float:
      os << "Float";
      break;
    case EirTypeTag::FloatPacked:
      os << "FloatPacked";
      break;
    case EirTypeTag::Nil:
      os << "Nil";
      break;
    case EirTypeTag::Cons:
      os << "Cons";
      break;
    case EirTypeTag::Tuple:
      os << "Tuple";
      break;
    case EirTypeTag::Map:
      os << "Map";
      break;
    case EirTypeTag::Closure:
      os << "Closure";
      break;
    case EirTypeTag::HeapBin:
      os << "HeapBin";
      break;
    case EirTypeTag::Box:
      os << "Box";
      break;
    case EirTypeTag::Ref:
      os << "Ref";
      break;
  }
  os << ")";
  return os;
}

static M::Type fromRust(M::Builder &builder, const EirType *wrapper) {
  EirTypeTag t = wrapper->any.tag;
  auto *context = builder.getContext();
  switch (t) {
  case EirTypeTag::Void:
    return builder.getType<M::NoneType>();
  case EirTypeTag::Term:
    return builder.getType<TermType>();
  case EirTypeTag::AnyList:
    return builder.getType<AnyListType>();
  case EirTypeTag::AnyNumber:
    return builder.getType<AnyNumberType>();
  case EirTypeTag::AnyInteger:
    return builder.getType<AnyIntegerType>();
  case EirTypeTag::AnyFloat:
    return builder.getType<AnyFloatType>();
  case EirTypeTag::AnyBinary:
    return builder.getType<AnyBinaryType>();
  case EirTypeTag::Atom:
    return builder.getType<AtomType>();
  case EirTypeTag::Boolean:
    return builder.getType<BooleanType>();
  case EirTypeTag::Fixnum:
    return builder.getType<FixnumType>();
  case EirTypeTag::BigInt:
    return builder.getType<BigIntType>();
  case EirTypeTag::Float:
    return builder.getType<eir::FloatType>();
  case EirTypeTag::FloatPacked:
    return builder.getType<PackedFloatType>();
  case EirTypeTag::Nil:
    return builder.getType<NilType>();
  case EirTypeTag::Cons:
    return builder.getType<ConsType>();
  case EirTypeTag::Tuple: {
    auto arity = wrapper->tuple.arity;
    Shape shape(builder.getType<TermType>(), arity);
    return builder.getType<eir::TupleType>(shape);
  }
  case EirTypeTag::Map:
    return builder.getType<MapType>();
  case EirTypeTag::HeapBin:
    return builder.getType<HeapBinType>();
  case EirTypeTag::Box: {
    auto elementType = builder.getType<TermType>();
    return builder.getType<BoxType>(elementType);
  }
  case EirTypeTag::Ref: {
    auto elementType = builder.getType<TermType>();
    return builder.getType<RefType>(elementType);
  }
  default:
    llvm::outs() << "EirType(tag=" << t << ", payload=";
    if (t == EirTypeTag::Tuple) {
      llvm::outs() << wrapper->tuple.arity << ")\n";
    } else {
      llvm::outs() << "n/a)\n";
    }
    llvm::report_fatal_error("Bad EirType.");
  }
}

M::Type ModuleBuilder::getArgType(const Arg *arg) {
  return fromRust(builder, &arg->ty);
}

bool unwrapValues(MLIRValueRef *argv, unsigned argc, L::SmallVectorImpl<M::Value> &list) {
  if (argc < 1) {
    return false;
  }
  ArrayRef<MLIRValueRef> args(argv, argv + argc);
  for (auto it = args.begin(); it + 1 != args.end(); ++it) {
    M::Value arg = unwrap(*it);
    assert(arg != nullptr);
    list.push_back(arg);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ModuleBuilder
//===----------------------------------------------------------------------===//

MLIRModuleBuilderRef MLIRCreateModuleBuilder(MLIRContextRef context,
                                             const char *name) {
  M::MLIRContext *ctx = unwrap(context);
  StringRef moduleName(name);
  return wrap(new ModuleBuilder(*ctx, moduleName));
}

ModuleBuilder::ModuleBuilder(M::MLIRContext &context, StringRef name) : builder(&context) {
  // Create an empty module into which we can codegen functions
  theModule = M::ModuleOp::create(builder.getUnknownLoc(), name);
}

ModuleBuilder::~ModuleBuilder() {
  if (theModule)
    theModule.erase();
}

MLIRModuleRef MLIRFinalizeModuleBuilder(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  M::ModuleOp finished = builder->finish();
  delete builder;
  if (failed(mlir::verify(finished))) {
    finished.emitError("module verification error");
    return nullptr;
  }

  // Move to the heap
  return wrap(new M::ModuleOp(finished));
}

M::ModuleOp ModuleBuilder::finish() {
  M::ModuleOp finished;
  std::swap(finished, theModule);
  return finished;
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

eir::FunctionDeclResult MLIRCreateFunction(MLIRModuleBuilderRef b, const char *name,
                                           const Arg *argv, int argc,
                                           eir::EirType *type) {
  llvm::outs() << "type = " << std::addressof(type) << "\n";
  ModuleBuilder *builder = unwrap(b);
  StringRef functionName(name);
  L::SmallVector<Arg, 2> functionArgs(argv, argv + argc);
  auto fun = builder->declare_function(functionName, functionArgs, type);
  if (!fun)
    return {nullptr, nullptr};

  MLIRFunctionOpRef funRef = wrap(new M::FuncOp(fun));
  M::FuncOp *tempFun = unwrap(funRef);
  MLIRBlockRef entry = wrap(tempFun->addEntryBlock());

  return {funRef, entry};
}

M::FuncOp ModuleBuilder::declare_function(StringRef functionName,
                                          L::SmallVectorImpl<Arg> &functionArgs,
                                          eir::EirType *resultType) {
  L::SmallVector<M::Type, 2> argTypes;
  argTypes.reserve(functionArgs.size());
  for (auto it = functionArgs.begin(); it != functionArgs.end(); it++) {
    M::Type type = getArgType(it);
    if (!type)
    return nullptr;
    argTypes.push_back(type);
  }
  if (resultType->any.tag == EirTypeTag::Unknown) {
    auto fnType = builder.getFunctionType(argTypes, L::None);
    return M::FuncOp::create(builder.getUnknownLoc(), functionName, fnType);
  } else {
    auto fnType = builder.getFunctionType(argTypes, fromRust(builder, resultType));
    return M::FuncOp::create(builder.getUnknownLoc(), functionName, fnType);
  }
}

//===----------------------------------------------------------------------===//
// Blocks
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRGetCurrentBlockArgument(MLIRModuleBuilderRef b, unsigned id) {
    ModuleBuilder *builder = unwrap(b);
    M::Block *block = builder->getBlock();
    assert(block != nullptr);
    return wrap(block->getArgument(id));
}

M::Block *ModuleBuilder::getBlock() {
    return builder.getBlock();
}

MLIRValueRef MLIRGetBlockArgument(MLIRBlockRef b, unsigned id) {
  M::Block *block = unwrap(b);
  M::Value arg = block->getArgument(id);
  return wrap(arg);
}

MLIRBlockRef MLIRAppendBasicBlock(MLIRModuleBuilderRef b, MLIRFunctionOpRef f, const Arg *argv, unsigned argc) {
  llvm::outs() << "Appending basic block with " << argc << " arguments\n";
  ModuleBuilder *builder = unwrap(b);
  M::FuncOp *fun = unwrap(f);
  auto block = builder->add_block(*fun);
  if (!block)
    return nullptr;
  if (argc > 0) {
    ArrayRef<Arg> args(argv, argv + argc);
    for (auto it = args.begin(); it != args.end(); ++it) {
      M::Type type = builder->getArgType(it);
      block->addArgument(type);
    }
  }
  assert((block->getNumArguments() == argc) && "number of block arguments doesn't match requested arity!");
  return wrap(block);
}

M::Block *ModuleBuilder::add_block(M::FuncOp &f) { return f.addBlock(); }

void MLIRBlockPositionAtEnd(MLIRModuleBuilderRef b, MLIRBlockRef blk) {
  ModuleBuilder *builder = unwrap(b);
  M::Block *block = unwrap(blk);
  builder->position_at_end(block);
}

void ModuleBuilder::position_at_end(M::Block *block) {
  builder.setInsertionPointToEnd(block);
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

void MLIRBuildBr(MLIRModuleBuilderRef b, MLIRBlockRef destBlk, MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  M::Block *dest = unwrap(destBlk);
  if (argc > 0) {
    L::SmallVector<M::Value , 2> args;
    unwrapValues(argv, argc, args);
    builder->build_br(dest, args);
  } else {
    builder->build_br(dest);
  }
}

void ModuleBuilder::build_br(M::Block *dest, M::ValueRange destArgs) {
  builder.create<M::BranchOp>(builder.getUnknownLoc(), dest, destArgs);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

void MLIRBuildIf(MLIRModuleBuilderRef b,
                 MLIRValueRef val,
                 MLIRBlockRef y,
                 MLIRValueRef *yArgv,
                 unsigned yArgc,
                 MLIRBlockRef n,
                 MLIRValueRef *nArgv,
                 unsigned nArgc,
                 MLIRBlockRef o,
                 MLIRValueRef *oArgv,
                 unsigned oArgc) {
  ModuleBuilder *builder = unwrap(b);
  M::Value value = unwrap(val);
  M::Block *yes = unwrap(y);
  M::Block *no = unwrap(n);
  M::Block *other = unwrap(o);
  // Unwrap block args
  L::SmallVector<M::Value , 1> yesArgs;
  unwrapValues(yArgv, yArgc, yesArgs);
  L::SmallVector<M::Value , 1> noArgs;
  unwrapValues(nArgv, nArgc, noArgs);
  L::SmallVector<M::Value , 1> otherArgs;
  unwrapValues(oArgv, oArgc, otherArgs);
  // Construct operation
  builder->build_if(value, yes, no, other, yesArgs, noArgs, otherArgs);
}

void ModuleBuilder::build_if(M::Value value,
                             M::Block *yes,
                             M::Block *no,
                             M::Block *other,
                             L::SmallVectorImpl<M::Value> &yesArgs,
                             L::SmallVectorImpl<M::Value> &noArgs,
                             L::SmallVectorImpl<M::Value> &otherArgs) {
  // Create the `if`
  bool withOtherwiseRegion = other != nullptr;
  auto op = builder.create<IfOp>(builder.getUnknownLoc(), value, withOtherwiseRegion);
  // For each condition, generate a branch to the appropriate destination block
  auto ifBuilder = op.getIfBodyBuilder();
  ifBuilder.create<M::BranchOp>(builder.getUnknownLoc(), yes, yesArgs);
  auto elseBuilder = op.getElseBodyBuilder();
  ifBuilder.create<M::BranchOp>(builder.getUnknownLoc(), no, noArgs);
  if (withOtherwiseRegion) {
    auto otherBuilder = op.getOtherwiseBodyBuilder();
    otherBuilder.create<M::BranchOp>(builder.getUnknownLoc(), other, otherArgs);
  }
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

void MLIRBuildMatchOp(MLIRModuleBuilderRef b, eir::Match op) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_match(op);
}

std::unique_ptr<MatchPattern> ModuleBuilder::convertMatchPattern(MLIRMatchPattern &inPattern) {
  auto tag = inPattern.tag;
  switch (tag) {
    default:
      llvm_unreachable("unrecognized match pattern tag!");
    case MatchPatternType::Any:
      return std::unique_ptr<AnyPattern>(new AnyPattern());
    case MatchPatternType::Cons:
      return std::unique_ptr<ConsPattern>(new ConsPattern());
    case MatchPatternType::Tuple:
      return std::unique_ptr<TuplePattern>(new TuplePattern(inPattern.payload.i));
    case MatchPatternType::MapItem:
      return std::unique_ptr<MapPattern>(new MapPattern(unwrap(inPattern.payload.v)));
    case MatchPatternType::IsType: {
      auto t = &inPattern.payload.t;
      return std::unique_ptr<IsTypePattern>(new IsTypePattern(fromRust(builder, t)));
    }
    case MatchPatternType::Value:
      return std::unique_ptr<ValuePattern>(new ValuePattern(unwrap(inPattern.payload.v)));
    case MatchPatternType::Binary:
      auto payload = inPattern.payload.b;
      auto sizePtr = payload.size;
      if (sizePtr == nullptr) {
        return std::unique_ptr<BinaryPattern>(new BinaryPattern(inPattern.payload.b.spec));
      } else {
        M::Value size = unwrap(sizePtr);
        return std::unique_ptr<BinaryPattern>(new BinaryPattern(inPattern.payload.b.spec, size));
      }
  }
}

void ModuleBuilder::build_match(Match op) {
  // Convert FFI types into internal MLIR representation
  M::Value selector = unwrap(op.selector);
  L::SmallVector<MatchBranch, 2> branches;

  ArrayRef<MLIRMatchBranch> inBranches(op.branches, op.branches + op.numBranches);
  for (auto it = inBranches.begin(); it + 1 != inBranches.end(); ++it) {
    MLIRMatchBranch inBranch = *it;
    // Extract destination block and base arguments
    M::Block *dest = unwrap(inBranch.dest);
    ArrayRef<MLIRValueRef> inDestArgs(inBranch.destArgv, inBranch.destArgv + inBranch.destArgc);
    L::SmallVector<M::Value, 1> destArgs;
    for (auto it2 = inDestArgs.begin(); it2 + 1 != inDestArgs.end(); ++it2) {
      M::Value arg = unwrap(*it2);
      destArgs.push_back(arg);
    }
    // Convert match pattern payload
    auto pattern = convertMatchPattern(inBranch.pattern);
    // Create internal branch type
    MatchBranch branch(dest, destArgs, std::move(pattern));
    branches.push_back(std::move(branch));
  }
  
  // Create the operation using the internal representation
  builder.create<MatchOp>(builder.getUnknownLoc(),
                          selector,
                          branches);
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

void MLIRBuildUnreachable(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_unreachable();
}

void ModuleBuilder::build_unreachable() {
  builder.create<UnreachableOp>(builder.getUnknownLoc());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void MLIRBuildReturn(MLIRModuleBuilderRef b, MLIRValueRef value) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_return(unwrap(value));
}

void ModuleBuilder::build_return(M::Value value) {
  if (!value) {
    builder.create<M::ReturnOp>(builder.getUnknownLoc());
  } else {
    builder.create<M::ReturnOp>(builder.getUnknownLoc(), value);
  }
}

//===----------------------------------------------------------------------===//
// TraceCaptureOp/TraceConstructOp
//===----------------------------------------------------------------------===//

void MLIRBuildTraceCaptureOp(MLIRModuleBuilderRef b, MLIRBlockRef d, MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  M::Block *dest = unwrap(d);
  // TODO: We should generate a runtime call to capture the trace before the branch
  if (argc > 0) {
    L::SmallVector<M::Value, 1> args;
    unwrapValues(argv, argc, args);
    builder->build_br(dest, args);
  } else {
    builder->build_br(dest);
  }
}

MLIRValueRef MLIRBuildTraceConstructOp(MLIRModuleBuilderRef, MLIRValueRef) {
  // TODO: For now we do nothing, but we'll want code that fetches
  // a term value representing the stack trace
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

void MLIRBuildMapOp(MLIRModuleBuilderRef b, MapUpdate op) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_map_update(op);
}

void ModuleBuilder::build_map_update(MapUpdate op) {
  assert(op.actionsc > 0 && "cannot construct empty map op");
  L::SmallVector<MapAction, 2> actions(op.actionsv, op.actionsv + op.actionsc);
  M::Value map = unwrap(op.map);
  M::Block *ok = unwrap(op.ok);
  M::Block *err = unwrap(op.err);
  // Each insert or update implicitly branches to a continuation block for the next
  // insert/update; the last continuation block simply branches unconditionally to
  // the ok block
  M::Block *current = builder.getInsertionBlock();
  M::Region *parent = current->getParent();
  for (auto it = actions.begin(); it + 1 != actions.end(); ++it) {
    MapAction action = *it;
    M::Value key = unwrap(action.key);
    M::Value val = unwrap(action.value);
    // Create the continuation block, which expects the updated map as an argument
    M::Block *cont = builder.createBlock(parent);
    auto mapType = builder.getType<MapType>();
    cont->addArgument(mapType);
    // createBlock implicitly sets the insertion point to the new block,
    // so make sure we set it back to where we are now
    builder.setInsertionPointToEnd(current);
    switch (action.action) {
    case MapActionType::Insert:
      build_map_insert_op(map, key, val, cont, err);
      break;
    case MapActionType::Update:
      build_map_update_op(map, key, val, cont, err);
      break;
    default:
      llvm::report_fatal_error("tried to construct map update op with invalid type");
    }
    current = cont;
    // We need to update the `map` pointer, since we're implicitly in a new block on
    // the next iteration
    map = current->getArgument(0);
  }
  // After all updates, we can unconditionally branch to `ok`, since no errors could have occurred
  builder.setInsertionPointToEnd(current);
  ArrayRef<M::Value> okArgs = {map};
  build_br(ok, okArgs);
}

void ModuleBuilder::build_map_insert_op(M::Value map, M::Value key, M::Value val, M::Block *ok, M::Block *err) {
  // Perform the insert
  ArrayRef<M::Value> pairs = {key, val};
  auto op = builder.create<MapInsertOp>(builder.getUnknownLoc(), map, pairs);
  // Get the results, which is the updated map, and a error condition flag
  M::Value newMap = op.getResult(0);
  assert(newMap != nullptr);
  M::Value isOk = op.getResult(1);
  assert(isOk != nullptr);
  // Then branch to either the ok block, or the error block
  M::ValueRange okArgs = {newMap};
  M::ValueRange errArgs = {};
  builder.create<M::CondBranchOp>(builder.getUnknownLoc(), isOk, ok, okArgs, err, errArgs);
}

void ModuleBuilder::build_map_update_op(M::Value map, M::Value key, M::Value val, M::Block *ok, M::Block *err) {
  // Perform the update
  ArrayRef<M::Value> pairs = {key, val};
  auto op = builder.create<MapUpdateOp>(builder.getUnknownLoc(), map, pairs);
  // Get the results, which is the updated map, and a error condition flag
  M::Value newMap = op.getResult(0);
  assert(newMap != nullptr);
  M::Value isOk = op.getResult(1);
  assert(isOk != nullptr);
  // Then branch to either the ok block, or the error block
  M::ValueRange okArgs = {newMap};
  M::ValueRange errArgs = {};
  builder.create<M::CondBranchOp>(builder.getUnknownLoc(), isOk, ok, okArgs, err, errArgs);
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildIsEqualOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r, bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_equal(lhs, rhs, isExact));
}

M::Value ModuleBuilder::build_is_equal(M::Value lhs, M::Value rhs, bool isExact) {
  auto op = builder.create<IsEqualOp>(builder.getUnknownLoc(), lhs, rhs, isExact);
  return op.getResult();
}

MLIRValueRef MLIRBuildIsNotEqualOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r, bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_not_equal(lhs, rhs, isExact));
}

M::Value ModuleBuilder::build_is_not_equal(M::Value lhs, M::Value rhs, bool isExact) {
  auto op = builder.create<IsNotEqualOp>(builder.getUnknownLoc(), lhs, rhs, isExact);
  return op.getResult();
}

MLIRValueRef MLIRBuildLessThanOrEqualOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than_or_equal(lhs, rhs));
}

M::Value ModuleBuilder::build_is_less_than_or_equal(M::Value lhs, M::Value rhs) {
  auto op = builder.create<IsLessThanOrEqualOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

MLIRValueRef MLIRBuildLessThanOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than(lhs, rhs));
}

M::Value ModuleBuilder::build_is_less_than(M::Value lhs, M::Value rhs) {
  auto op = builder.create<IsLessThanOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

MLIRValueRef MLIRBuildGreaterThanOrEqualOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than_or_equal(lhs, rhs));
}

M::Value ModuleBuilder::build_is_greater_than_or_equal(M::Value lhs, M::Value rhs) {
  auto op = builder.create<IsGreaterThanOrEqualOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

MLIRValueRef MLIRBuildGreaterThanOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than(lhs, rhs));
}

M::Value ModuleBuilder::build_is_greater_than(M::Value lhs, M::Value rhs) {
  auto op = builder.create<IsGreaterThanOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Logical Operators
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildLogicalAndOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_logical_and(lhs, rhs));
}

M::Value ModuleBuilder::build_logical_and(M::Value lhs, M::Value rhs) {
  auto op = builder.create<LogicalAndOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

MLIRValueRef MLIRBuildLogicalOrOp(MLIRModuleBuilderRef b, MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  M::Value lhs = unwrap(l);
  M::Value rhs = unwrap(r);
  return wrap(builder->build_logical_or(lhs, rhs));
}

M::Value ModuleBuilder::build_logical_or(M::Value lhs, M::Value rhs) {
  auto op = builder.create<LogicalOrOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildStaticCall(MLIRModuleBuilderRef b, const char *name, MLIRValueRef *argv, unsigned argc, bool isTail) {
  ModuleBuilder *builder = unwrap(b);
  StringRef functionName(name);
  L::SmallVector<M::Value, 2> args;
  unwrapValues(argv, argc, args);
  return wrap(builder->build_static_call(functionName, args, isTail));
}

M::Value ModuleBuilder::build_static_call(StringRef target, ArrayRef<M::Value> args, bool isTail) {
  auto symbol = builder.getSymbolRefAttr(target);
  auto op = builder.create<CallOp>(builder.getUnknownLoc(), symbol, args, isTail);
  return op.getResult(0);
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRCons(MLIRModuleBuilderRef b, MLIRValueRef h, MLIRValueRef t) {
  ModuleBuilder *builder = unwrap(b);
  M::Value head = unwrap(h);
  M::Value tail = unwrap(t);
  return wrap(builder->build_cons(head, tail));
}

M::Value ModuleBuilder::build_cons(M::Value head, M::Value tail) {
  auto op = builder.create<ConsOp>(builder.getUnknownLoc(), head, tail);
  return op.getResult();
}

MLIRValueRef MLIRConstructTuple(MLIRModuleBuilderRef b, MLIRValueRef *es, unsigned num_es) {
  ModuleBuilder *builder = unwrap(b);
  L::SmallVector<M::Value, 2> elements;
  unwrapValues(es, num_es, elements);
  return wrap(builder->build_tuple(elements));
}

M::Value ModuleBuilder::build_tuple(ArrayRef<M::Value> elements) {
  auto op = builder.create<TupleOp>(builder.getUnknownLoc(), elements);
  return op.getResult();
}

MLIRValueRef MLIRConstructMap(MLIRModuleBuilderRef b, eir::MapEntry *es, unsigned num_es) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<eir::MapEntry> entries(es, es + num_es);
  return wrap(builder->build_map(entries));
}

M::Value ModuleBuilder::build_map(ArrayRef<eir::MapEntry> entries) {
  auto op = builder.create<ConstructMapOp>(builder.getUnknownLoc(), entries);
  return op.getResult(0);
}

//===----------------------------------------------------------------------===//
// ConstantFloat
//===----------------------------------------------------------------------===//


MLIRValueRef MLIRBuildConstantFloat(MLIRModuleBuilderRef b, double value, bool isPacked) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_float(value, isPacked));
}

M::Value ModuleBuilder::build_constant_float(double value, bool isPacked) {
  M::Type type;
  if (isPacked) {
    type = builder.getType<PackedFloatType>();
  } else {
    type = builder.getType<eir::FloatType>();
  }
  L::APFloat f(value);
  ConstantOp constantOp = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), f, type);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildFloatAttr(MLIRModuleBuilderRef b, double value, bool isPacked) {
  ModuleBuilder *builder = unwrap(b);
  if (isPacked) {
    auto type = builder->getType<PackedFloatType>();
    return wrap(builder->build_float_attr(type, value));
  } else {
    auto type = builder->getType<eir::FloatType>();
    return wrap(builder->build_float_attr(type, value));
  }
}

M::Attribute ModuleBuilder::build_float_attr(M::Type type, double value) {
  return builder.getFloatAttr(type, value);
}

//===----------------------------------------------------------------------===//
// ConstantInt
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantInt(MLIRModuleBuilderRef b, int64_t value, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_int(value, width));
}

M::Value ModuleBuilder::build_constant_int(int64_t value, unsigned width) {
  ConstantOp constantOp = builder.create<ConstantIntOp>(builder.getUnknownLoc(), value, width);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildIntAttr(MLIRModuleBuilderRef b, int64_t value, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_int_attr(value, width));
}

M::Attribute ModuleBuilder::build_int_attr(int64_t value, unsigned width, bool isSigned) {
  auto type = builder.getType<FixnumType>();
  L::APInt i(width, value, isSigned);
  return builder.getIntegerAttr(type, i);
}

//===----------------------------------------------------------------------===//
// ConstantBigInt
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantBigInt(MLIRModuleBuilderRef b, const char *str, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_constant_bigint(value, width));
}

M::Value ModuleBuilder::build_constant_bigint(StringRef value, unsigned width) {
  L::APInt i(width, value, /*radix=*/10);
  ConstantOp constantOp = builder.create<ConstantBigIntOp>(builder.getUnknownLoc(), i);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildBigIntAttr(MLIRModuleBuilderRef b, const char *str, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_bigint_attr(value, width));
}

M::Attribute ModuleBuilder::build_bigint_attr(StringRef value, unsigned width) {
  auto type = builder.getType<BigIntType>();
  L::APInt i(width, value, /*radix=*/10);
  return builder.getIntegerAttr(type, i);
}

//===----------------------------------------------------------------------===//
// ConstantAtom
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantAtom(MLIRModuleBuilderRef b, const char *str, uint64_t id, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_constant_atom(value, id, width));
}

M::Value ModuleBuilder::build_constant_atom(StringRef value, uint64_t valueId, unsigned width) {
  L::APInt id(width, valueId, /*isSigned=*/false);
  ConstantOp constantOp = builder.create<ConstantAtomOp>(builder.getUnknownLoc(), value, id);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildAtomAttr(MLIRModuleBuilderRef b, const char *str, uint64_t id, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_atom_attr(value, id, width));
}

M::Attribute ModuleBuilder::build_atom_attr(StringRef value, uint64_t valueId, unsigned width) {
  L::APInt id(width, valueId, /*isSigned=*/false);
  return AtomAttr::get(builder.getContext(), value, id);
}

M::Attribute ModuleBuilder::build_string_attr(StringRef value) {
  return builder.getStringAttr(value);
}

//===----------------------------------------------------------------------===//
// ConstantBinary
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantBinary(MLIRModuleBuilderRef b, const char *str, unsigned size, uint64_t header, uint64_t flags, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<char> value(str, str + size);
  return wrap(builder->build_constant_binary(value, header, flags, width));
}

M::Value ModuleBuilder::build_constant_binary(ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width) {
  ConstantOp constantOp = builder.create<ConstantBinaryOp>(
      builder.getUnknownLoc(), value, header, flags, width);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildBinaryAttr(MLIRModuleBuilderRef b, const char *str, unsigned size, uint64_t header, uint64_t flags, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<char> value(str, str + size);
  return wrap(builder->build_binary_attr(value, header, flags, width));
}

M::Attribute ModuleBuilder::build_binary_attr(ArrayRef<char> value, uint64_t header, uint64_t flags, unsigned width) {
  return BinaryAttr::get(builder.getContext(), value, header, flags, width);
}

//===----------------------------------------------------------------------===//
// ConstantNil
//===----------------------------------------------------------------------===//

MLIRValueRef MLIRBuildConstantNil(MLIRModuleBuilderRef b, int64_t value, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_nil(value, width));
}

M::Value ModuleBuilder::build_constant_nil(uint64_t value, unsigned width) {
  ConstantOp constantOp = builder.create<ConstantNilOp>(builder.getUnknownLoc(), value, width);
  return constantOp.getResult();
}

MLIRAttributeRef MLIRBuildNilAttr(MLIRModuleBuilderRef b, int64_t value, unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_int_attr(value, width, /*isSigned=*/false));
}

//===----------------------------------------------------------------------===//
// ConstantSeq (List/Tuple/Map)
//===----------------------------------------------------------------------===//

M::Value build_seq_op(ModuleBuilder *builder, ArrayRef<MLIRAttributeRef> elements, M::Type type) {
  L::SmallVector<M::Attribute, 3> list;
  list.reserve(elements.size());
  for (auto it = elements.begin(); it + 1 != elements.end(); ++it) {
    M::Attribute attr = unwrap(*it);
    if (!attr)
      return nullptr;
    list.push_back(attr);
  }
  return builder->build_constant_seq(list, type);
}

M::Value ModuleBuilder::build_constant_seq(ArrayRef<M::Attribute> elements, M::Type type) {
  ConstantOp constantOp = builder.create<ConstantSeqOp>(builder.getUnknownLoc(), elements, type);
  return constantOp.getResult();
}

MLIRValueRef MLIRBuildConstantList(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  auto type = builder->getType<ConsType>();

  return wrap(build_seq_op(builder, xs, type));
}

MLIRValueRef MLIRBuildConstantTuple(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);

  std::vector<M::Type> types;
  types.reserve(xs.size());
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    M::Attribute attr = unwrap(*it);
    if (!attr)
      return nullptr;
    types.push_back(attr.getType());
  }
  Shape shape(types);
  auto type = builder->getType<eir::TupleType>(shape);

  return wrap(build_seq_op(builder, xs, type));
}

MLIRValueRef MLIRBuildConstantMap(MLIRModuleBuilderRef b, const KeyValuePair *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<KeyValuePair> xs(elements, elements + num_elements);
  L::SmallVector<M::Attribute, 4> list;
  list.reserve(xs.size() * 2);
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    M::Attribute key = unwrap(it->key);
    if (!key)
      return nullptr;
    list.push_back(key);
    M::Attribute value = unwrap(it->value);
    if (!value)
      return nullptr;
    list.push_back(value);
  }
  auto type = builder->getType<MapType>();
  return wrap(builder->build_constant_seq(list, type));
}

M::Attribute build_seq_attr(ModuleBuilder *builder, ArrayRef<MLIRAttributeRef> elements, M::Type type) {
  L::SmallVector<M::Attribute, 3> list;
  list.reserve(elements.size());
  for (auto it = elements.begin(); it + 1 != elements.end(); ++it) {
    M::Attribute attr = unwrap(*it);
    if (!attr)
      return nullptr;
    list.push_back(attr);
  }
  return builder->build_seq_attr(list, type);
}

M::Attribute ModuleBuilder::build_seq_attr(ArrayRef<M::Attribute> elements, M::Type type) {
  return SeqAttr::get(elements, type);
}

MLIRAttributeRef MLIRBuildListAttr(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  auto type = builder->getType<ConsType>();

  return wrap(build_seq_attr(builder, xs, type));
}

MLIRAttributeRef MLIRBuildTupleAttr(MLIRModuleBuilderRef b, const MLIRAttributeRef *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);

  std::vector<M::Type> types;
  types.reserve(xs.size());
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    M::Attribute attr = unwrap(*it);
    if (!attr)
      return nullptr;
    types.push_back(attr.getType());
  }
  Shape shape(types);
  auto type = builder->getType<eir::TupleType>(shape);

  return wrap(build_seq_attr(builder, xs, type));
}

MLIRAttributeRef MLIRBuildMapAttr(MLIRModuleBuilderRef b, const KeyValuePair *elements, int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<KeyValuePair> xs(elements, elements + num_elements);
  L::SmallVector<M::Attribute, 4> list;
  list.reserve(xs.size() * 2);
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    M::Attribute key = unwrap(it->key);
    if (!key)
      return nullptr;
    list.push_back(key);
    M::Attribute value = unwrap(it->value);
    if (!value)
      return nullptr;
    list.push_back(value);
  }
  auto type = builder->getType<MapType>();
  return wrap(builder->build_seq_attr(list, type));
}

//===----------------------------------------------------------------------===//
// Locations/Spans
//===----------------------------------------------------------------------===//

MLIRLocationRef MLIRCreateLocation(MLIRContextRef context, const char *filename,
                                   unsigned line, unsigned column) {
  M::MLIRContext *ctx = unwrap(context);
  StringRef FileName(filename);
  M::Location loc = M::FileLineColLoc::get(FileName, line, column, ctx);
  return wrap(&loc);
}

M::Location ModuleBuilder::loc(Span span) {
  MLIRLocationRef fileLocRef = EIRSpanToMLIRLocation(span.start, span.end);
  M::Location *fileLoc = unwrap(fileLocRef);
  return *fileLoc;
}

//===----------------------------------------------------------------------===//
// Type Checking
//===----------------------------------------------------------------------===//

M::Value ModuleBuilder::build_is_type_op(M::Value value, M::Type matchType) {
  auto op = builder.create<IsTypeOp>(builder.getUnknownLoc(), value, matchType);
  return op.getResult();
}

MLIRValueRef MLIRBuildIsTypeTupleWithArity(MLIRModuleBuilderRef b, MLIRValueRef value, unsigned arity) {
  ModuleBuilder *builder = unwrap(b);
  M::Value val = unwrap(value);
  auto termType = builder->getType<TermType>();
  Shape shape(termType, arity);
  auto type = builder->getType<eir::TupleType>(shape);
  return wrap(builder->build_is_type_op(val, type));
}

#define DEFINE_IS_TYPE_OP(NAME, TYPE)                                   \
    MLIRValueRef NAME(MLIRModuleBuilderRef b, MLIRValueRef value) {    \
      ModuleBuilder *builder = unwrap(b);                               \
      M::Value val = unwrap(value);                                    \
      auto type = builder->getType<TYPE>();                             \
      return wrap(builder->build_is_type_op(val, type));                \
    }

DEFINE_IS_TYPE_OP(MLIRBuildIsTypeList, AnyListType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNonEmptyList, ConsType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNil, NilType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeMap, MapType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNumber, AnyNumberType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeFloat, AnyFloatType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeInteger, AnyIntegerType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeFixnum, FixnumType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeBigInt, BigIntType);
