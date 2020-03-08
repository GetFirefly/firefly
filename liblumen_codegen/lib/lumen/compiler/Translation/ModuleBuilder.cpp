#include "lumen/compiler/Translation/ModuleBuilder.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

using ::llvm::Optional;
using ::llvm::StringSwitch;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Builder, MLIRBuilderRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Location, MLIRLocationRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(mlir::Block, MLIRBlockRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(llvm::TargetMachine, LLVMTargetMachineRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(lumen::eir::FuncOp, MLIRFunctionOpRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(lumen::eir::ModuleBuilder,
                                   MLIRModuleBuilderRef);

inline Attribute unwrap(const void *P) {
  return Attribute::getFromOpaquePointer(P);
}

inline MLIRAttributeRef wrap(const Attribute &attr) {
  auto ptr = attr.getAsOpaquePointer();
  return reinterpret_cast<MLIRAttributeRef>(const_cast<void *>(ptr));
}

inline Value unwrap(MLIRValueRef v) { return Value::getFromOpaquePointer(v); }

inline MLIRValueRef wrap(const Value &attr) {
  auto ptr = attr.getAsOpaquePointer();
  return reinterpret_cast<MLIRValueRef>(const_cast<void *>(ptr));
}

namespace lumen {
namespace eir {

inline raw_ostream &operator<<(raw_ostream &os, EirTypeTag::TypeTag tag) {
  auto i = static_cast<uint32_t>(tag);
  os << "eir.term_kind(raw=" << i << ", val=";
  switch (tag) {
#define EIR_TERM_KIND(Name, Val) \
  case EirTypeTag::Name: {       \
    os << #Name;                 \
    break;                       \
  }
#define FIRST_EIR_TERM_KIND(Name, Val) EIR_TERM_KIND(Name, Val)
#include "lumen/compiler/Dialect/EIR/IR/EIREncoding.h.inc"
  }
  os << ")";
  return os;
}

static Type fromRust(Builder &builder, const EirType *wrapper) {
  EirTypeTag::TypeTag t = wrapper->any.tag;
  auto *context = builder.getContext();
  switch (t) {
    case EirTypeTag::None:
      return builder.getType<NoneType>();
    case EirTypeTag::Term:
      return builder.getType<TermType>();
    case EirTypeTag::List:
      return builder.getType<ListType>();
    case EirTypeTag::Number:
      return builder.getType<NumberType>();
    case EirTypeTag::Integer:
      return builder.getType<IntegerType>();
    case EirTypeTag::Float:
      return builder.getType<eir::FloatType>();
    case EirTypeTag::Atom:
      return builder.getType<AtomType>();
    case EirTypeTag::Boolean:
      return builder.getType<BooleanType>();
    case EirTypeTag::Fixnum:
      return builder.getType<FixnumType>();
    case EirTypeTag::BigInt:
      return builder.getType<BigIntType>();
    case EirTypeTag::Nil:
      return builder.getType<NilType>();
    case EirTypeTag::Cons:
      return builder.getType<ConsType>();
    case EirTypeTag::Tuple: {
      auto arity = wrapper->tuple.arity;
      return builder.getType<eir::TupleType>(arity);
    }
    case EirTypeTag::Closure:
      return builder.getType<ClosureType>();
    case EirTypeTag::Map:
      return builder.getType<MapType>();
    case EirTypeTag::Binary:
      return builder.getType<BinaryType>();
    case EirTypeTag::HeapBin:
      return builder.getType<HeapBinType>();
    case EirTypeTag::Box: {
      auto elementType = builder.getType<TermType>();
      return BoxType::get(elementType);
    }
    default:
      llvm::outs() << t << "\n";
      llvm::report_fatal_error("Unrecognized EirTypeTag.");
  }
}

Type ModuleBuilder::getArgType(const Arg *arg) {
  return fromRust(builder, &arg->ty);
}

bool unwrapValues(MLIRValueRef *argv, unsigned argc,
                  SmallVectorImpl<Value> &list) {
  if (argc < 1) {
    return false;
  }
  ArrayRef<MLIRValueRef> args(argv, argv + argc);
  for (auto it = args.begin(); it != args.end(); it++) {
    Value arg = unwrap(*it);
    assert(arg != nullptr);
    list.push_back(arg);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ModuleBuilder
//===----------------------------------------------------------------------===//

extern "C" MLIRModuleBuilderRef MLIRCreateModuleBuilder(
    MLIRContextRef context, const char *name, LLVMTargetMachineRef tm) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  StringRef moduleName(name);
  return wrap(new ModuleBuilder(*ctx, moduleName, targetMachine));
}

ModuleBuilder::ModuleBuilder(MLIRContext &context, StringRef name,
                             const TargetMachine *targetMachine)
    : builder(&context), targetMachine(targetMachine) {
  // Create an empty module into which we can codegen functions
  theModule = mlir::ModuleOp::create(builder.getUnknownLoc(), name);
  assert(isa<mlir::ModuleOp>(theModule) && "expected moduleop");
}

extern "C" void MLIRDumpModule(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  builder->dump();
}

void ModuleBuilder::dump() {
  if (theModule) theModule.dump();
}

ModuleBuilder::~ModuleBuilder() {
  if (theModule) theModule.erase();
}

extern "C" MLIRModuleRef MLIRFinalizeModuleBuilder(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  auto finished = builder->finish();
  delete builder;
  if (failed(mlir::verify(finished))) {
    finished.dump();
    finished.emitError("module verification error");
    return nullptr;
  }

  // Move to the heap
  return wrap(new mlir::ModuleOp(finished));
}

mlir::ModuleOp ModuleBuilder::finish() {
  mlir::ModuleOp finished;
  std::swap(finished, theModule);
  return finished;
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

extern "C" FunctionDeclResult MLIRCreateFunction(MLIRModuleBuilderRef b,
                                                 const char *name,
                                                 const Arg *argv, int argc,
                                                 EirType *type) {
  ModuleBuilder *builder = unwrap(b);
  StringRef functionName(name);
  llvm::SmallVector<Arg, 2> functionArgs(argv, argv + argc);
  auto fun = builder->create_function(functionName, functionArgs, type);
  if (!fun) return {nullptr, nullptr};

  MLIRFunctionOpRef funRef = wrap(new FuncOp(fun));
  FuncOp *tempFun = unwrap(funRef);
  auto *entryBlock = tempFun->addEntryBlock();
  builder->position_at_end(entryBlock);
  MLIRBlockRef entry = wrap(entryBlock);

  return {funRef, entry};
}

FuncOp ModuleBuilder::create_function(StringRef functionName,
                                      SmallVectorImpl<Arg> &functionArgs,
                                      EirType *resultType) {
  llvm::SmallVector<Type, 2> argTypes;
  argTypes.reserve(functionArgs.size());
  for (auto it = functionArgs.begin(); it != functionArgs.end(); it++) {
    Type type = getArgType(it);
    if (!type) return nullptr;
    argTypes.push_back(type);
  }
  ArrayRef<NamedAttribute> attrs({});
  if (resultType->any.tag == EirTypeTag::None) {
    auto fnType = builder.getFunctionType(argTypes, llvm::None);
    return FuncOp::create(builder.getUnknownLoc(), functionName, fnType, attrs);
  } else {
    auto fnType =
        builder.getFunctionType(argTypes, fromRust(builder, resultType));
    return FuncOp::create(builder.getUnknownLoc(), functionName, fnType, attrs);
  }
}

void ModuleBuilder::declare_function(StringRef functionName,
                                     mlir::FunctionType fnType) {
  ArrayRef<NamedAttribute> attrs({});
  builder.create<FuncOp>(builder.getUnknownLoc(), functionName, fnType, attrs);
}

extern "C" void MLIRAddFunction(MLIRModuleBuilderRef b, MLIRFunctionOpRef f) {
  ModuleBuilder *builder = unwrap(b);
  FuncOp *fun = unwrap(f);
  builder->add_function(*fun);
}

void ModuleBuilder::add_function(FuncOp f) { theModule.push_back(f); }

//===----------------------------------------------------------------------===//
// Blocks
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRGetCurrentBlockArgument(MLIRModuleBuilderRef b,
                                                    unsigned id) {
  ModuleBuilder *builder = unwrap(b);
  Block *block = builder->getBlock();
  assert(block != nullptr);
  return wrap(block->getArgument(id));
}

Block *ModuleBuilder::getBlock() { return builder.getBlock(); }

extern "C" MLIRValueRef MLIRGetBlockArgument(MLIRBlockRef b, unsigned id) {
  Block *block = unwrap(b);
  Value arg = block->getArgument(id);
  return wrap(arg);
}

extern "C" MLIRBlockRef MLIRAppendBasicBlock(MLIRModuleBuilderRef b,
                                             MLIRFunctionOpRef f,
                                             const Arg *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  FuncOp *fun = unwrap(f);
  auto *block = builder->add_block(*fun);
  if (!block) return nullptr;

  builder->position_at_end(block);

  if (argc > 0) {
    ArrayRef<Arg> args(argv, argv + argc);
    for (auto it = args.begin(); it != args.end(); ++it) {
      Type type = builder->getArgType(it);
      block->addArgument(type);
    }
  }
  assert((block->getNumArguments() == argc) &&
         "number of block arguments doesn't match requested arity!");
  return wrap(block);
}

Block *ModuleBuilder::add_block(FuncOp &f) { return f.addBlock(); }

extern "C" void MLIRBlockPositionAtEnd(MLIRModuleBuilderRef b,
                                       MLIRBlockRef blk) {
  ModuleBuilder *builder = unwrap(b);
  Block *block = unwrap(blk);
  builder->position_at_end(block);
}

void ModuleBuilder::position_at_end(Block *block) {
  builder.setInsertionPointToEnd(block);
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildBr(MLIRModuleBuilderRef b, MLIRBlockRef destBlk,
                            MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Block *dest = unwrap(destBlk);
  if (argc > 0) {
    llvm::SmallVector<Value, 2> args;
    unwrapValues(argv, argc, args);
    builder->build_br(dest, args);
  } else {
    builder->build_br(dest);
  }
}

void ModuleBuilder::build_br(Block *dest, ValueRange destArgs) {
  builder.create<BranchOp>(builder.getUnknownLoc(), dest, destArgs);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildIf(MLIRModuleBuilderRef b, MLIRValueRef val,
                            MLIRBlockRef y, MLIRValueRef *yArgv, unsigned yArgc,
                            MLIRBlockRef n, MLIRValueRef *nArgv, unsigned nArgc,
                            MLIRBlockRef o, MLIRValueRef *oArgv,
                            unsigned oArgc) {
  ModuleBuilder *builder = unwrap(b);
  Value value = unwrap(val);
  Block *yes = unwrap(y);
  Block *no = unwrap(n);
  Block *other = unwrap(o);
  // Unwrap block args
  SmallVector<Value, 1> yesArgs;
  unwrapValues(yArgv, yArgc, yesArgs);
  SmallVector<Value, 1> noArgs;
  unwrapValues(nArgv, nArgc, noArgs);
  SmallVector<Value, 1> otherArgs;
  unwrapValues(oArgv, oArgc, otherArgs);
  // Construct operation
  builder->build_if(value, yes, no, other, yesArgs, noArgs, otherArgs);
}

void ModuleBuilder::build_if(Value value, Block *yes, Block *no, Block *other,
                             SmallVectorImpl<Value> &yesArgs,
                             SmallVectorImpl<Value> &noArgs,
                             SmallVectorImpl<Value> &otherArgs) {
  //  Create the `if`, if necessary
  bool withOtherwiseRegion = other != nullptr;
  Value isTrue = value;
  if (!value.getType().isa<BooleanType>()) {
    // The condition is not boolean, so we need to do a comparison
    auto trueConst =
        builder.create<ConstantAtomOp>(builder.getUnknownLoc(), true);
    isTrue = builder.create<CmpEqOp>(builder.getUnknownLoc(), value, trueConst,
                                     /*strict=*/true);
  }

  if (!other) {
    // No need to do any additional comparisons
    builder.create<CondBranchOp>(builder.getUnknownLoc(), isTrue, yes, yesArgs,
                                 no, noArgs);
    return;
  }

  // Otherwise we need an additional check to see if we use the otherwise branch
  auto falseConst =
      builder.create<ConstantAtomOp>(builder.getUnknownLoc(), false);
  Value isFalse = builder.create<CmpEqOp>(builder.getUnknownLoc(), value,
                                          falseConst, /*strict=*/true);

  Block *currentBlock = builder.getBlock();
  Block *falseBlock = currentBlock->splitBlock(falseConst);

  builder.setInsertionPointToEnd(falseBlock);
  builder.create<CondBranchOp>(builder.getUnknownLoc(), isFalse, no, noArgs,
                               other, otherArgs);

  // Go back to original block and insert conditional branch for first
  // comparison
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<CondBranchOp>(builder.getUnknownLoc(), isTrue, yes, yesArgs,
                               falseBlock, ArrayRef<Value>{});
  return;
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildMatchOp(MLIRModuleBuilderRef b, eir::Match op) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_match(op);
}

std::unique_ptr<MatchPattern> ModuleBuilder::convertMatchPattern(
    const MLIRMatchPattern &inPattern) {
  auto tag = inPattern.tag;
  switch (tag) {
    default:
      llvm_unreachable("unrecognized match pattern tag!");
    case MatchPatternType::Any:
      return std::unique_ptr<AnyPattern>(new AnyPattern());
    case MatchPatternType::Cons:
      return std::unique_ptr<ConsPattern>(new ConsPattern());
    case MatchPatternType::Tuple:
      return std::unique_ptr<TuplePattern>(
          new TuplePattern(inPattern.payload.i));
    case MatchPatternType::MapItem:
      return std::unique_ptr<MapPattern>(
          new MapPattern(unwrap(inPattern.payload.v)));
    case MatchPatternType::IsType: {
      auto t = &inPattern.payload.t;
      return std::unique_ptr<IsTypePattern>(
          new IsTypePattern(fromRust(builder, t)));
    }
    case MatchPatternType::Value:
      return std::unique_ptr<ValuePattern>(
          new ValuePattern(unwrap(inPattern.payload.v)));
    case MatchPatternType::Binary:
      auto payload = inPattern.payload.b;
      auto sizePtr = payload.size;
      if (sizePtr == nullptr) {
        return std::unique_ptr<BinaryPattern>(
            new BinaryPattern(inPattern.payload.b.spec));
      } else {
        Value size = unwrap(sizePtr);
        return std::unique_ptr<BinaryPattern>(
            new BinaryPattern(inPattern.payload.b.spec, size));
      }
  }
}

void ModuleBuilder::build_match(Match op) {
  // Convert FFI types into internal MLIR representation
  Value selector = unwrap(op.selector);
  SmallVector<MatchBranch, 2> branches;

  ArrayRef<MLIRMatchBranch> inBranches(op.branches,
                                       op.branches + op.numBranches);
  for (auto &inBranch : inBranches) {
    // Extract destination block and base arguments
    Block *dest = unwrap(inBranch.dest);
    ArrayRef<MLIRValueRef> inDestArgs(inBranch.destArgv,
                                      inBranch.destArgv + inBranch.destArgc);
    SmallVector<Value, 1> destArgs;
    for (auto argRef : inDestArgs) {
      Value arg = unwrap(argRef);
      destArgs.push_back(arg);
    }
    // Convert match pattern payload
    auto pattern = convertMatchPattern(inBranch.pattern);
    // Create internal branch type
    MatchBranch branch(dest, destArgs, std::move(pattern));
    branches.push_back(std::move(branch));
  }

  // We don't use an explicit operation for matches, as currently
  // there isn't enough structure in place to allow nested regions
  // to reference blocks from containing ops
  lumen::eir::lowerPatternMatch(builder, selector, branches);
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildUnreachable(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_unreachable();
}

void ModuleBuilder::build_unreachable() {
  builder.create<UnreachableOp>(builder.getUnknownLoc());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildReturn(MLIRModuleBuilderRef b, MLIRValueRef value) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_return(unwrap(value));
}

void ModuleBuilder::build_return(Value value) {
  if (!value) {
    builder.create<ReturnOp>(builder.getUnknownLoc());
  } else {
    builder.create<ReturnOp>(builder.getUnknownLoc(), value);
  }
}

//===----------------------------------------------------------------------===//
// TraceCaptureOp/TraceConstructOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildTraceCaptureOp(MLIRModuleBuilderRef b, MLIRBlockRef d,
                                        MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Block *dest = unwrap(d);
  if (argc > 0) {
    ArrayRef<MLIRValueRef> args(argv, argv + argc);
    builder->build_trace_capture_op(dest, args);
  } else {
    builder->build_trace_capture_op(dest);
  }
}

void ModuleBuilder::build_trace_capture_op(Block *dest,
                                           ArrayRef<MLIRValueRef> destArgs) {
  auto termType = TermType::get(builder.getContext());
  auto captureOp =
      builder.create<TraceCaptureOp>(builder.getUnknownLoc(), termType);
  auto capture = captureOp.getResult();

  SmallVector<Value, 1> extendedArgs;
  extendedArgs.push_back(capture);
  for (auto destArg : destArgs) {
    Value arg = unwrap(destArg);
    extendedArgs.push_back(arg);
  }

  builder.create<BranchOp>(builder.getUnknownLoc(), dest, extendedArgs);
}

extern "C" MLIRValueRef MLIRBuildTraceConstructOp(MLIRModuleBuilderRef,
                                                  MLIRValueRef) {
  // TODO: For now we do nothing, but we'll want code that fetches
  // a term value representing the stack trace
  return nullptr;
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildMapOp(MLIRModuleBuilderRef b, MapUpdate op) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_map_update(op);
}

void ModuleBuilder::build_map_update(MapUpdate op) {
  assert(op.actionsc > 0 && "cannot construct empty map op");
  SmallVector<MapAction, 2> actions(op.actionsv, op.actionsv + op.actionsc);
  Value map = unwrap(op.map);
  Block *ok = unwrap(op.ok);
  Block *err = unwrap(op.err);
  // Each insert or update implicitly branches to a continuation block for the
  // next insert/update; the last continuation block simply branches
  // unconditionally to the ok block
  Block *current = builder.getInsertionBlock();
  Region *parent = current->getParent();
  for (auto it = actions.begin(); it + 1 != actions.end(); ++it) {
    MapAction action = *it;
    Value key = unwrap(action.key);
    Value val = unwrap(action.value);
    // Create the continuation block, which expects the updated map as an
    // argument
    Block *cont = builder.createBlock(parent);
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
        llvm::report_fatal_error(
            "tried to construct map update op with invalid type");
    }
    current = cont;
    // We need to update the `map` pointer, since we're implicitly in a new
    // block on the next iteration
    map = current->getArgument(0);
  }
  // After all updates, we can unconditionally branch to `ok`, since no errors
  // could have occurred
  builder.setInsertionPointToEnd(current);
  ArrayRef<Value> okArgs = {map};
  build_br(ok, okArgs);
}

void ModuleBuilder::build_map_insert_op(Value map, Value key, Value val,
                                        Block *ok, Block *err) {
  // Perform the insert
  ArrayRef<Value> pairs = {key, val};
  auto op = builder.create<MapInsertOp>(builder.getUnknownLoc(), map, pairs);
  // Get the results, which is the updated map, and a error condition flag
  Value newMap = op.getResult(0);
  assert(newMap != nullptr);
  Value isOk = op.getResult(1);
  assert(isOk != nullptr);
  // Then branch to either the ok block, or the error block
  ValueRange okArgs = {newMap};
  ValueRange errArgs = {};
  builder.create<CondBranchOp>(builder.getUnknownLoc(), isOk, ok, okArgs, err,
                               errArgs);
}

void ModuleBuilder::build_map_update_op(Value map, Value key, Value val,
                                        Block *ok, Block *err) {
  // Perform the update
  ArrayRef<Value> pairs = {key, val};
  auto op = builder.create<MapUpdateOp>(builder.getUnknownLoc(), map, pairs);
  // Get the results, which is the updated map, and a error condition flag
  Value newMap = op.getResult(0);
  assert(newMap != nullptr);
  Value isOk = op.getResult(1);
  assert(isOk != nullptr);
  // Then branch to either the ok block, or the error block
  ValueRange okArgs = {newMap};
  ValueRange errArgs = {};
  builder.create<CondBranchOp>(builder.getUnknownLoc(), isOk, ok, okArgs, err,
                               errArgs);
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildIsEqualOp(MLIRModuleBuilderRef b,
                                           MLIRValueRef l, MLIRValueRef r,
                                           bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_equal(lhs, rhs, isExact));
}

Value ModuleBuilder::build_is_equal(Value lhs, Value rhs, bool isExact) {
  auto op = builder.create<CmpEqOp>(builder.getUnknownLoc(), lhs, rhs, isExact);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildIsNotEqualOp(MLIRModuleBuilderRef b,
                                              MLIRValueRef l, MLIRValueRef r,
                                              bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_not_equal(lhs, rhs, isExact));
}

Value ModuleBuilder::build_is_not_equal(Value lhs, Value rhs, bool isExact) {
  auto op =
      builder.create<CmpNeqOp>(builder.getUnknownLoc(), lhs, rhs, isExact);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildLessThanOrEqualOp(MLIRModuleBuilderRef b,
                                                   MLIRValueRef l,
                                                   MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than_or_equal(lhs, rhs));
}

Value ModuleBuilder::build_is_less_than_or_equal(Value lhs, Value rhs) {
  auto op = builder.create<CmpLteOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildLessThanOp(MLIRModuleBuilderRef b,
                                            MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than(lhs, rhs));
}

Value ModuleBuilder::build_is_less_than(Value lhs, Value rhs) {
  auto op = builder.create<CmpLtOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildGreaterThanOrEqualOp(MLIRModuleBuilderRef b,
                                                      MLIRValueRef l,
                                                      MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than_or_equal(lhs, rhs));
}

Value ModuleBuilder::build_is_greater_than_or_equal(Value lhs, Value rhs) {
  auto op = builder.create<CmpGteOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildGreaterThanOp(MLIRModuleBuilderRef b,
                                               MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than(lhs, rhs));
}

Value ModuleBuilder::build_is_greater_than(Value lhs, Value rhs) {
  auto op = builder.create<CmpGtOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Logical Operators
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildLogicalAndOp(MLIRModuleBuilderRef b,
                                              MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_logical_and(lhs, rhs));
}

Value ModuleBuilder::build_logical_and(Value lhs, Value rhs) {
  auto op = builder.create<LogicalAndOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildLogicalOrOp(MLIRModuleBuilderRef b,
                                             MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_logical_or(lhs, rhs));
}

Value ModuleBuilder::build_logical_or(Value lhs, Value rhs) {
  auto op = builder.create<LogicalOrOp>(builder.getUnknownLoc(), lhs, rhs);
  return op.getResult();
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

#define INTRINSIC_BUILDER(Alias, Op)                             \
  static Value Alias(OpBuilder &builder, ArrayRef<Value> args) { \
    auto op = builder.create<Op>(builder.getUnknownLoc(), args); \
                                                                 \
    return op.getResult();                                       \
  }

INTRINSIC_BUILDER(buildIntrinsicPrintOp, PrintOp);
INTRINSIC_BUILDER(buildIntrinsicAddOp, AddOp);
INTRINSIC_BUILDER(buildIntrinsicSubOp, SubOp);
INTRINSIC_BUILDER(buildIntrinsicMulOp, MulOp);
INTRINSIC_BUILDER(buildIntrinsicDivOp, DivOp);
INTRINSIC_BUILDER(buildIntrinsicRemOp, RemOp);
INTRINSIC_BUILDER(buildIntrinsicFDivOp, FDivOp);
INTRINSIC_BUILDER(buildIntrinsicBslOp, BslOp);
INTRINSIC_BUILDER(buildIntrinsicBsrOp, BsrOp);
INTRINSIC_BUILDER(buildIntrinsicBandOp, BandOp);
// INTRINSIC_BUILDER(buildIntrinsicBnotOp, BnotOp);
INTRINSIC_BUILDER(buildIntrinsicBorOp, BorOp);
INTRINSIC_BUILDER(buildIntrinsicBxorOp, BxorOp);
INTRINSIC_BUILDER(buildIntrinsicCmpEqOp, CmpEqOp);
INTRINSIC_BUILDER(buildIntrinsicCmpNeqOp, CmpNeqOp);
INTRINSIC_BUILDER(buildIntrinsicCmpLtOp, CmpLtOp);
INTRINSIC_BUILDER(buildIntrinsicCmpLteOp, CmpLteOp);
INTRINSIC_BUILDER(buildIntrinsicCmpGtOp, CmpGtOp);
INTRINSIC_BUILDER(buildIntrinsicCmpGteOp, CmpGteOp);

static Value buildIntrinsicCmpEqStrictOp(OpBuilder &builder,
                                         ArrayRef<Value> args) {
  assert(args.size() == 2 && "expected =:= operator to receive two operands");
  Value lhs = args[0];
  Value rhs = args[1];
  auto op = builder.create<CmpEqOp>(builder.getUnknownLoc(), lhs, rhs,
                                    /*strict=*/true);
}

static Value buildIntrinsicCmpNeqStrictOp(OpBuilder &builder,
                                          ArrayRef<Value> args) {
  assert(args.size() == 2 && "expected =/= operator to receive two operands");
  Value lhs = args[0];
  Value rhs = args[1];
  auto op = builder.create<CmpNeqOp>(builder.getUnknownLoc(), lhs, rhs,
                                     /*strict=*/true);
}

using BuildIntrinsicFnT = Value (*)(OpBuilder &, ArrayRef<Value>);

static Optional<BuildIntrinsicFnT> getIntrinsicBuilder(StringRef target) {
  auto fnPtr = StringSwitch<BuildIntrinsicFnT>(target)
                   .Case("erlang:print/1", buildIntrinsicPrintOp)
                   .Case("erlang:+/2", buildIntrinsicAddOp)
                   .Case("erlang:-/2", buildIntrinsicSubOp)
                   .Case("erlang:*/2", buildIntrinsicMulOp)
                   .Case("erlang:div/2", buildIntrinsicDivOp)
                   .Case("erlang:rem/2", buildIntrinsicRemOp)
                   .Case("erlang://2", buildIntrinsicFDivOp)
                   .Case("erlang:bsl/2", buildIntrinsicBslOp)
                   .Case("erlang:bsr/2", buildIntrinsicBsrOp)
                   .Case("erlang:band/2", buildIntrinsicBandOp)
                   //.Case("erlang:bnot/2", buildIntrinsicBnotOp)
                   .Case("erlang:bor/2", buildIntrinsicBorOp)
                   .Case("erlang:bxor/2", buildIntrinsicBxorOp)
                   .Case("erlang:=:=/2", buildIntrinsicCmpEqStrictOp)
                   .Case("erlang:=/=/2", buildIntrinsicCmpNeqStrictOp)
                   .Case("erlang:==/2", buildIntrinsicCmpEqOp)
                   .Case("erlang:/=/2", buildIntrinsicCmpNeqOp)
                   .Case("erlang:</2", buildIntrinsicCmpLtOp)
                   .Case("erlang:=</2", buildIntrinsicCmpLteOp)
                   .Case("erlang:>/2", buildIntrinsicCmpGtOp)
                   .Case("erlang:>=/2", buildIntrinsicCmpGteOp)
                   .Default(nullptr);
  if (fnPtr == nullptr) {
    return llvm::None;
  } else {
    return fnPtr;
  }
}

extern "C" bool MLIRIsIntrinsic(const char *name) {
  return getIntrinsicBuilder(name).hasValue();
}

extern "C" MLIRValueRef MLIRBuildIntrinsic(MLIRModuleBuilderRef b,
                                           const char *name, MLIRValueRef *argv,
                                           unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  auto buildIntrinsicFnOpt = getIntrinsicBuilder(name);
  if (!buildIntrinsicFnOpt) return nullptr;

  SmallVector<Value, 1> args;
  unwrapValues(argv, argc, args);
  auto buildIntrinsicFn = buildIntrinsicFnOpt.getValue();
  return wrap(buildIntrinsicFn(builder->getBuilder(), args));
}

extern "C" void MLIRBuildStaticCall(MLIRModuleBuilderRef b, const char *name,
                                    MLIRValueRef *argv, unsigned argc,
                                    bool isTail, MLIRBlockRef okBlock,
                                    MLIRValueRef *okArgv, unsigned okArgc,
                                    MLIRBlockRef errBlock,
                                    MLIRValueRef *errArgv, unsigned errArgc) {
  ModuleBuilder *builder = unwrap(b);
  Block *ok = unwrap(okBlock);
  Block *err = unwrap(errBlock);
  StringRef functionName(name);
  SmallVector<Value, 2> args;
  unwrapValues(argv, argc, args);
  SmallVector<Value, 1> okArgs;
  unwrapValues(okArgv, okArgc, okArgs);
  SmallVector<Value, 1> errArgs;
  unwrapValues(errArgv, errArgc, errArgs);
  builder->build_static_call(functionName, args, isTail, ok, okArgs, err,
                             errArgs);
}

void ModuleBuilder::build_static_call(StringRef target, ArrayRef<Value> args,
                                      bool isTail, Block *ok,
                                      ArrayRef<Value> okArgs, Block *err,
                                      ArrayRef<Value> errArgs) {
  // If this is a call to an intrinsic, lower accordingly
  auto buildIntrinsicFnOpt = getIntrinsicBuilder(target);
  Value result;
  if (buildIntrinsicFnOpt.hasValue()) {
    auto buildIntrinsicFn = buildIntrinsicFnOpt.getValue();
    auto result = buildIntrinsicFn(builder, args);
    if (isTail) {
      if (result) {
        builder.create<ReturnOp>(builder.getUnknownLoc(), result);
      } else {
        builder.create<ReturnOp>(builder.getUnknownLoc());
      }
    } else {
      if (result) {
        auto termTy = builder.getType<TermType>();
        auto boolResult =
            builder.create<CastOp>(builder.getUnknownLoc(), result, termTy);
        builder.create<BranchOp>(builder.getUnknownLoc(), ok,
                                 ArrayRef<Value>{boolResult});
      } else {
        builder.create<BranchOp>(builder.getUnknownLoc(), ok,
                                 ArrayRef<Value>{});
      }
    }
    return;
  }

  // Create symbolref and lookup function definition (if present)
  auto callee = builder.getSymbolRefAttr(target);
  auto fn = theModule.lookupSymbol<FuncOp>(callee.getValue());

  // Build result types list
  SmallVector<Type, 1> fnResults;
  if (fn) {
    auto rs = fn.getCallableResults();
    fnResults.append(rs.begin(), rs.end());
  } else {
    auto termType = builder.getType<TermType>();
    SmallVector<Type, 1> argTypes;
    for (auto arg : args) {
      argTypes.push_back(arg.getType());
    }
    fnResults.push_back(termType);
  }

  // Build call
  Operation *call =
      builder.create<CallOp>(builder.getUnknownLoc(), callee, fnResults, args);
  assert(call->getNumResults() == 1 && "unsupported number of results");
  result = call->getResult(0);

  // If this is a tail call, we're returning the results directly
  if (isTail) {
    if (result) {
      builder.create<ReturnOp>(builder.getUnknownLoc(), result);
    } else {
      builder.create<ReturnOp>(builder.getUnknownLoc());
    }
    return;
  }

  auto currentBlock = builder.getBlock();
  auto currentRegion = currentBlock->getParent();

  // It isn't possible at this point to have neither ok or err blocks
  assert(((!ok && !err) == false) &&
         "expected isTail when no ok/error destination provided");
  // In addition to any block arguments, we have to append the call results
  SmallVector<Value, 1> okArgsFinal;
  SmallVector<Value, 1> errArgsFinal;

  Value rhs = builder.create<ConstantNoneOp>(builder.getUnknownLoc());
  Value isErr = builder.create<CmpEqOp>(builder.getUnknownLoc(), result, rhs,
                                        /*strict=*/true);
  Block *ifOk = ok;
  Block *ifErr = err;

  // No success block provided, so create block to handle returning
  if (!ok) {
    auto ret = builder.create<ReturnOp>(builder.getUnknownLoc(), result);
    ifOk = currentBlock->splitBlock(ret);
    builder.setInsertionPointToEnd(currentBlock);
  } else {
    okArgsFinal.push_back(result);
  }

  // No error block provided, so create block to handle throwing
  if (!err) {
    auto thr = builder.create<ThrowOp>(builder.getUnknownLoc(), result);
    ifErr = currentBlock->splitBlock(thr);
    builder.setInsertionPointToEnd(currentBlock);
  } else {
    errArgsFinal.append(errArgs.begin(), errArgs.end());
    errArgsFinal.push_back(result);
  }

  // Insert conditional branch in call block
  // Use comparison to NONE as condition for cond_br
  builder.create<CondBranchOp>(builder.getUnknownLoc(), isErr, ifErr,
                               errArgsFinal, ifOk, okArgsFinal);
  return;
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRCons(MLIRModuleBuilderRef b, MLIRValueRef h,
                                 MLIRValueRef t) {
  ModuleBuilder *builder = unwrap(b);
  Value head = unwrap(h);
  Value tail = unwrap(t);
  return wrap(builder->build_cons(head, tail));
}

Value ModuleBuilder::build_cons(Value head, Value tail) {
  auto op = builder.create<ConsOp>(builder.getUnknownLoc(), head, tail);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRConstructTuple(MLIRModuleBuilderRef b,
                                           MLIRValueRef *ev, unsigned ec) {
  ModuleBuilder *builder = unwrap(b);
  SmallVector<Value, 2> elements;
  unwrapValues(ev, ec, elements);
  return wrap(builder->build_tuple(elements));
}

Value ModuleBuilder::build_tuple(ArrayRef<Value> elements) {
  auto op = builder.create<TupleOp>(builder.getUnknownLoc(), elements);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRConstructMap(MLIRModuleBuilderRef b, MapEntry *ev,
                                         unsigned ec) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MapEntry> entries(ev, ev + ec);
  return wrap(builder->build_map(entries));
}

Value ModuleBuilder::build_map(ArrayRef<MapEntry> entries) {
  auto op = builder.create<ConstructMapOp>(builder.getUnknownLoc(), entries);
  return op.getResult(0);
}

//===----------------------------------------------------------------------===//
// ConstantFloat
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantFloat(MLIRModuleBuilderRef b,
                                               double value) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_float(value));
}

Value ModuleBuilder::build_constant_float(double value) {
  auto type = builder.getType<FloatType>();
  APFloat f(value);
  auto op = builder.create<ConstantFloatOp>(builder.getUnknownLoc(), f);
  return op.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildFloatAttr(MLIRModuleBuilderRef b,
                                               double value) {
  ModuleBuilder *builder = unwrap(b);
  auto type = builder->getType<FloatType>();
  return wrap(builder->build_float_attr(type, value));
}

Attribute ModuleBuilder::build_float_attr(Type type, double value) {
  return builder.getFloatAttr(type, value);
}

//===----------------------------------------------------------------------===//
// ConstantInt
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantInt(MLIRModuleBuilderRef b,
                                             int64_t value) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_int(value));
}

Value ModuleBuilder::build_constant_int(int64_t value) {
  auto op = builder.create<ConstantIntOp>(builder.getUnknownLoc(), value);
  return op.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildIntAttr(MLIRModuleBuilderRef b,
                                             int64_t value) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_int_attr(value, /*signed=*/true));
}

Attribute ModuleBuilder::build_int_attr(int64_t value, bool isSigned) {
  auto type = builder.getIntegerType(64);
  APInt i(64, value, isSigned);
  return builder.getIntegerAttr(type, i);
}

//===----------------------------------------------------------------------===//
// ConstantBigInt
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantBigInt(MLIRModuleBuilderRef b,
                                                const char *str,
                                                unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_constant_bigint(value, width));
}

Value ModuleBuilder::build_constant_bigint(StringRef value, unsigned width) {
  APInt i(width, value, /*radix=*/10);
  auto op = builder.create<ConstantBigIntOp>(builder.getUnknownLoc(), i);
  return op.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildBigIntAttr(MLIRModuleBuilderRef b,
                                                const char *str,
                                                unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_bigint_attr(value, width));
}

Attribute ModuleBuilder::build_bigint_attr(StringRef value, unsigned width) {
  auto type = builder.getType<BigIntType>();
  APInt i(width, value, /*radix=*/10);
  return builder.getIntegerAttr(type, i);
}

//===----------------------------------------------------------------------===//
// ConstantAtom
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantAtom(MLIRModuleBuilderRef b,
                                              const char *str, uint64_t id) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_constant_atom(value, id));
}

Value ModuleBuilder::build_constant_atom(StringRef value, uint64_t valueId) {
  APInt id(64, valueId, /*isSigned=*/false);
  auto op = builder.create<ConstantAtomOp>(builder.getUnknownLoc(), id, value);
  auto result = op.getResult();
  auto termTy = builder.getType<TermType>();
  auto castOp = builder.create<CastOp>(builder.getUnknownLoc(), result, termTy);
  return castOp.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildAtomAttr(MLIRModuleBuilderRef b,
                                              const char *str, uint64_t id) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_atom_attr(value, id));
}

Attribute ModuleBuilder::build_atom_attr(StringRef value, uint64_t valueId) {
  APInt id(64, valueId, /*isSigned=*/false);
  return AtomAttr::get(builder.getContext(), id, value);
}

Attribute ModuleBuilder::build_string_attr(StringRef value) {
  return builder.getStringAttr(value);
}

//===----------------------------------------------------------------------===//
// ConstantBinary
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantBinary(MLIRModuleBuilderRef b,
                                                const char *str, unsigned size,
                                                uint64_t header,
                                                uint64_t flags) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str, (size_t)size);
  return wrap(builder->build_constant_binary(value, header, flags));
}

Value ModuleBuilder::build_constant_binary(StringRef value, uint64_t header,
                                           uint64_t flags) {
  auto op = builder.create<ConstantBinaryOp>(builder.getUnknownLoc(), value,
                                             header, flags);
  return op.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildBinaryAttr(MLIRModuleBuilderRef b,
                                                const char *str, unsigned size,
                                                uint64_t header,
                                                uint64_t flags) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str, (size_t)size);
  return wrap(builder->build_binary_attr(value, header, flags));
}

Attribute ModuleBuilder::build_binary_attr(StringRef value, uint64_t header,
                                           uint64_t flags) {
  return BinaryAttr::get(builder.getContext(), value, header, flags);
}

//===----------------------------------------------------------------------===//
// ConstantNil
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantNil(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_constant_nil());
}

Value ModuleBuilder::build_constant_nil() {
  auto op = builder.create<ConstantNilOp>(builder.getUnknownLoc());
  return op.getResult();
}

extern "C" MLIRAttributeRef MLIRBuildNilAttr(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_nil_attr());
}

Attribute ModuleBuilder::build_nil_attr() {
  return mlir::TypeAttr::get(builder.getType<NilType>());
}

//===----------------------------------------------------------------------===//
// ConstantSeq (List/Tuple/Map)
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantList(MLIRModuleBuilderRef b,
                                              const MLIRAttributeRef *elements,
                                              int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  SmallVector<Attribute, 1> list;
  if (num_elements > 0) {
    for (auto ar : xs) {
      Attribute attr = unwrap(ar);
      list.push_back(attr);
    }
    return wrap(builder->build_constant_list(list));
  } else {
    return wrap(builder->build_constant_nil());
  }
}

Value ModuleBuilder::build_constant_list(ArrayRef<Attribute> elements) {
  auto op = builder.create<ConstantListOp>(builder.getUnknownLoc(), elements);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildConstantTuple(MLIRModuleBuilderRef b,
                                               const MLIRAttributeRef *elements,
                                               int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  SmallVector<Attribute, 1> tuple;
  for (auto ar : xs) {
    Attribute attr = unwrap(ar);
    tuple.push_back(attr);
  }
  return wrap(builder->build_constant_tuple(tuple));
}

Value ModuleBuilder::build_constant_tuple(ArrayRef<Attribute> elements) {
  auto op = builder.create<ConstantTupleOp>(builder.getUnknownLoc(), elements);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildConstantMap(MLIRModuleBuilderRef b,
                                             const KeyValuePair *elements,
                                             int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<KeyValuePair> xs(elements, elements + num_elements);
  SmallVector<Attribute, 4> list;
  list.reserve(xs.size() * 2);
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    Attribute key = unwrap(it->key);
    if (!key) return nullptr;
    list.push_back(key);
    Attribute value = unwrap(it->value);
    if (!value) return nullptr;
    list.push_back(value);
  }
  return wrap(builder->build_constant_map(list));
}

Value ModuleBuilder::build_constant_map(ArrayRef<Attribute> elements) {
  auto op = builder.create<ConstantMapOp>(builder.getUnknownLoc(), elements);
  return op.getResult();
}

Attribute build_seq_attr(ModuleBuilder *builder,
                         ArrayRef<MLIRAttributeRef> elements, Type type) {
  SmallVector<Attribute, 3> list;
  list.reserve(elements.size());
  for (auto it = elements.begin(); it + 1 != elements.end(); ++it) {
    Attribute attr = unwrap(*it);
    if (!attr) return nullptr;
    list.push_back(attr);
  }
  return builder->build_seq_attr(list, type);
}

Attribute ModuleBuilder::build_seq_attr(ArrayRef<Attribute> elements,
                                        Type type) {
  return SeqAttr::get(type, elements);
}

extern "C" MLIRAttributeRef MLIRBuildListAttr(MLIRModuleBuilderRef b,
                                              const MLIRAttributeRef *elements,
                                              int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  auto type = builder->getType<ConsType>();

  return wrap(build_seq_attr(builder, xs, type));
}

extern "C" MLIRAttributeRef MLIRBuildTupleAttr(MLIRModuleBuilderRef b,
                                               const MLIRAttributeRef *elements,
                                               int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);

  std::vector<Type> types;
  types.reserve(xs.size());
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    Attribute attr = unwrap(*it);
    if (!attr) return nullptr;
    types.push_back(attr.getType());
  }
  auto type = eir::TupleType::get(ArrayRef(types));

  return wrap(build_seq_attr(builder, xs, type));
}

extern "C" MLIRAttributeRef MLIRBuildMapAttr(MLIRModuleBuilderRef b,
                                             const KeyValuePair *elements,
                                             int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<KeyValuePair> xs(elements, elements + num_elements);
  SmallVector<Attribute, 4> list;
  list.reserve(xs.size() * 2);
  for (auto it = xs.begin(); it + 1 != xs.end(); ++it) {
    Attribute key = unwrap(it->key);
    if (!key) return nullptr;
    list.push_back(key);
    Attribute value = unwrap(it->value);
    if (!value) return nullptr;
    list.push_back(value);
  }
  auto type = builder->getType<MapType>();
  return wrap(builder->build_seq_attr(list, type));
}

//===----------------------------------------------------------------------===//
// Locations/Spans
//===----------------------------------------------------------------------===//

extern "C" MLIRLocationRef MLIRCreateLocation(MLIRContextRef context,
                                              const char *filename,
                                              unsigned line, unsigned column) {
  MLIRContext *ctx = unwrap(context);
  StringRef FileName(filename);
  Location loc = mlir::FileLineColLoc::get(FileName, line, column, ctx);
  return wrap(&loc);
}

//===----------------------------------------------------------------------===//
// Type Checking
//===----------------------------------------------------------------------===//

Value ModuleBuilder::build_is_type_op(Value value, Type matchType) {
  auto op = builder.create<IsTypeOp>(builder.getUnknownLoc(), value, matchType);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildIsTypeTupleWithArity(MLIRModuleBuilderRef b,
                                                      MLIRValueRef value,
                                                      unsigned arity) {
  ModuleBuilder *builder = unwrap(b);
  Value val = unwrap(value);
  auto type = builder->getType<eir::TupleType>(arity);
  return wrap(builder->build_is_type_op(val, type));
}

#define DEFINE_IS_TYPE_OP(NAME, TYPE)                                        \
  extern "C" MLIRValueRef NAME(MLIRModuleBuilderRef b, MLIRValueRef value) { \
    ModuleBuilder *builder = unwrap(b);                                      \
    Value val = unwrap(value);                                               \
    auto type = builder->getType<TYPE>();                                    \
    return wrap(builder->build_is_type_op(val, type));                       \
  }

DEFINE_IS_TYPE_OP(MLIRBuildIsTypeList, ListType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNonEmptyList, ConsType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNil, NilType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeMap, MapType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeNumber, NumberType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeInteger, IntegerType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeFixnum, FixnumType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeBigInt, BigIntType);
DEFINE_IS_TYPE_OP(MLIRBuildIsTypeFloat, FloatType);

}  // namespace eir
}  // namespace lumen
