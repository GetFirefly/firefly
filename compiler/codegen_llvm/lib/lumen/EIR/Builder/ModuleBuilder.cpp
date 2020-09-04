#include "lumen/term/Encoding.h"
#include "lumen/EIR/Builder/ModuleBuilder.h"
#include "lumen/EIR/IR/EIROps.h"
#include "lumen/EIR/IR/EIRTypes.h"
#include "lumen/EIR/Builder/Passes.h"

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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using ::llvm::Optional;
using ::llvm::raw_ostream;
using ::llvm::StringSwitch;
using ::llvm::TargetMachine;

using ::mlir::edsc::appendToBlock;
using ::mlir::edsc::buildInNewBlock;
using ::mlir::edsc::createBlock;
using ::mlir::edsc::OperationBuilder;
using ::mlir::edsc::ScopedContext;
using ::mlir::edsc::ValueBuilder;
using ::mlir::LLVM::LLVMIntegerType;
using ::mlir::LLVM::LLVMStructType;
using ::mlir::LLVM::LLVMType;
using ::mlir::PassManager;
using ::mlir::OpPassManager;
using namespace ::mlir::edsc::intrinsics;

namespace LLVM = ::mlir::LLVM;

using std_cmpi = ValueBuilder<::mlir::CmpIOp>;
using llvm_addressof = ValueBuilder<LLVM::AddressOfOp>;
using llvm_bitcast = ValueBuilder<LLVM::BitcastOp>;
using llvm_null = ValueBuilder<LLVM::NullOp>;
using eir_br = OperationBuilder<::lumen::eir::BranchOp>;
using eir_call = OperationBuilder<::lumen::eir::CallOp>;
using eir_invoke = OperationBuilder<::lumen::eir::InvokeOp>;
using eir_cond_br = OperationBuilder<::lumen::eir::CondBranchOp>;
using eir_landingpad = OperationBuilder<::lumen::eir::LandingPadOp>;
using eir_return = OperationBuilder<::lumen::eir::ReturnOp>;
using eir_cast = ValueBuilder<::lumen::eir::CastOp>;
using eir_cmpeq = ValueBuilder<::lumen::eir::CmpEqOp>;
using eir_cmpneq = ValueBuilder<::lumen::eir::CmpNeqOp>;
using eir_cmpgt = ValueBuilder<::lumen::eir::CmpGtOp>;
using eir_cmpgte = ValueBuilder<::lumen::eir::CmpGteOp>;
using eir_cmplt = ValueBuilder<::lumen::eir::CmpLtOp>;
using eir_cmplte = ValueBuilder<::lumen::eir::CmpLteOp>;
using eir_atom = ValueBuilder<::lumen::eir::ConstantAtomOp>;
using eir_nil = ValueBuilder<::lumen::eir::ConstantNilOp>;
using eir_list = ValueBuilder<::lumen::eir::ListOp>;
using eir_none = ValueBuilder<::lumen::eir::ConstantNoneOp>;
using eir_int = ValueBuilder<::lumen::eir::ConstantIntOp>;
using eir_bigint = ValueBuilder<::lumen::eir::ConstantBigIntOp>;
using eir_float = ValueBuilder<::lumen::eir::ConstantFloatOp>;
using eir_map_insert = OperationBuilder<::lumen::eir::MapInsertOp>;
using eir_map_update = OperationBuilder<::lumen::eir::MapUpdateOp>;
using eir_constant_binary = ValueBuilder<::lumen::eir::ConstantBinaryOp>;
using eir_constant_list = ValueBuilder<::lumen::eir::ConstantListOp>;
using eir_constant_tuple = ValueBuilder<::lumen::eir::ConstantTupleOp>;
using eir_constant_map = ValueBuilder<::lumen::eir::ConstantMapOp>;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(lumen::eir::FuncOp, MLIRFunctionOpRef);
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(lumen::eir::ModuleBuilder,
                                   MLIRModuleBuilderRef);

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
#include "lumen/EIR/IR/EIREncoding.h.inc"
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

static Value getOrInsertGlobal(OpBuilder &builder, ModuleOp &mod, Location loc,
                               StringRef name, LLVMType valueTy,
                               bool isConstant, LLVM::Linkage linkage,
                               LLVM::ThreadLocalMode tlsMode, Attribute value);

//===----------------------------------------------------------------------===//
// ModuleBuilder
//===----------------------------------------------------------------------===//

extern "C" MLIRModuleBuilderRef MLIRCreateModuleBuilder(
    MLIRContextRef context, const char *name, SourceLocation sl,
    LLVMTargetMachineRef tm) {
  MLIRContext *ctx = unwrap(context);
  TargetMachine *targetMachine = unwrap(tm);
  StringRef moduleName(name);
  StringRef filename(sl.filename);

  auto archType = targetMachine->getTargetTriple().getArch();
  auto pointerSizeInBits =
      targetMachine->createDataLayout().getPointerSizeInBits(0);
  bool supportsNanboxing = archType == llvm::Triple::ArchType::x86_64;
  auto encoding = lumen::Encoding{pointerSizeInBits, supportsNanboxing};
  auto immediateMask = lumen_immediate_mask(&encoding);
  auto maxAllowedImmediateVal =
      APInt(64, immediateMask.maxAllowedValue, /*signed=*/false);
  auto immediateBits = maxAllowedImmediateVal.getActiveBits();
  Location loc = mlir::FileLineColLoc::get(filename, sl.line, sl.column, ctx);
  return wrap(
      new ModuleBuilder(*ctx, moduleName, loc, targetMachine, immediateBits));
}

ModuleBuilder::ModuleBuilder(MLIRContext &context, StringRef name, Location loc,
                             const TargetMachine *targetMachine,
                             unsigned immediateBits)
    : builder(&context),
      targetMachine(targetMachine),
      immediateBitWidth(immediateBits) {
  // Create an empty module into which we can codegen functions
  theModule = builder.create<mlir::ModuleOp>(loc, name);
  assert(theModule != nullptr);
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

  // Move to the heap
  return wrap(new mlir::ModuleOp(finished));
}

mlir::ModuleOp ModuleBuilder::finish() {
  mlir::ModuleOp finished;
  std::swap(finished, theModule);

  // Apply some fixup passes to the generated IR
  auto pm = std::unique_ptr<PassManager>(new PassManager(builder.getContext(), /*verifyPasses=*/false));
  OpPassManager &fm = pm->nest<::lumen::eir::FuncOp>();
  fm.addPass(::lumen::eir::createInsertTraceConstructorsPass());
  pm->run(finished);

  return finished;
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

extern "C" FunctionDeclResult MLIRCreateFunction(MLIRModuleBuilderRef b,
                                                 MLIRLocationRef locref,
                                                 const char *name,
                                                 const Arg *argv, int argc,
                                                 EirType *type) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  StringRef functionName(name);
  llvm::SmallVector<Arg, 2> functionArgs(argv, argv + argc);
  auto fun = builder->create_function(loc, functionName, functionArgs, type);
  if (!fun) return {nullptr, nullptr};

  MLIRFunctionOpRef funRef = wrap(new FuncOp(fun));
  FuncOp *tempFun = unwrap(funRef);
  auto *entryBlock = tempFun->addEntryBlock();
  builder->position_at_end(entryBlock);
  MLIRBlockRef entry = wrap(entryBlock);

  return {funRef, entry};
}

FuncOp ModuleBuilder::getOrDeclareFunction(StringRef symbol, Type resultTy,
                                           bool isVarArg,
                                           ArrayRef<Type> argTypes) {
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(theModule, symbol);
  if (funcOp) return dyn_cast_or_null<FuncOp>(funcOp);

  // Create a function declaration for the symbol
  FunctionType fnTy;
  if (resultTy) {
    fnTy = builder.getFunctionType(argTypes, ArrayRef<Type>{resultTy});
  } else {
    fnTy = builder.getFunctionType(argTypes, ArrayRef<Type>{});
  }

  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(theModule.getBody());
  auto op = builder.create<FuncOp>(theModule.getLoc(), symbol, fnTy);
  if (isVarArg) {
    op.setAttr("std.varargs", builder.getBoolAttr(true));
  }
  builder.restoreInsertionPoint(ip);

  return op;
}

FuncOp ModuleBuilder::create_function(Location loc, StringRef functionName,
                                      SmallVectorImpl<Arg> &functionArgs,
                                      EirType *resultType) {
  llvm::SmallVector<Type, 2> argTypes;
  argTypes.reserve(functionArgs.size());
  for (auto it = functionArgs.begin(); it != functionArgs.end(); it++) {
    Type type = getArgType(it);
    if (!type) return nullptr;
    argTypes.push_back(type);
  }

  // All generated functions get our custom exception handling personality
  Type i32Ty = builder.getIntegerType(32);
  auto personalityFn =
      getOrDeclareFunction("lumen_eh_personality", i32Ty, /*vararg=*/true);
  auto personalityFnSymbol = builder.getSymbolRefAttr("lumen_eh_personality");
  auto personalityAttr =
      builder.getNamedAttr("personality", personalityFnSymbol);

  ArrayRef<NamedAttribute> attrs({personalityAttr});

  if (resultType->any.tag == EirTypeTag::None) {
    auto fnType = builder.getFunctionType(argTypes, llvm::None);
    return FuncOp::create(loc, functionName, fnType, attrs);
  } else {
    auto fnType =
        builder.getFunctionType(argTypes, fromRust(builder, resultType));
    return FuncOp::create(loc, functionName, fnType, attrs);
  }
}

extern "C" void MLIRAddFunction(MLIRModuleBuilderRef b, MLIRFunctionOpRef f) {
  ModuleBuilder *builder = unwrap(b);
  FuncOp *fun = unwrap(f);
  builder->add_function(*fun);
}

void ModuleBuilder::add_function(FuncOp f) { theModule.push_back(f); }

extern "C" MLIRValueRef MLIRBuildClosure(MLIRModuleBuilderRef b,
                                         eir::Closure *closure) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_closure(closure));
}

Value ModuleBuilder::build_closure(Closure *closure) {
  llvm::SmallVector<Value, 2> args;
  unwrapValues(closure->env, closure->envLen, args);
  Location loc = unwrap(closure->loc);
  auto op = builder.create<ClosureOp>(loc, closure, args);
  return op.getResult();
}

extern "C" bool MLIRBuildUnpackEnv(MLIRModuleBuilderRef b,
                                   MLIRLocationRef locref, MLIRValueRef ev,
                                   MLIRValueRef *values, unsigned numValues) {
  assert(numValues > 0 && "expected env size of 1 or more");
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value envBox = unwrap(ev);
  auto opBuilder = builder->getBuilder();
  Value env = opBuilder.create<CastOp>(
      loc, envBox, BoxType::get(opBuilder.getType<ClosureType>()));
  for (auto i = 0; i < numValues; i++) {
    values[i] = wrap(builder->build_unpack_op(loc, env, i));
  }
  return true;
}

Value ModuleBuilder::build_unpack_op(Location loc, Value env, unsigned index) {
  auto unpack = builder.create<UnpackEnvOp>(loc, env, index);
  return unpack.getResult();
}

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

extern "C" void MLIRBuildBr(MLIRModuleBuilderRef b, MLIRLocationRef locref,
                            MLIRBlockRef destBlk, MLIRValueRef *argv,
                            unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *dest = unwrap(destBlk);
  if (argc > 0) {
    llvm::SmallVector<Value, 2> args;
    unwrapValues(argv, argc, args);
    builder->build_br(loc, dest, args);
  } else {
    builder->build_br(loc, dest);
  }
}

void ModuleBuilder::build_br(Location loc, Block *dest, ValueRange destArgs) {
  builder.create<BranchOp>(loc, dest, destArgs);
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildIf(MLIRModuleBuilderRef b, MLIRLocationRef locref,
                            MLIRValueRef val, MLIRBlockRef y,
                            MLIRValueRef *yArgv, unsigned yArgc, MLIRBlockRef n,
                            MLIRValueRef *nArgv, unsigned nArgc, MLIRBlockRef o,
                            MLIRValueRef *oArgv, unsigned oArgc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
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
  builder->build_if(loc, value, yes, no, other, yesArgs, noArgs, otherArgs);
}

void ModuleBuilder::build_if(Location loc, Value value, Block *yes, Block *no,
                             Block *other, SmallVectorImpl<Value> &yesArgs,
                             SmallVectorImpl<Value> &noArgs,
                             SmallVectorImpl<Value> &otherArgs) {
  //  Create the `if`, if necessary
  bool withOtherwiseRegion = other != nullptr;
  Value isTrue = value;
  if (!value.getType().isa<BooleanType>()) {
    // The condition is not boolean, so we need to do a comparison
    auto trueConst = builder.create<ConstantAtomOp>(loc, true);
    isTrue = builder.create<CmpEqOp>(loc, value, trueConst, /*strict=*/true);
  }

  if (!other) {
    // No need to do any additional comparisons
    builder.create<CondBranchOp>(loc, isTrue, yes, yesArgs, no, noArgs);
    return;
  }

  // Otherwise we need an additional check to see if we use the otherwise branch
  auto falseConst = builder.create<ConstantAtomOp>(loc, false);
  Value isFalse =
      builder.create<CmpEqOp>(loc, value, falseConst, /*strict=*/true);

  Block *currentBlock = builder.getBlock();
  Block *falseBlock = currentBlock->splitBlock(falseConst);

  builder.setInsertionPointToEnd(falseBlock);
  builder.create<CondBranchOp>(loc, isFalse, no, noArgs, other, otherArgs);

  // Go back to original block and insert conditional branch for first
  // comparison
  builder.setInsertionPointToEnd(currentBlock);
  builder.create<CondBranchOp>(loc, isTrue, yes, yesArgs, falseBlock,
                               ArrayRef<Value>{});
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
  Location loc = unwrap(op.loc);
  SmallVector<MatchBranch, 2> branches;

  ArrayRef<MLIRMatchBranch> inBranches(op.branches,
                                       op.branches + op.numBranches);
  for (auto &inBranch : inBranches) {
    // Extract destination block and base arguments
    Block *dest = unwrap(inBranch.dest);
    Location branchLoc = unwrap(inBranch.loc);
    SmallVector<Value, 1> destArgs;
    if (inBranch.destArgc > 0) {
      ArrayRef<MLIRValueRef> inDestArgs(inBranch.destArgv,
                                        inBranch.destArgv + inBranch.destArgc);
      for (auto argRef : inDestArgs) {
        Value arg = unwrap(argRef);
        destArgs.push_back(arg);
      }
    }
    // Convert match pattern payload
    auto pattern = convertMatchPattern(inBranch.pattern);
    // Create internal branch type
    MatchBranch branch(branchLoc, dest, destArgs, std::move(pattern));
    branches.push_back(std::move(branch));
  }

  // We don't use an explicit operation for matches, as currently
  // there isn't enough structure in place to allow nested regions
  // to reference blocks from containing ops
  lumen::eir::lowerPatternMatch(builder, loc, selector, branches);
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildUnreachable(MLIRModuleBuilderRef b,
                                     MLIRLocationRef locref) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  builder->build_unreachable(loc);
}

void ModuleBuilder::build_unreachable(Location loc) {
  builder.create<UnreachableOp>(loc);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildReturn(MLIRModuleBuilderRef b, MLIRLocationRef locref,
                                MLIRValueRef value) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  builder->build_return(loc, unwrap(value));
}

void ModuleBuilder::build_return(Location loc, Value value) {
  ScopedContext scope(builder, loc);

  if (!value) {
    builder.create<ReturnOp>(loc);
  } else {
    Block *block = builder.getBlock();
    auto func = cast<FuncOp>(block->getParentOp());
    auto resultTypes = func.getCallableResults();
    Type expectedType = resultTypes.front();
    Type valueType = value.getType();
    if (expectedType == valueType) {
      builder.create<ReturnOp>(loc, value);
    } else {
      Value cast = eir_cast(value, expectedType);
      builder.create<ReturnOp>(loc, cast);
    }
  }
}

//===----------------------------------------------------------------------===//
// TraceCaptureOp/TraceConstructOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildTraceCaptureOp(MLIRModuleBuilderRef b,
                                        MLIRLocationRef locref, MLIRBlockRef d,
                                        MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *dest = unwrap(d);
  if (argc > 0) {
    ArrayRef<MLIRValueRef> args(argv, argv + argc);
    builder->build_trace_capture_op(loc, dest, args);
  } else {
    builder->build_trace_capture_op(loc, dest);
  }
}

void ModuleBuilder::build_trace_capture_op(Location loc, Block *dest,
                                           ArrayRef<MLIRValueRef> destArgs) {
  auto traceType = TraceRefType::get(builder.getContext());
  auto captureOp = builder.create<TraceCaptureOp>(loc, traceType);
  auto capture = captureOp.getResult();

  // Fixup type of destination block argument
  dest->getArgument(0).setType(traceType);

  SmallVector<Value, 1> extendedArgs;
  extendedArgs.push_back(capture);
  for (auto destArg : destArgs) {
    Value arg = unwrap(destArg);
    extendedArgs.push_back(arg);
  }

  builder.create<BranchOp>(loc, dest, extendedArgs);
}

extern "C" MLIRValueRef MLIRBuildTraceConstructOp(MLIRModuleBuilderRef b,
                                                  MLIRLocationRef locref,
                                                  MLIRValueRef t) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value trace = unwrap(t);
  return wrap(builder->build_trace_construct_op(loc, trace));
}

Value ModuleBuilder::build_trace_construct_op(Location loc, Value trace) {
  builder.create<TraceConstructOp>(loc, trace);
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildMapOp(MLIRModuleBuilderRef b, MapUpdate op) {
  ModuleBuilder *builder = unwrap(b);
  builder->build_map_update(op);
}

void ModuleBuilder::build_map_update(MapUpdate op) {
  Location loc = unwrap(op.loc);
  ScopedContext scope(builder, loc);

  ArrayRef<MapAction> actions(op.actionsv, op.actionsv + op.actionsc);
  Value map = unwrap(op.map);
  // Each insert or update implicitly branches to a continuation block for the
  // next insert/update; the last continuation block simply branches
  // unconditionally to the ok block
  Block *current = builder.getInsertionBlock();
  Block *ok;
  Block *err = unwrap(op.err);

  // For empty maps, we simply branch to the continuation block
  if (actions.size() == 0) {
    ok = unwrap(op.ok);
    builder.create<BranchOp>(loc, ok, ValueRange{map});
    return;
  }

  auto termType = builder.getType<TermType>();

  unsigned numActions = actions.size();
  unsigned lastAction = numActions - 1;
  for (unsigned i = 0; i < numActions; i++) {
    MapAction action = actions[i];
    Value key = unwrap(action.key);
    Value val = unwrap(action.value);
    // Create the continuation block, which expects the updated map as arg;
    // as well as the error block. Use the ok/error blocks provided as part
    // of the op if this is the last action being generated
    if (i == lastAction) {
      ok = unwrap(op.ok);
    } else {
      ok = builder.createBlock(err, {termType});
    }
    builder.setInsertionPointToEnd(current);
    switch (action.action) {
      case MapActionType::Insert: {
        auto op = builder.create<MapInsertOp>(loc, map, key, val);
        Value newMap = op.getResult(0);
        assert(newMap && "expected result #0");
        Value newMapAsTerm = eir_cast(newMap, termType);
        Value isOk = op.getResult(1);
        assert(isOk && "expected result #1");
        builder.create<CondBranchOp>(loc, isOk, ok, ValueRange{newMapAsTerm},
                                     err, ValueRange{key});
        break;
      }
      case MapActionType::Update: {
        auto op = builder.create<MapUpdateOp>(loc, map, key, val);
        Value newMap = op.getResult(0);
        assert(newMap && "expected result #0");
        Value newMapAsTerm = eir_cast(newMap, termType);
        Value isOk = op.getResult(1);
        assert(isOk && "expected result #1");
        builder.create<CondBranchOp>(loc, isOk, ok, ValueRange{newMapAsTerm},
                                     err, ValueRange{key});
        break;
      }
      default:
        llvm::report_fatal_error(
            "tried to construct map update op with invalid type");
    }
    current = ok;
    // We need to update the `map` pointer, since we're implicitly in a new
    // block on the next iteration
    map = current->getArgument(0);
  }
}

//===----------------------------------------------------------------------===//
// Binary Operators
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildIsEqualOp(MLIRModuleBuilderRef b,
                                           MLIRLocationRef locref,
                                           MLIRValueRef l, MLIRValueRef r,
                                           bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_equal(loc, lhs, rhs, isExact));
}

Value ModuleBuilder::build_is_equal(Location loc, Value lhs, Value rhs,
                                    bool isExact) {
  ScopedContext scope(builder, loc);
  return eir_cmpeq(lhs, rhs, isExact);
}

extern "C" MLIRValueRef MLIRBuildIsNotEqualOp(MLIRModuleBuilderRef b,
                                              MLIRLocationRef locref,
                                              MLIRValueRef l, MLIRValueRef r,
                                              bool isExact) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_not_equal(loc, lhs, rhs, isExact));
}

Value ModuleBuilder::build_is_not_equal(Location loc, Value lhs, Value rhs,
                                        bool isExact) {
  ScopedContext scope(builder, loc);
  return eir_cmpneq(lhs, rhs, isExact);
}

extern "C" MLIRValueRef MLIRBuildLessThanOrEqualOp(MLIRModuleBuilderRef b,
                                                   MLIRLocationRef locref,
                                                   MLIRValueRef l,
                                                   MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than_or_equal(loc, lhs, rhs));
}

Value ModuleBuilder::build_is_less_than_or_equal(Location loc, Value lhs,
                                                 Value rhs) {
  ScopedContext scope(builder, loc);
  return eir_cmplte(lhs, rhs);
}

extern "C" MLIRValueRef MLIRBuildLessThanOp(MLIRModuleBuilderRef b,
                                            MLIRLocationRef locref,
                                            MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_less_than(loc, lhs, rhs));
}

Value ModuleBuilder::build_is_less_than(Location loc, Value lhs, Value rhs) {
  ScopedContext scope(builder, loc);
  return eir_cmplt(lhs, rhs);
}

extern "C" MLIRValueRef MLIRBuildGreaterThanOrEqualOp(MLIRModuleBuilderRef b,
                                                      MLIRLocationRef locref,
                                                      MLIRValueRef l,
                                                      MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than_or_equal(loc, lhs, rhs));
}

Value ModuleBuilder::build_is_greater_than_or_equal(Location loc, Value lhs,
                                                    Value rhs) {
  ScopedContext scope(builder, loc);
  return eir_cmpgte(lhs, rhs);
}

extern "C" MLIRValueRef MLIRBuildGreaterThanOp(MLIRModuleBuilderRef b,
                                               MLIRLocationRef locref,
                                               MLIRValueRef l, MLIRValueRef r) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value lhs = unwrap(l);
  Value rhs = unwrap(r);
  return wrap(builder->build_is_greater_than(loc, lhs, rhs));
}

Value ModuleBuilder::build_is_greater_than(Location loc, Value lhs, Value rhs) {
  ScopedContext scope(builder, loc);
  return eir_cmpgt(lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Logical Operators
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildLogicalAndOp(MLIRModuleBuilderRef b,
                                              MLIRLocationRef locref,
                                              MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);

  assert(argc >= 2 && "logical and operation does not have at least 2 operands");
  SmallVector<Value, 2> args;
  unwrapValues(argv, argc, args);

  return wrap(builder->build_logical_and(loc, args));
}

Value ModuleBuilder::build_logical_and(Location loc, ArrayRef<Value> args) {
  auto begin = args.begin();
  Value acc = *begin;

  for (auto it = std::next(begin); it != args.end(); ++it) {
    auto op = builder.create<LogicalAndOp>(loc, acc, *it);
    acc = op.getResult();
  }

  return acc;
}

extern "C" MLIRValueRef MLIRBuildLogicalOrOp(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref,
                                             MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);

  assert(argc >= 2 && "logical or operation does not have at least 2 operands");
  SmallVector<Value, 2> args;
  unwrapValues(argv, argc, args);

  return wrap(builder->build_logical_or(loc, args));
}

Value ModuleBuilder::build_logical_or(Location loc, ArrayRef<Value> args) {
  auto begin = args.begin();
  Value acc = *begin;

  for (auto it = std::next(begin); it != args.end(); ++it) {
    auto op = builder.create<LogicalOrOp>(loc, acc, *it);
    acc = op.getResult();
  }

  return acc;
}

//===----------------------------------------------------------------------===//
// Function Calls
//===----------------------------------------------------------------------===//

#define INTRINSIC_BUILDER(Alias, Op)                                    \
  static Optional<Value> Alias(ModuleBuilder *modBuilder, Location loc, \
                               ArrayRef<Value> args) {                  \
    auto builder = modBuilder->getBuilder();                            \
    auto op = builder.create<Op>(loc, args);                            \
                                                                        \
    return op.getResult();                                              \
  }

INTRINSIC_BUILDER(buildIntrinsicPrintOp, PrintOp);
INTRINSIC_BUILDER(buildIntrinsicAddOp, AddOp);
INTRINSIC_BUILDER(buildIntrinsicSubOp, SubOp);
INTRINSIC_BUILDER(buildIntrinsicNegOp, NegOp);
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

#define ERROR_SYMBOL 46
#define THROW_SYMBOL 58
#define EXIT_SYMBOL 59

static Optional<Value> buildIntrinsicError1Op(ModuleBuilder *modBuilder,
                                              Location loc,
                                              ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  APInt id(modBuilder->immediateBitWidth, ERROR_SYMBOL, /*signed=*/false);
  auto aError = builder.create<ConstantAtomOp>(loc, id, "error");
  Value kind = aError.getResult();
  Value reason = args.front();
  Type traceTy = builder.getType<TraceRefType>();
  Value trace = builder.create<TraceCaptureOp>(loc, traceTy);
  builder.create<ThrowOp>(loc, kind, reason, trace);

  return llvm::None;
}

static Optional<Value> buildIntrinsicError2Op(ModuleBuilder *modBuilder,
                                              Location loc,
                                              ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  APInt id(modBuilder->immediateBitWidth, ERROR_SYMBOL, /*signed=*/false);
  auto aError = builder.create<ConstantAtomOp>(loc, id, "error");
  Value kind = aError.getResult();
  Value reason = args[0];
  Value where = args[1];
  auto tuple = builder.create<TupleOp>(loc, ArrayRef<Value>{reason, where});
  Value errorReason = tuple.getResult();
  Type traceTy = builder.getType<TraceRefType>();
  Value trace = builder.create<TraceCaptureOp>(loc, traceTy);
  builder.create<ThrowOp>(loc, kind, errorReason, trace);

  return llvm::None;
}

static Optional<Value> buildIntrinsicExit1Op(ModuleBuilder *modBuilder,
                                             Location loc,
                                             ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  APInt id(modBuilder->immediateBitWidth, ERROR_SYMBOL, /*signed=*/false);
  auto aExit = builder.create<ConstantAtomOp>(loc, id, "exit");
  Value kind = aExit.getResult();
  Value reason = args.front();
  Type traceTy = builder.getType<TraceRefType>();
  Value trace = builder.create<TraceCaptureOp>(loc, traceTy);
  builder.create<ThrowOp>(loc, kind, reason, trace);

  return llvm::None;
}

static Optional<Value> buildIntrinsicThrowOp(ModuleBuilder *modBuilder,
                                             Location loc,
                                             ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  APInt id(modBuilder->immediateBitWidth, THROW_SYMBOL, /*signed=*/false);
  auto aThrow = builder.create<ConstantAtomOp>(loc, id, "throw");
  Value kind = aThrow.getResult();
  Value reason = args.front();
  Type traceTy = builder.getType<TraceRefType>();
  Value trace = builder.create<TraceCaptureOp>(loc, traceTy);
  builder.create<ThrowOp>(loc, kind, reason, trace);

  return llvm::None;
}

static Optional<Value> buildIntrinsicRaiseOp(ModuleBuilder *modBuilder,
                                             Location loc,
                                             ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  Value kind = args[0];
  Value reason = args[1];
  Value trace = args[2];
  builder.create<ThrowOp>(loc, kind, reason, trace);

  return llvm::None;
}

static Optional<Value> buildIntrinsicCmpEqStrictOp(ModuleBuilder *modBuilder,
                                                   Location loc,
                                                   ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  assert(args.size() == 2 && "expected =:= operator to receive two operands");
  Value lhs = args[0];
  Value rhs = args[1];
  auto op = builder.create<CmpEqOp>(loc, lhs, rhs, /*strict=*/true);
  return op.getResult();
}

static Optional<Value> buildIntrinsicCmpNeqStrictOp(ModuleBuilder *modBuilder,
                                                    Location loc,
                                                    ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  assert(args.size() == 2 && "expected =/= operator to receive two operands");
  Value lhs = args[0];
  Value rhs = args[1];
  auto op = builder.create<CmpNeqOp>(loc, lhs, rhs, /*strict=*/true);
  return op.getResult();
}

static Optional<Value> buildIntrinsicFailOp(ModuleBuilder *modBuilder,
                                            Location loc,
                                            ArrayRef<Value> args) {
  auto builder = modBuilder->getBuilder();
  assert(args.size() == 1 && "expected fail/1 to receive one operand");
  auto calleeSymbol =
      FlatSymbolRefAttr::get("__lumen_builtin_fail/1", builder.getContext());
  auto termTy = builder.getType<TermType>();
  SmallVector<Type, 1> resultTypes{termTy};
  auto op = builder.create<CallOp>(loc, calleeSymbol, resultTypes, args);
  op.setAttr("tail", builder.getUnitAttr());
  return op.getResult(0);
}

using BuildIntrinsicFnT = Optional<Value> (*)(ModuleBuilder *, Location loc,
                                              ArrayRef<Value>);

static Optional<BuildIntrinsicFnT> getIntrinsicBuilder(StringRef target) {
  auto fnPtr = StringSwitch<BuildIntrinsicFnT>(target)
                   .Case("erlang:error/1", buildIntrinsicError1Op)
                   .Case("erlang:exit/1", buildIntrinsicExit1Op)
                   .Case("erlang:error/2", buildIntrinsicError2Op)
                   .Case("erlang:throw/1", buildIntrinsicThrowOp)
                   .Case("erlang:raise/3", buildIntrinsicRaiseOp)
                   .Case("erlang:print/1", buildIntrinsicPrintOp)
                   .Case("erlang:fail/1", buildIntrinsicFailOp)
                   .Case("erlang:+/2", buildIntrinsicAddOp)
                   .Case("erlang:-/1", buildIntrinsicNegOp)
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

extern "C" void MLIRBuildThrow(MLIRModuleBuilderRef b, MLIRLocationRef locref,
                               MLIRValueRef kind_ref, MLIRValueRef reason_ref,
                               MLIRValueRef trace_ref) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value kind = unwrap(kind_ref);
  Value reason = unwrap(reason_ref);
  Value trace = unwrap(trace_ref);

  builder->getBuilder().create<ThrowOp>(loc, kind, reason, trace);
}

extern "C" void MLIRBuildStaticCall(MLIRModuleBuilderRef b,
                                    MLIRLocationRef locref, const char *name,
                                    MLIRValueRef *argv, unsigned argc,
                                    bool isTail, MLIRBlockRef okBlock,
                                    MLIRValueRef *okArgv, unsigned okArgc,
                                    MLIRBlockRef errBlock,
                                    MLIRValueRef *errArgv, unsigned errArgc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *ok = unwrap(okBlock);
  Block *err = unwrap(errBlock);
  StringRef functionName(name);
  SmallVector<Value, 2> args;
  unwrapValues(argv, argc, args);
  SmallVector<Value, 1> okArgs;
  unwrapValues(okArgv, okArgc, okArgs);
  SmallVector<Value, 1> errArgs;
  unwrapValues(errArgv, errArgc, errArgs);

  bool isInvoke = !isTail && err != nullptr;
  if (isInvoke) {
    builder->build_static_invoke(loc, functionName, args, isTail, ok, okArgs,
                                 err, errArgs);
  } else {
    builder->build_static_call(loc, functionName, args, isTail, ok, okArgs);
  }
}

bool ModuleBuilder::maybe_build_intrinsic(Location loc, StringRef target,
                                          ArrayRef<Value> args, bool isTail,
                                          Block *ok, ArrayRef<Value> okArgs) {
  // If this is a call to an intrinsic, lower accordingly
  auto buildIntrinsicFnOpt = getIntrinsicBuilder(target);

  if (!buildIntrinsicFnOpt) return false;

  auto buildIntrinsicFn = buildIntrinsicFnOpt.getValue();
  auto resultOpt = buildIntrinsicFn(this, loc, args);
  auto isThrow = StringSwitch<bool>(target)
                     .Case("erlang:error/1", true)
                     .Case("erlang:error/2", true)
                     .Case("erlang:exit/1", true)
                     .Case("erlang:throw/1", true)
                     .Case("erlang:raise/3", true)
                     .Default(false);

  if (isThrow) return true;

  auto termTy = builder.getType<TermType>();
  // Tail calls directly return to caller
  if (isTail) {
    if (resultOpt) {
      auto result = resultOpt.getValue();
      auto func = result.getDefiningOp()->getParentOfType<FuncOp>();
      if (result.getType() != func.getType().getResult(0)) {
        Value coercedResult = eir_cast(result, termTy);
        coercedResult.dump();
        eir_return(ValueRange(coercedResult));
      } else {
        eir_return(ValueRange(result));
      }
    } else {
      eir_return();
    }
    return true;
  }

  // If the call has a result, branch unconditionally to the success
  // block, since intrinsics are supposed to be validated by the compiler;
  // this is not 100% the case right now, but will be soon
  if (resultOpt) {
    auto result = resultOpt.getValue();
    if (result.getType() != ok->getArgument(0).getType()) {
      Value okArg = eir_cast(result, termTy);
      eir_br(ok, ValueRange(okArg));
    } else {
      eir_br(ok, ValueRange(result));
    }
    return true;
  }

  // If the call has no result and isn't an error intrinsic,
  // then branch to the next block directly
  eir_br(ok, ValueRange());
  return true;
}

void ModuleBuilder::build_static_invoke(Location loc, StringRef target,
                                        ArrayRef<Value> args, bool isTail,
                                        Block *ok, ArrayRef<Value> okArgs,
                                        Block *err, ArrayRef<Value> errArgs) {
  ScopedContext scope(builder, loc);

  if (maybe_build_intrinsic(loc, target, args, isTail, ok, okArgs)) return;

  // Create symbolref and lookup function definition (if present)
  auto callee = builder.getSymbolRefAttr(target);
  auto fn = theModule.lookupSymbol<FuncOp>(callee.getValue());

  // Set up landing pad in error block
  Block *unwind = build_landing_pad(loc, err);

  // Create "normal" landing pad before the "unwind" pad
  if (!ok) {
    // If no normal block was given, create one to hold the return
    Block *normal;
    if (fn) {
      normal = createBlock(fn.getCallableResults());
    } else {
      auto termType = builder.getType<TermType>();
      normal = createBlock({termType});
    }
    eir_invoke(callee, args, normal, okArgs, unwind, errArgs);
    appendToBlock(normal,
                  [&](ValueRange args) { eir_return(ValueRange(args)); });
  } else {
    // Otherwise create a new block that will relay the result
    // to the "real" normal block
    eir_invoke(callee, args, ok, okArgs, unwind, errArgs);
  }
}

void ModuleBuilder::build_static_call(Location loc, StringRef target,
                                      ArrayRef<Value> args, bool isTail,
                                      Block *cont, ArrayRef<Value> contArgs) {
  ScopedContext scope(builder, loc);

  if (maybe_build_intrinsic(loc, target, args, isTail, cont, contArgs)) return;

  // Create symbolref and lookup function definition (if present)
  auto callee = builder.getSymbolRefAttr(target);
  auto fn = theModule.lookupSymbol<FuncOp>(callee.getValue());
  auto termType = builder.getType<TermType>();

  // Build result types list
  SmallVector<Type, 1> fnResults;
  if (fn) {
    auto rs = fn.getCallableResults();
    fnResults.append(rs.begin(), rs.end());
  } else {
    SmallVector<Type, 1> argTypes;
    for (auto arg : args) {
      argTypes.push_back(arg.getType());
    }
    fnResults.push_back(termType);
  }

  if (isTail) {
    auto mustTail = builder.getNamedAttr("musttail", builder.getUnitAttr());
    Operation *call =
        eir_call(callee, fnResults, args, ArrayRef<NamedAttribute>{mustTail});
    auto parentFunc = call->getParentOfType<FuncOp>();
    if (fnResults.size() > 0 &&
        fnResults[0] != parentFunc.getType().getResult(0)) {
      Value coercedResult = eir_cast(call->getResults().front(), termType);
      eir_return(ValueRange(coercedResult));
    } else {
      eir_return(call->getResults());
    }
  } else {
    auto tail = builder.getNamedAttr("tail", builder.getUnitAttr());
    Operation *call =
        eir_call(callee, fnResults, args, ArrayRef<NamedAttribute>{tail});
    SmallVector<Value, 1> contArgsFinal;
    if (fnResults.size() > 0 &&
        fnResults[0] != cont->getArgument(0).getType()) {
      Value coercedResult = eir_cast(call->getResults().front(), termType);
      contArgsFinal.push_back(coercedResult);
    } else {
      contArgsFinal.push_back(call->getResults().front());
    }
    for (auto arg : contArgs) {
      contArgsFinal.push_back(arg);
    }
    eir_br(cont, contArgsFinal);
  }
}

extern "C" void MLIRBuildClosureCall(MLIRModuleBuilderRef b,
                                     MLIRLocationRef locref, MLIRValueRef cls,
                                     MLIRValueRef *argv, unsigned argc,
                                     bool isTail, MLIRBlockRef okBlock,
                                     MLIRValueRef *okArgv, unsigned okArgc,
                                     MLIRBlockRef errBlock,
                                     MLIRValueRef *errArgv, unsigned errArgc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *ok = unwrap(okBlock);
  Block *err = unwrap(errBlock);
  SmallVector<Value, 2> args;
  unwrapValues(argv, argc, args);
  SmallVector<Value, 1> okArgs;
  unwrapValues(okArgv, okArgc, okArgs);
  SmallVector<Value, 1> errArgs;
  unwrapValues(errArgv, errArgc, errArgs);

  bool isInvoke = !isTail && err != nullptr;

  Value closure = unwrap(cls);
  if (auto closureOp = getDefinition<ClosureOp>(closure)) {
    // If this closure has no environment, we can replace the
    // call to the closure with a call directly to the actual function
    auto callee = closureOp.getCallee();
    if (isInvoke) {
      builder->build_static_invoke(loc, callee.getValue(), args, isTail, ok, okArgs,
                                   err, errArgs);
    } else {
      builder->build_static_call(loc, callee.getValue(), args, isTail, ok, okArgs);
    }
  } else {
    // We can't find the original closure definition, so this
    // function cannot be called directly, it must be called through `apply/2`
    builder->build_apply_2(loc, closure, args, isTail, ok, okArgs, err, errArgs);
  }
}

void ModuleBuilder::build_apply_2(Location loc, Value cls,
                                  ArrayRef<Value> args, bool isTail,
                                  Block *ok, ArrayRef<Value> okArgs,
                                  Block *err, ArrayRef<Value> errArgs) {
  ScopedContext scope(builder, loc);

  // We need to call apply/2 with the closure, and a list of arguments
  SmallVector<Value, 2> applyArgs;
  applyArgs.push_back(cls);
  applyArgs.push_back(eir_list(args));

  // Then, based on whether this was an invoke or not, call apply/2 appropriately
  bool isInvoke = !isTail && err != nullptr;
  if (isInvoke) {
    build_static_invoke(loc, "erlang:apply/2", applyArgs, isTail, ok, okArgs,
                        err, errArgs);
  } else {
    build_static_call(loc, "erlang:apply/2", applyArgs, isTail, ok, okArgs);
  }
  return;
}

Block *ModuleBuilder::build_landing_pad(Location loc, Block *err) {
  auto ip = builder.saveInsertionPoint();
  // This block is intended as the LLVM landing pad, and exists to
  // ensure that exception unwinding is handled properly
  Block *unwind = builder.createBlock(err);

  LLVMType i8Ty = builder.getType<LLVMIntegerType>(8);
  LLVMType i8PtrTy = i8Ty.getPointerTo();

  // Obtain catch type value from entry block
  Region *funcRegion = unwind->getParent();
  Block *entry = &funcRegion->front();
  Value catchType;
  // There are two ways to define the type we catch, depending
  // on whether the exceptions are SEH or DWARF/EHABI
  if (isLikeMsvc()) {
    StringRef typeInfoName("__lumen_erlang_error_type_info");
    for (auto bitcastOp : entry->getOps<LLVM::BitcastOp>()) {
      auto maybeAddrOf = bitcastOp.arg().getDefiningOp();
      if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(maybeAddrOf)) {
        if (addrOf.global_name() == typeInfoName) {
          catchType = bitcastOp.getResult();
          break;
        }
      }
    }
    if (!catchType) {
      // Not defined yet, we need to insert the operation
      builder.setInsertionPointToStart(entry);
      if (auto global = theModule.lookupSymbol<LLVM::GlobalOp>(typeInfoName)) {
        catchType = llvm_bitcast(i8PtrTy, llvm_addressof(global));
      } else {
        // type_name = lumen_panic\0
        LLVMType typeNameTy = LLVMType::getArrayTy(i8Ty, 12);
        LLVMType typeInfoTy = LLVMType::createStructTy(
            builder.getContext(),
            ArrayRef<LLVMType>{i8PtrTy, i8PtrTy, typeNameTy.getPointerTo()},
            StringRef("type_info"));
        Value catchTypeGlobal =
            getOrInsertGlobal(builder, theModule, loc, typeInfoName, typeInfoTy,
                              false, LLVM::Linkage::External,
                              LLVM::ThreadLocalMode::NotThreadLocal, nullptr);
        catchType = llvm_bitcast(i8PtrTy, catchTypeGlobal);
      }
    }
  } else {
    for (auto nullOp : entry->getOps<LLVM::NullOp>()) {
      catchType = nullOp;
      break;
    }
    if (!catchType) {
      // Not defined yet, we need to insert the operation
      builder.setInsertionPointToStart(entry);
      catchType = llvm_null(i8PtrTy);
    }
  }

  // Insert the landing pad in the unwind block
  builder.setInsertionPointToEnd(unwind);

  Operation *lp = eir_landingpad(catchType);
  eir_br(err, lp->getResults());

  Type traceTy = TraceRefType::get(builder.getContext());
  err->getArgument(2).setType(traceTy);

  // Restore original insertion point
  builder.restoreInsertionPoint(ip);

  return unwind;
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRCons(MLIRModuleBuilderRef b, MLIRLocationRef locref,
                                 MLIRValueRef h, MLIRValueRef t) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value head = unwrap(h);
  Value tail = unwrap(t);
  return wrap(builder->build_cons(loc, head, tail));
}

Value ModuleBuilder::build_cons(Location loc, Value head, Value tail) {
  auto op = builder.create<ConsOp>(loc, head, tail);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRConstructTuple(MLIRModuleBuilderRef b,
                                           MLIRLocationRef locref,
                                           MLIRValueRef *ev, unsigned ec) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  SmallVector<Value, 2> elements;
  unwrapValues(ev, ec, elements);
  return wrap(builder->build_tuple(loc, elements));
}

Value ModuleBuilder::build_tuple(Location loc, ArrayRef<Value> elements) {
  auto op = builder.create<TupleOp>(loc, elements);
  auto castOp =
      builder.create<CastOp>(loc, op.getResult(), builder.getType<TermType>());
  return castOp.getResult();
}

extern "C" MLIRValueRef MLIRConstructMap(MLIRModuleBuilderRef b,
                                         MLIRLocationRef locref, MapEntry *ev,
                                         unsigned ec) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  ArrayRef<MapEntry> entries(ev, ev + ec);
  return wrap(builder->build_map(loc, entries));
}

Value ModuleBuilder::build_map(Location loc, ArrayRef<MapEntry> entries) {
  auto op = builder.create<ConstructMapOp>(loc, entries);
  return op.getResult(0);
}

extern "C" void MLIRBuildBinaryStart(MLIRModuleBuilderRef b,
                                     MLIRLocationRef locref,
                                     MLIRBlockRef contBlock) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *cont = unwrap(contBlock);
  builder->build_binary_start(loc, cont);
}

void ModuleBuilder::build_binary_start(Location loc, Block *cont) {
  auto op = builder.create<BinaryStartOp>(loc, builder.getType<TermType>());
  auto bin = op.getResult();
  builder.create<BranchOp>(loc, cont, ArrayRef<Value>{bin});
}

extern "C" void MLIRBuildBinaryFinish(MLIRModuleBuilderRef b,
                                      MLIRLocationRef locref,
                                      MLIRBlockRef contBlock,
                                      MLIRValueRef binRef) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *cont = unwrap(contBlock);
  Value bin = unwrap(binRef);
  builder->build_binary_finish(loc, cont, bin);
}

void ModuleBuilder::build_binary_finish(Location loc, Block *cont, Value bin) {
  auto op =
      builder.create<BinaryFinishOp>(loc, builder.getType<TermType>(), bin);
  auto finished = op.getResult();
  // If the continuation is not a block but a return, then cont will be null
  if (cont) {
    builder.create<BranchOp>(loc, cont, ArrayRef<Value>{finished});
  } else {
    builder.create<ReturnOp>(loc, ArrayRef<Value>{finished});
  }
}

extern "C" void MLIRBuildBinaryPush(MLIRModuleBuilderRef b,
                                    MLIRLocationRef locref, MLIRValueRef bref,
                                    MLIRValueRef vref, MLIRValueRef sz,
                                    BinarySpecifier *spec, MLIRBlockRef okBlock,
                                    MLIRBlockRef errBlock) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *ok = unwrap(okBlock);
  Block *err = unwrap(errBlock);
  Value bin = unwrap(bref);
  Value value = unwrap(vref);
  Value size;
  if (sz) {
    size = unwrap(sz);
  }

  builder->build_binary_push(loc, bin, value, size, spec, ok, err);
}

void ModuleBuilder::build_binary_push(Location loc, Value bin, Value value,
                                      Value size, BinarySpecifier *spec,
                                      Block *ok, Block *err) {
  SmallVector<NamedAttribute, 4> attrs;
  auto tag = spec->tag;
  switch (tag) {
    case BinarySpecifierType::Bytes:
    case BinarySpecifierType::Bits:
      attrs.push_back(
          {builder.getIdentifier("type"), builder.getI8IntegerAttr(tag)});
      attrs.push_back({builder.getIdentifier("unit"),
                       builder.getI8IntegerAttr(spec->payload.us.unit)});
      break;
    case BinarySpecifierType::Utf8:
    case BinarySpecifierType::Utf16:
    case BinarySpecifierType::Utf32:
      attrs.push_back(
          {builder.getIdentifier("type"), builder.getI8IntegerAttr(tag)});
      attrs.push_back({builder.getIdentifier("endianness"),
                       builder.getI8IntegerAttr(spec->payload.es.endianness)});
      break;
    case BinarySpecifierType::Integer:
      attrs.push_back(
          {builder.getIdentifier("type"), builder.getI8IntegerAttr(tag)});
      attrs.push_back({builder.getIdentifier("unit"),
                       builder.getI8IntegerAttr(spec->payload.i.unit)});
      attrs.push_back({builder.getIdentifier("endianness"),
                       builder.getI8IntegerAttr(spec->payload.i.endianness)});
      attrs.push_back({builder.getIdentifier("signed"),
                       builder.getBoolAttr(spec->payload.i.isSigned)});
      break;
    case BinarySpecifierType::Float:
      attrs.push_back(
          {builder.getIdentifier("type"), builder.getI8IntegerAttr(tag)});
      attrs.push_back({builder.getIdentifier("unit"),
                       builder.getI8IntegerAttr(spec->payload.f.unit)});
      attrs.push_back({builder.getIdentifier("endianness"),
                       builder.getI8IntegerAttr(spec->payload.f.endianness)});
      break;
    default:
      llvm_unreachable("invalid binary specifier type");
  }
  auto op = builder.create<BinaryPushOp>(loc, bin, value, size, attrs);
  auto newBin = op.getResult(0);
  auto success = op.getResult(1);
  ArrayRef<Value> okArgs{newBin};
  ArrayRef<Value> errArgs{};
  builder.create<CondBranchOp>(loc, success, ok, okArgs, err, errArgs);
}

//===----------------------------------------------------------------------===//
// Receive
//===----------------------------------------------------------------------===//

extern "C" void MLIRBuildReceiveStart(MLIRModuleBuilderRef b,
                                      MLIRLocationRef locref,
                                      MLIRBlockRef contBlock,
                                      MLIRValueRef timeoutRef) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *cont = unwrap(contBlock);
  Value timeout = unwrap(timeoutRef);
  builder->build_receive_start(loc, cont, timeout);
}

void ModuleBuilder::build_receive_start(Location loc, Block *cont,
                                        Value timeout) {
  ScopedContext scope(builder, loc);
  // Make sure continuation block has correct type for ReceiveRef argument
  auto arg = cont->getArgument(0);
  auto recvRefType = builder.getType<ReceiveRefType>();
  arg.setType(builder.getType<ReceiveRefType>());
  // Create op
  auto op = builder.create<ReceiveStartOp>(loc, timeout);
  eir_br(cont, op.getResult());
}

extern "C" void MLIRBuildReceiveWait(MLIRModuleBuilderRef b,
                                     MLIRLocationRef locref,
                                     MLIRBlockRef timeoutBlock,
                                     MLIRBlockRef checkBlock,
                                     MLIRValueRef receiveRef) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *timeout = unwrap(timeoutBlock);
  Block *check = unwrap(checkBlock);
  Value receive_ref = unwrap(receiveRef);
  builder->build_receive_wait(loc, timeout, check, receive_ref);
}

void ModuleBuilder::build_receive_wait(Location loc, Block *timeout,
                                       Block *check, Value receive_ref) {
  ScopedContext scope(builder, loc);

  Block *currentBlock = builder.getBlock();

  auto op = builder.create<ReceiveWaitOp>(loc, receive_ref);
  auto waitStatus = op.getResult();
  auto receivedStatus = static_cast<int64_t>(ReceiveStatus::Received);
  Value received = std_cmpi(::mlir::CmpIPredicate::eq, waitStatus,
                            std_constant_int(receivedStatus, 8));

  Block *recvFailedBlock = builder.createBlock(currentBlock);
  Block *getMessageBlock = builder.createBlock(recvFailedBlock);

  // Conditionally branch based on wait status
  builder.setInsertionPointToEnd(currentBlock);
  eir_cond_br(received, getMessageBlock, ArrayRef<Value>{}, recvFailedBlock,
              ArrayRef<Value>{});

  // If received, get the message and branch to the check block
  builder.setInsertionPointToEnd(getMessageBlock);
  auto msg = builder.create<ReceiveMessageOp>(loc, receive_ref);
  eir_br(check, ArrayRef<Value>{msg.getResult()});

  // If not, check whether it was an error or a timeout
  builder.setInsertionPointToEnd(recvFailedBlock);

  // If the timeout block is unreachable, then we skip generating
  // the timeout checking code, and simply handle the error case by
  // raising a fatal error
  auto termTy = builder.getType<TermType>();
  /*
  if (isa<UnreachableOp>(check->getTerminator())) {
    auto calleeSymbol =
      FlatSymbolRefAttr::get("__lumen_builtin_fatal_error",
  builder.getContext()); auto callOp = builder.create<CallOp>(loc, calleeSymbol,
  ArrayRef<Type>{}, ValueRange()); callOp.setAttr("tail",
  builder.getUnitAttr()); callOp.setAttr("noreturn", builder.getUnitAttr());
    Value noneVal = eir_none(termTy);
    builder.create<ReturnOp>(loc, ValueRange(noneVal));
    return;
  }
  */

  // Otherwise we handle the timeout/error cases
  Block *fatalBlock = builder.createBlock(recvFailedBlock);

  builder.setInsertionPointToEnd(recvFailedBlock);

  // Conditionally branch based on whether this was a timeout
  auto timeoutStatus = static_cast<int64_t>(ReceiveStatus::Timeout);
  Value timedOut = std_cmpi(::mlir::CmpIPredicate::eq, waitStatus,
                            std_constant_int(timeoutStatus, 8));
  eir_cond_br(timedOut, timeout, ArrayRef<Value>{}, fatalBlock,
              ArrayRef<Value>{});

  // If this was an error, drop an abort, for now
  builder.setInsertionPointToEnd(fatalBlock);

  auto calleeSymbol = FlatSymbolRefAttr::get("__lumen_builtin_fatal_error",
                                             builder.getContext());
  auto callOp =
      builder.create<CallOp>(loc, calleeSymbol, ArrayRef<Type>{}, ValueRange());
  callOp.setAttr("tail", builder.getUnitAttr());
  callOp.setAttr("noreturn", builder.getUnitAttr());
  Value noneVal = eir_none(termTy);
  builder.create<ReturnOp>(loc, ValueRange(noneVal));
}

extern "C" void MLIRBuildReceiveDone(MLIRModuleBuilderRef b,
                                     MLIRLocationRef locref,
                                     MLIRBlockRef contBlock,
                                     MLIRValueRef receiveRef,
                                     MLIRValueRef *argv, unsigned argc) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Block *cont = unwrap(contBlock);
  Value receive_ref = unwrap(receiveRef);
  SmallVector<Value, 1> args;
  unwrapValues(argv, argc, args);
  builder->build_receive_done(loc, cont, receive_ref, args);
}

void ModuleBuilder::build_receive_done(Location loc, Block *cont,
                                       Value receive_ref,
                                       ArrayRef<Value> args) {
  // Inform the runtime that the receive was successful,
  // which removes the received message from the mailbox
  builder.create<ReceiveDoneOp>(loc, receive_ref);
  builder.create<BranchOp>(loc, cont, args);
}

//===----------------------------------------------------------------------===//
// ConstantFloat
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantFloat(MLIRModuleBuilderRef b,
                                               MLIRLocationRef locref,
                                               double value) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  return wrap(builder->build_constant_float(loc, value));
}

Value ModuleBuilder::build_constant_float(Location loc, double value) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_float(APFloat(value)), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildFloatAttr(MLIRModuleBuilderRef b,
                                               MLIRLocationRef locref,
                                               double value) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_float_attr(value));
}

Attribute ModuleBuilder::build_float_attr(double value) {
  return APFloatAttr::get(builder.getContext(), APFloat(value));
}

//===----------------------------------------------------------------------===//
// ConstantInt
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantInt(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref,
                                             uint64_t value) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  return wrap(builder->build_constant_int(loc, value));
}

Value ModuleBuilder::build_constant_int(Location loc, uint64_t value) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  APInt i(immediateBitWidth, value, /*signed=*/true);
  return eir_cast(eir_int(i), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildIntAttr(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref,
                                             uint64_t value) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->build_int_attr(value));
}

Attribute ModuleBuilder::build_int_attr(uint64_t value) {
  APInt i(immediateBitWidth, value, /*signed=*/true);
  auto intType = builder.getType<FixnumType>();
  return APIntAttr::get(builder.getContext(), intType, i);
}

//===----------------------------------------------------------------------===//
// ConstantBigInt
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantBigInt(MLIRModuleBuilderRef b,
                                                MLIRLocationRef locref,
                                                const char *str,
                                                unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  StringRef value(str);
  return wrap(builder->build_constant_bigint(loc, value, width));
}

Value ModuleBuilder::build_constant_bigint(Location loc, StringRef value,
                                           unsigned width) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  APInt i(width, value, /*radix=*/10);
  return eir_cast(eir_bigint(i), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildBigIntAttr(MLIRModuleBuilderRef b,
                                                MLIRLocationRef locref,
                                                const char *str,
                                                unsigned width) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_bigint_attr(value, width));
}

Attribute ModuleBuilder::build_bigint_attr(StringRef value, unsigned width) {
  APInt i(width, value, /*radix=*/10);
  auto bigIntType = builder.getType<BigIntType>();
  return APIntAttr::get(builder.getContext(), bigIntType, i);
}

//===----------------------------------------------------------------------===//
// ConstantAtom
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantAtom(MLIRModuleBuilderRef b,
                                              MLIRLocationRef locref,
                                              const char *str, uint64_t id) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  StringRef value(str);
  return wrap(builder->build_constant_atom(loc, value, id));
}

Value ModuleBuilder::build_constant_atom(Location loc, StringRef value,
                                         uint64_t valueId) {
  ScopedContext scope(builder, loc);
  APInt id(immediateBitWidth, valueId, /*isSigned=*/false);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_atom(id, value), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildAtomAttr(MLIRModuleBuilderRef b,
                                              MLIRLocationRef locref,
                                              const char *str, uint64_t id) {
  ModuleBuilder *builder = unwrap(b);
  StringRef value(str);
  return wrap(builder->build_atom_attr(value, id));
}

Attribute ModuleBuilder::build_atom_attr(StringRef value, uint64_t valueId) {
  APInt id(immediateBitWidth, valueId, /*isSigned=*/false);
  return AtomAttr::get(builder.getContext(), id, value);
}

Attribute ModuleBuilder::build_string_attr(StringRef value) {
  return builder.getStringAttr(value);
}

//===----------------------------------------------------------------------===//
// ConstantBinary
//===----------------------------------------------------------------------===//

extern "C" MLIRValueRef MLIRBuildConstantBinary(MLIRModuleBuilderRef b,
                                                MLIRLocationRef locref,
                                                const char *str, unsigned size,
                                                uint64_t header,
                                                uint64_t flags) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  StringRef value(str, (size_t)size);
  return wrap(builder->build_constant_binary(loc, value, header, flags));
}

Value ModuleBuilder::build_constant_binary(Location loc, StringRef value,
                                           uint64_t header, uint64_t flags) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_constant_binary(value, header, flags), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildBinaryAttr(MLIRModuleBuilderRef b,
                                                MLIRLocationRef locref,
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

extern "C" MLIRValueRef MLIRBuildConstantNil(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  return wrap(builder->build_constant_nil(loc));
}

Value ModuleBuilder::build_constant_nil(Location loc) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_nil(), termTy);
}

extern "C" MLIRAttributeRef MLIRBuildNilAttr(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref) {
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
                                              MLIRLocationRef locref,
                                              const MLIRAttributeRef *elements,
                                              int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  SmallVector<Attribute, 1> list;
  if (num_elements > 0) {
    for (auto ar : xs) {
      Attribute attr = unwrap(ar);
      list.push_back(attr);
    }
    return wrap(builder->build_constant_list(loc, list));
  } else {
    return wrap(builder->build_constant_nil(loc));
  }
}

Value ModuleBuilder::build_constant_list(Location loc,
                                         ArrayRef<Attribute> elements) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_constant_list(elements), termTy);
}

extern "C" MLIRValueRef MLIRBuildConstantTuple(MLIRModuleBuilderRef b,
                                               MLIRLocationRef locref,
                                               const MLIRAttributeRef *elements,
                                               int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  SmallVector<Attribute, 2> tuple;
  for (auto ar : xs) {
    Attribute attr = unwrap(ar);
    tuple.push_back(attr);
  }
  return wrap(builder->build_constant_tuple(loc, tuple));
}

Value ModuleBuilder::build_constant_tuple(Location loc,
                                          ArrayRef<Attribute> elements) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_constant_tuple(elements), termTy);
}

extern "C" MLIRValueRef MLIRBuildConstantMap(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref,
                                             const KeyValuePair *elements,
                                             int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  ArrayRef<KeyValuePair> xs(elements, elements + num_elements);
  SmallVector<Attribute, 4> list;
  list.reserve(xs.size() * 2);
  for (auto kvp : xs) {
    Attribute key = unwrap(kvp.key);
    assert(key && "expected constant map element to define a key");
    Attribute value = unwrap(kvp.value);
    assert(value && "expected constant map element to define a value");
    list.push_back(key);
    list.push_back(value);
  }
  return wrap(builder->build_constant_map(loc, list));
}

Value ModuleBuilder::build_constant_map(Location loc,
                                        ArrayRef<Attribute> elements) {
  ScopedContext scope(builder, loc);
  auto termTy = builder.getType<TermType>();
  return eir_cast(eir_constant_map(elements), termTy);
}

Attribute build_seq_attr(ModuleBuilder *builder,
                         ArrayRef<MLIRAttributeRef> elements, Type type) {
  SmallVector<Attribute, 3> list;
  list.reserve(elements.size());
  for (auto el : elements) {
    Attribute attr = unwrap(el);
    assert(attr && "unexpected nullptr found in attribute list");
    list.push_back(attr);
  }
  return builder->build_seq_attr(list, type);
}

Attribute ModuleBuilder::build_seq_attr(ArrayRef<Attribute> elements,
                                        Type type) {
  return SeqAttr::get(type, elements);
}

extern "C" MLIRAttributeRef MLIRBuildListAttr(MLIRModuleBuilderRef b,
                                              MLIRLocationRef locref,
                                              const MLIRAttributeRef *elements,
                                              int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);
  auto type = builder->getType<ConsType>();

  return wrap(build_seq_attr(builder, xs, type));
}

extern "C" MLIRAttributeRef MLIRBuildTupleAttr(MLIRModuleBuilderRef b,
                                               MLIRLocationRef locref,
                                               const MLIRAttributeRef *elements,
                                               int num_elements) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRAttributeRef> xs(elements, elements + num_elements);

  std::vector<Type> types;
  types.reserve(xs.size());
  for (auto el : xs) {
    Attribute attr = unwrap(el);
    assert(attr && "unexpected nullptr found in tuple attributes");
    types.push_back(attr.getType());
  }
  auto type = eir::TupleType::get(ArrayRef(types));

  return wrap(build_seq_attr(builder, xs, type));
}

extern "C" MLIRAttributeRef MLIRBuildMapAttr(MLIRModuleBuilderRef b,
                                             MLIRLocationRef locref,
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
// Helpers
//===----------------------------------------------------------------------===//

static Value getOrInsertGlobal(OpBuilder &builder, ModuleOp &mod, Location loc,
                               StringRef name, LLVMType valueTy,
                               bool isConstant, LLVM::Linkage linkage,
                               LLVM::ThreadLocalMode tlsMode,
                               Attribute value = Attribute()) {
  ScopedContext scope(builder, loc);

  if (auto global = mod.lookupSymbol<LLVM::GlobalOp>(name))
    return llvm_addressof(global);

  auto savePoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(mod.getBody());

  auto global = builder.create<LLVM::GlobalOp>(
      mod.getLoc(), valueTy, isConstant, linkage, tlsMode, name, value);

  builder.restoreInsertionPoint(savePoint);
  return llvm_addressof(global);
}

//===----------------------------------------------------------------------===//
// Locations/Spans
//===----------------------------------------------------------------------===//

extern "C" MLIRLocationRef MLIRCreateLocation(MLIRModuleBuilderRef b,
                                              SourceLocation sloc) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->getLocation(sloc));
}

extern "C" MLIRLocationRef MLIRCreateFusedLocation(MLIRModuleBuilderRef b,
                                                   MLIRLocationRef *locRefs,
                                                   unsigned numLocs) {
  ModuleBuilder *builder = unwrap(b);
  ArrayRef<MLIRLocationRef> lrs(locRefs, locRefs + numLocs);
  SmallVector<Location, 1> locs;
  for (auto it = lrs.begin(); it != lrs.end(); it++) {
    Location loc = unwrap(*it);
    locs.push_back(loc);
  }
  return wrap(builder->getFusedLocation(locs));
}

extern "C" MLIRLocationRef MLIRUnknownLocation(MLIRModuleBuilderRef b) {
  ModuleBuilder *builder = unwrap(b);
  return wrap(builder->getBuilder().getUnknownLoc());
}

Location ModuleBuilder::getLocation(SourceLocation sloc) {
  StringRef filename(sloc.filename);
  return mlir::FileLineColLoc::get(filename, sloc.line, sloc.column,
                                   builder.getContext());
}

Location ModuleBuilder::getFusedLocation(ArrayRef<Location> locs) {
  return mlir::FusedLoc::get(locs, builder.getContext());
}

//===----------------------------------------------------------------------===//
// Type Checking
//===----------------------------------------------------------------------===//

Value ModuleBuilder::build_is_type_op(Location loc, Value value,
                                      Type matchType) {
  auto op = builder.create<IsTypeOp>(loc, value, matchType);
  return op.getResult();
}

extern "C" MLIRValueRef MLIRBuildIsTypeTupleWithArity(MLIRModuleBuilderRef b,
                                                      MLIRLocationRef locref,
                                                      MLIRValueRef value,
                                                      unsigned arity) {
  ModuleBuilder *builder = unwrap(b);
  Location loc = unwrap(locref);
  Value val = unwrap(value);
  auto type = builder->getType<eir::TupleType>(arity);
  return wrap(builder->build_is_type_op(loc, val, type));
}

#define DEFINE_IS_TYPE_OP(NAME, TYPE)                                          \
  extern "C" MLIRValueRef NAME(MLIRModuleBuilderRef b, MLIRLocationRef locref, \
                               MLIRValueRef value) {                           \
    ModuleBuilder *builder = unwrap(b);                                        \
    Location loc = unwrap(locref);                                             \
    Value val = unwrap(value);                                                 \
    auto type = builder->getType<TYPE>();                                      \
    return wrap(builder->build_is_type_op(loc, val, type));                    \
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

//===----------------------------------------------------------------------===//
// Target Info
//===----------------------------------------------------------------------===//

bool ModuleBuilder::isLikeMsvc() {
  auto triple = targetMachine->getTargetTriple();
  if (triple.getEnvironment() == llvm::Triple::EnvironmentType::MSVC)
    return true;
  if (triple.getOS() == llvm::Triple::OSType::Win32) return true;
  return false;
}

}  // namespace eir
}  // namespace lumen
