#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"

#include "mlir/IR/Module.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/Support/Format.h"

using namespace lumen::eir;

using ::mlir::Attribute;
using ::mlir::DialectAsmPrinter;

static DialectRegistration<EirDialect> eir_dialect;

/// Create an instance of the EIR dialect, owned by the context.
///
/// This is where EIR types, operations, and attributes are registered.
EirDialect::EirDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  // addInterfaces<EirInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "lumen/compiler/Dialect/EIR/IR/EIROps.cpp.inc"
      >();
  addTypes<TermType,
           ListType,
           NumberType,
           IntegerType,
           BinaryType,
           AtomType,
           BooleanType,
           FixnumType,
           BigIntType,
           FloatType,
           NilType,
           ConsType,
           TupleType,
           MapType,
           ClosureType,
           BinaryType,
           HeapBinType,
           ProcBinType,
           BoxType>();

  addAttributes<AtomAttr,
                BinaryAttr,
                SeqAttr>();
}

void EirDialect::printAttribute(Attribute attr, DialectAsmPrinter &p) const {
  auto &os = p.getStream();
  switch (attr.getKind()) {
  case AttributeKind::Atom: {
    auto atomAttr = attr.cast<AtomAttr>();
    os << AtomAttr::getAttrName() << '<';
    os << "{ id = " << atomAttr.getValue();
    auto name = atomAttr.getStringValue();
    if (name.size() > 0) {
      os << ", value = '" << name << "'";
    }
    os << " }>";
  } break;
  case AttributeKind::Binary: {
    auto binAttr = attr.cast<BinaryAttr>();
    auto bytes = binAttr.getValue();
    auto size = bytes.size();
    os << BinaryAttr::getAttrName() << '<';
    os << '[';
    if (size > 0) {
      os << "0x";
      for (unsigned i = 0; i < size; i++) {
        auto c = bytes[i];
        os << llvm::format_hex_no_prefix(bytes[i], 2, true);
      }
    }
    os << ']';
  } break;
  case AttributeKind::Seq: {
    auto seqAttr = attr.cast<SeqAttr>();
    os << SeqAttr::getAttrName() << '<';
    os << '[';
    auto count = seqAttr.size();
    if (count > 0) {
      auto elements = seqAttr.getValue();
      bool printSeparator = count > 1;
      for (unsigned i = 0; i < count; i++) {
        p.printAttribute(elements[i]);
        if (i != (count - 1))
          os << ", ";
      }
    }
    os << "]>";
  } break;
  default:
    llvm_unreachable("unhandled EIR type");
  }
}
