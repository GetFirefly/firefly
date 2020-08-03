#include "lumen/compiler/Dialect/EIR/IR/EIRDialect.h"

#include "llvm/Support/Format.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRAttributes.h"
#include "lumen/compiler/Dialect/EIR/IR/EIROps.h"
#include "lumen/compiler/Dialect/EIR/IR/EIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace lumen::eir;

using ::mlir::Attribute;
using ::mlir::DialectAsmPrinter;

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
  addTypes<::lumen::eir::NoneType, TermType, ListType, NumberType,
           ::lumen::eir::IntegerType, AtomType, ::lumen::eir::BooleanType,
           FixnumType, BigIntType, ::lumen::eir::FloatType, NilType, ConsType,
           TupleType, MapType, ClosureType, BinaryType, HeapBinType,
           ProcBinType, BoxType, RefType, PtrType>();

  addAttributes<AtomAttr, BinaryAttr, SeqAttr>();
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
      os << BinaryAttr::getAttrName();
      os << "<{ value = ";
      if (binAttr.isPrintable()) {
        auto s = binAttr.getValue();
        os << '"' << s << '"';
      } else {
        auto bin = binAttr.getValue();
        auto size = bin.size();
        os << "0x";
        for (char c : bin.bytes()) {
          os << llvm::format_hex_no_prefix(c, 2, true);
        }
      }
      os << " }>";
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
          if (i != (count - 1)) os << ", ";
        }
      }
      os << "]>";
    } break;
    default:
      llvm_unreachable("unhandled EIR type");
  }
}
