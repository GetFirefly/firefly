#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"

namespace lumen {
namespace eir {
namespace tablegen {

using ::llvm::formatv;
using ::llvm::Record;
using ::mlir::tblgen::EnumAttrCase;

bool emitEncodingDefs(const llvm::RecordKeeper &recordKeeper,
                      llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("EIR Term Encoding Definitions", os);

  auto flags = recordKeeper.getAllDerivedDefinitions("eir_EC");
  auto numFlags = flags.size();

  os << "#ifndef EIR_ENCODING_FLAG\n";
  os << "#define EIR_ENCODING_FLAG(FLAG, VAL)\n";
  os << "#define FIRST_EIR_ENCODING_FLAG(FLAG, VAL) EIR_ENCODING_FLAG(FLAG, "
        "VAL)\n";
  os << "#endif\n\n";
  unsigned flg = 0;
  for (const auto *def : flags) {
    EnumAttrCase ec(def);

    if (flg == 0) {
      os << formatv("FIRST_EIR_ENCODING_FLAG({0}, {1})\n", ec.getSymbol(),
                    llvm::format_hex(ec.getValue(), 4, true));
    } else {
      os << formatv("EIR_ENCODING_FLAG({0}, {1})\n", ec.getSymbol(),
                    llvm::format_hex(ec.getValue(), 4, true));
    }
    flg++;
  }
  os << "\n\n";
  os << "#undef EIR_ENCODING_FLAG\n";
  os << "#undef FIRST_EIR_ENCODING_FLAG\n\n";

  auto kinds = recordKeeper.getAllDerivedDefinitions("eir_TermKind");
  auto numKinds = kinds.size();

  os << "#ifndef EIR_TERM_KIND\n";
  os << "#define EIR_TERM_KIND(KIND, VAL)\n";
  os << "#define FIRST_EIR_TERM_KIND(KIND, VAL) EIR_TERM_KIND(KIND, VAL)\n";
  os << "#endif\n\n";
  unsigned k = 0;
  for (const auto *def : kinds) {
    EnumAttrCase ec(def);

    if (k == 0) {
      os << formatv("FIRST_EIR_TERM_KIND({0}, {1})", ec.getSymbol(),
                    ec.getValue());
    } else {
      os << formatv("EIR_TERM_KIND({0}, {1})", ec.getSymbol(), ec.getValue());
    }
    k++;
    if (k < numKinds) {
      os << "  \\\n";
    }
  }
  os << "\n\n";
  os << "#undef EIR_TERM_KIND\n";
  os << "#undef FIRST_EIR_TERM_KIND\n\n";

  return false;
}

bool emitRustEncodingDefs(const llvm::RecordKeeper &recordKeeper,
                          llvm::raw_ostream &os) {
  auto kinds = recordKeeper.getAllDerivedDefinitions("eir_TermKind");
  auto numKinds = kinds.size();

  // TermKind enum, used for exchanging type kind between frontend/backend
  os << "#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash)]\n";
  os << "#[repr(C)]\n";
  os << "pub enum TermKind {\n";
  for (const auto *def : kinds) {
    EnumAttrCase ec(def);

    os << formatv("    {0} = {1},\n", ec.getSymbol(), ec.getValue());
  }
  os << "}\n\n";
  os << "impl core::convert::TryFrom<u32> for TermKind {\n";
  os << "    type Error = ();\n";
  os << "    fn try_from(value: u32) -> core::result::Result<Self, "
        "Self::Error> {\n";
  os << "        match value {\n";
  for (const auto *def : kinds) {
    EnumAttrCase ec(def);

    os << formatv("            {0} => Ok(Self::{1}),\n", ec.getValue(),
                  ec.getSymbol());
  }
  os << "            _ => Err(()),\n";
  os << "        }\n";
  os << "    }\n";
  os << "}\n\n";

  // Type enum, used for communicating type information from EIR to MLIR
  os << "#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Hash)]\n";
  os << "#[repr(u32)]\n";
  os << "pub enum Type {\n";
  for (const auto *def : kinds) {
    EnumAttrCase ec(def);

    if (ec.getSymbol() == "Tuple") {
      os << formatv("    {0}(u32) = {1},\n", ec.getSymbol(), ec.getValue());
    } else {
      os << formatv("    {0} = {1},\n", ec.getSymbol(), ec.getValue());
    }
  }
  os << "}\n\n";

  return false;
}

}  // namespace tablegen
}  // namespace eir
}  // namespace lumen

static mlir::GenRegistration genEncodingDefs(
    "gen-eir-encoding-defs", "Generates EIR term encoding definitions (.cpp)",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return lumen::eir::tablegen::emitEncodingDefs(records, os);
    });

static mlir::GenRegistration genRustEncodingDefs(
    "gen-rust-eir-encoding-defs",
    "Generates EIR term encoding definitions (.rs)",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return lumen::eir::tablegen::emitRustEncodingDefs(records, os);
    });
