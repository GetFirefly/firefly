#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"

namespace lumen {
namespace eir {
namespace tablegen {

using ::llvm::formatv;
using ::llvm::Record;
using ::mlir::tblgen::EnumAttrCase;

bool emitEncodingDefs(const llvm::RecordKeeper &recordKeeper, llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("EIR Term Encoding Definitions", os);

  auto flags = recordKeeper.getAllDerivedDefinitions("eir_EC");
  auto numFlags = flags.size();

  os << "#ifndef EIR_ENCODING_FLAG\n";
  os << "#define EIR_ENCODING_FLAG(FLAG, VAL)\n";
  os << "#define FIRST_EIR_ENCODING_FLAG(FLAG, VAL) EIR_ENCODING_FLAG(FLAG, VAL)\n";
  os << "#endif\n\n";
  unsigned flg = 0;
  for (const auto *def : flags) {
    EnumAttrCase ec(def);

    if (flg == 0) {
        os << formatv("FIRST_EIR_ENCODING_FLAG({0}, {1})\n",
                      ec.getSymbol(),
                      llvm::format_hex(ec.getValue(), 4, true));
    } else {
        os << formatv("EIR_ENCODING_FLAG({0}, {1})\n",
                      ec.getSymbol(),
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
        os << formatv("FIRST_EIR_TERM_KIND({0}, {1})", ec.getSymbol(), ec.getValue());
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


}  // namespace tablegen
}  // namespace eir
}  // namespace lumen

static mlir::GenRegistration genEncodingDefs(
    "gen-eir-encoding-defs",
    "Generates EIR term encoding definitions (.cpp)",
    [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
      return lumen::eir::tablegen::emitEncodingDefs(records, os);
    });
