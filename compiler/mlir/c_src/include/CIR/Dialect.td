#ifndef CIR_DIALECT
#define CIR_DIALECT

include "mlir/IR/OpBase.td"

def CIR_Dialect : Dialect {
    let name = "cir";
    let summary = "A direct representation of Lumen's SSA IR in the MLIR framework";
    let description = [{
        This dialect is meant to be as direct a lowering as possible from Rust into
        MLIR, and is where the bulk of our optimizations and target-specific lowerings
        occur.
    }];
    let cppNamespace = "::mlir::cir";
    let dependentDialects = ["func::FuncDialect", "cf::ControlFlowDialect", "scf::SCFDialect"];
    let hasConstantMaterializer = 1;
    let extraClassDeclaration = [{
    private:
      void registerAttributes();
      void registerTypes();
      void registerOperations();
      void registerInterfaces();
    public:
    }];
}

class CIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<CIR_Dialect, mnemonic, traits>;

#endif
