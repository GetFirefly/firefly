mod atom_table;
mod exceptions;
mod symbol_table;

use std::collections::HashSet;
use std::path::Path;

use libeir_intern::Symbol;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_llvm::target::TargetMachine;
use liblumen_llvm::Context;
use liblumen_session::Options;

use crate::meta::CodegenResults;
use crate::Result;

pub fn run(
    options: &Options,
    result: &mut CodegenResults,
    context: &Context,
    target_machine: &TargetMachine,
    output_dir: &Path,
    atoms: HashSet<Symbol>,
    symbols: HashSet<FunctionSymbol>,
) -> Result<()> {
    let atom_table = atom_table::generate(options, context, target_machine, atoms, output_dir)?;
    result.modules.push(atom_table);

    let symbol_table =
        symbol_table::generate(options, context, target_machine, symbols, output_dir)?;
    result.modules.push(symbol_table);

    let exception_handler = exceptions::generate(options, context, target_machine, output_dir)?;
    result.modules.push(exception_handler);

    Ok(())
}
