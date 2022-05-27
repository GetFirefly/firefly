//mod atom_table;
//mod exceptions;
//mod symbol_table;

use std::collections::HashSet;

use liblumen_intern::Symbol;
use liblumen_llvm::target::TargetMachine;
use liblumen_llvm::Context;
use liblumen_rt::function::FunctionSymbol;
use liblumen_session::Options;

use crate::meta::CodegenResults;

pub fn run(
    _options: &Options,
    _result: &mut CodegenResults,
    _context: &Context,
    _target_machine: TargetMachine,
    _atoms: HashSet<Symbol>,
    _symbols: HashSet<FunctionSymbol>,
) -> anyhow::Result<()> {
    //   let atom_table = atom_table::generate(options, context, target_machine, atoms)?;
    //  result.modules.push(atom_table);

    // let symbol_table = symbol_table::generate(options, context, target_machine, symbols)?;
    //result.modules.push(symbol_table);

    //let exception_handler = exceptions::generate(options, context, target_machine)?;
    //result.modules.push(exception_handler);

    Ok(())
}
