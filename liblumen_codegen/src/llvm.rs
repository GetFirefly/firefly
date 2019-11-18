pub mod enums;
pub mod target;

use std::path::Path;

use inkwell::module::Module;
use inkwell::passes::{PassManagerBuilder, PassManager};
use inkwell::support::LLVMString;

use self::enums::OptimizationLevel;

/// Parses an LLVM bitcode file into a `Module`
#[allow(unused)]
#[inline]
pub fn parse_bitcode_from_path<P: AsRef<Path>>(path: P) -> Result<Module, LLVMString> {
    Module::parse_bitcode_from_path(path)
}

/// Runs optimizations against an LLVM module
#[allow(unused)]
pub fn optimize(module: &Module, level: OptimizationLevel) {
    // Create builder for function passes
    let builder = PassManagerBuilder::create();
    builder.set_optimization_level(level);
    // Magic threshold from Clang for -02
    builder.set_inliner_with_threshold(225);
    // Create PassManager and populate with passes
    let fpm = PassManager::create(module);
    builder.populate_function_pass_manager(&fpm);
    if !fpm.initialize() {
        panic!("Unable to initialize function pass manager!");
    }
    // Run passes on functions first
    let mut fun = module.get_first_function();
    while let Some(f) = fun {
        fpm.run_on(&f);
        fun = f.get_next_function();
    }
    if !fpm.finalize() {
        panic!("Failed to finalize function pass manager!");
    }

    // Create builder for module passes
    let builder = PassManagerBuilder::create();
    builder.set_optimization_level(level);
    // Magic threshold from Clang for -02
    builder.set_inliner_with_threshold(225);
    let pm = PassManager::create(());
    builder.populate_module_pass_manager(&pm);
    if !pm.initialize() {
        panic!("Unable to initialize module pass manager!");
    }
    // Then run passes on the whole module
    pm.run_on(module);
    if !pm.finalize() {
        panic!("Failed to finalize module pass manager!");
    }
}

/// Perform link-time optimization (LTO) on a module
#[allow(unused)]
pub fn lto(module: &Module, level: OptimizationLevel) {
    let builder = PassManagerBuilder::create();
    builder.set_optimization_level(level);
    builder.set_inliner_with_threshold(225);

    let internalize = true;
    let run_inliner = true;
    let pm = PassManager::create(());
    builder.populate_lto_pass_manager(&pm, internalize, run_inliner);

    pm.run_on(module);
}
