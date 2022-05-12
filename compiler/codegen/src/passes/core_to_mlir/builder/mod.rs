mod function;

use std::collections::{HashMap, HashSet};

use log::debug;

use liblumen_rt::function::FunctionSymbol;
use liblumen_diagnostics::{CodeMap, SourceSpan};
use liblumen_intern::{symbols, Symbol};
use liblumen_llvm::Linkage;
use liblumen_mlir as mlir;
use liblumen_mlir::cir::{CirBuilder, DispatchTableOp};
use liblumen_mlir::llvm::LlvmBuilder;
use liblumen_mlir::{Builder, OpBuilder, Operation, OwnedOpBuilder, Variadic};
use liblumen_session::Options;
use liblumen_syntax_core as syntax_core;

/// This builder holds the state necessary to build an MLIR module
/// from a CIR module.
///
/// It maintains a module-local atom table, and a table of function symbols created during
/// the build. These are later combined with the same tables of other modules to form a
/// global set of atoms and symbols.
pub struct ModuleBuilder<'m> {
    options: &'m Options,
    codemap: &'m CodeMap,
    module: &'m syntax_core::Module,
    mlir_module: mlir::OwnedModule,
    builder: OwnedOpBuilder,
    dispatch_table: DispatchTableOp,
    // The current syntax_core block being translated
    current_source_block: syntax_core::Block,
    // The current MLIR block being built
    current_block: mlir::Block,
    // Used to track the set of atoms used in this module
    #[allow(dead_code)]
    atoms: HashSet<Symbol>,
    // Used to track the set of symbols used in this module
    #[allow(dead_code)]
    symbols: HashSet<FunctionSymbol>,
    // Used to track the mapping of blocks in the current function being translated
    blocks: HashMap<syntax_core::Block, mlir::Block>,
    // Used to track the mapping of values in the current function being translated
    values: HashMap<syntax_core::Value, mlir::ValueBase>,
}
impl<'m> ModuleBuilder<'m> {
    /// Creates a new builder for the given module, using the provided MLIR context
    pub fn new(
        module: &'m syntax_core::Module,
        codemap: &'m CodeMap,
        context: mlir::Context,
        options: &'m Options,
    ) -> Self {
        let builder = OwnedOpBuilder::new(context);
        let module_span = module.span();
        let source_file = codemap
            .get(module_span.start().source_id())
            .expect("invalid module span, no corresponding source file!");
        let source_filename = source_file.name();
        let source_filename_id = source_filename.to_string();
        let name = module.name();

        let mut atoms = HashSet::new();
        atoms.insert(name);

        let module_span = source_file.location(module_span).unwrap();
        let loc = builder.get_file_line_col_loc(
            source_filename_id.as_str(),
            module_span.line.number().to_usize() as u32,
            (module_span.column.to_usize() + 1) as u32,
        );
        let mlir_module = builder.create_module(loc, name);
        let entry_region = mlir_module.body();
        builder.set_insertion_point_to_end(entry_region);

        let dispatch_table = {
            let cir = CirBuilder::new(&builder);
            cir.build_dispatch_table(loc, name)
        };

        Self {
            options,
            codemap,
            module,
            mlir_module,
            builder,
            dispatch_table,
            current_source_block: syntax_core::Block::default(),
            current_block: mlir::Block::default(),
            atoms,
            symbols: HashSet::new(),
            // For both of these maps, we know that most functions will contain a fair number of blocks
            // and values, so we allocate enough capacity to accomodate the size of small/medium functions
            // (probably, this is estimated). Benchmarking in the future should be used to dial this in.
            //
            // NOTE: The size of the keys in these maps is u32, and the values are usize, and there may be
            // some additional overhead for each map entry, so we should try and pick sizes that will result
            // in power-of-two sizes for the allocator to make the most of the allocations
            blocks: HashMap::with_capacity(64),
            values: HashMap::with_capacity(64),
        }
    }

    #[inline]
    pub fn cir(&self) -> CirBuilder<'_, OwnedOpBuilder> {
        CirBuilder::new(&self.builder)
    }

    pub fn find_function(&self, f: syntax_core::FuncRef) -> syntax_core::Signature {
        self.module.call_signature(f).clone()
    }

    pub fn get_or_declare_function(&self, name: &str) -> anyhow::Result<mlir::FuncOp> {
        if let Some(found) = self.builder.get_func_by_symbol(name) {
            assert!(!found.base().is_null());
            return Ok(found);
        }
        let function_name = name.parse().unwrap();
        let callee = self.module.get_callee(function_name).unwrap();
        let sig = self.module.call_signature(callee);
        self.declare_function(self.module.span(), &sig)
    }

    #[inline]
    pub fn location_from_span(&self, span: SourceSpan) -> mlir::Location {
        if span.is_unknown() {
            self.builder.get_unknown_loc()
        } else {
            // Get the source file in which this span belongs
            let source_file = self
                .codemap
                .get_with_span(span)
                .expect("invalid source span, no corresponding source file!");
            // Get the location (i.e. line/col index) which this span represents
            let loc = self.codemap.location_for_span(span).unwrap();
            // Convert the source file name to an MLIR identifier (stringattr)
            let source_filename = source_file.name();
            if let Some(filename) = source_filename.as_str() {
                self.builder.get_file_line_col_loc(
                    filename,
                    loc.line.number().to_usize() as u32,
                    (loc.column.to_usize() + 1) as u32,
                )
            } else {
                let filename = source_filename.to_string();
                self.builder.get_file_line_col_loc(
                    filename.as_str(),
                    loc.line.number().to_usize() as u32,
                    (loc.column.to_usize() + 1) as u32,
                )
            }
        }
    }

    /// Builds an MLIR module from the underlying syntax_core module, consuming the builder
    pub fn build(mut self) -> anyhow::Result<Result<mlir::OwnedModule, mlir::OwnedModule>> {
        use liblumen_mlir::{PassManager, PassManagerOptions};

        let module_name = self.module.name();
        debug!("building mlir module for {}", module_name);

        // Inject declarations for all local functions in advance
        for f in self.module.functions.iter() {
            // Don't declare module_info/0 and module_info/1 for now
            if f.signature.is_local(symbols::ModuleInfo, 0)
                || f.signature.is_local(symbols::ModuleInfo, 1)
            {
                continue;
            }
            let func = self.declare_function(f.span, &f.signature)?;
            if f.signature.is_erlang() {
                func.set_attribute_by_name(
                    "personality",
                    self.builder
                        .get_flat_symbol_ref_attr_by_name("lumen_eh_personality"),
                );
                //TODO: Need to re-enable when garbage collector lowering is implemented
                //func.set_attribute_by_name("garbageCollector", self.builder.get_string_attr("erlang"));
            }

            // Register with the dispatch table for this module if public
            if f.signature.visibility.is_public() {
                let name = f.signature.mfa().to_string();
                self.dispatch_table.append(
                    self.location_from_span(f.span),
                    self.builder.get_string_attr(f.signature.name),
                    self.builder.get_i8_attr(f.signature.arity() as i8),
                    self.builder.get_flat_symbol_ref_attr_by_name(name.as_str()),
                );
            }
        }

        // Inject declaration for personality function
        {
            let loc = self.location_from_span(self.module.span());
            let _ip = self.builder.insertion_guard();
            self.builder
                .set_insertion_point_to_end(self.mlir_module.body());
            let i32ty = self.builder.get_i32_type();
            let llvm = LlvmBuilder::new(&self.builder);
            let ty = llvm.get_function_type(i32ty, &[], Variadic::Yes);
            llvm.build_func(loc, "lumen_eh_personality", ty, Linkage::External, &[]);
        }

        // Then build them out
        for f in self.module.functions.iter() {
            // Don't generate module_info/0 and module_info/1 for now
            if f.signature.is_local(symbols::ModuleInfo, 0)
                || f.signature.is_local(symbols::ModuleInfo, 1)
            {
                continue;
            }
            self.build_function(f)?;
        }

        debug!("finished building {}, cleaning up..", module_name);

        // Create a pass manager to perform canonicalization and other cleanups
        let mut pm_opts = PassManagerOptions::new(self.options);
        pm_opts.enable_verifier = false;
        let mut pm = PassManager::new(self.mlir_module.context(), &pm_opts);
        pm.parse_pipeline("builtin.module(func.func(canonicalize))")
            .unwrap();
        //opt_pm.add(InlinerPass::new());
        //opt_pm.add(ControlFlowSinkPass::new());

        let successful = pm.run(&self.mlir_module);
        if !successful {
            debug!(
                "failed to run canonicalization and other fixups on {}",
                module_name
            );
            return Ok(Err(self.mlir_module));
        }

        debug!("validating {}..", module_name);

        if self.mlir_module.is_valid() {
            Ok(Ok(self.mlir_module))
        } else {
            Ok(Err(self.mlir_module))
        }
    }
}
