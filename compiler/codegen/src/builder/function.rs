mod function;
pub use self::function::*;

use std::mem;
use std::ptr;
use std::sync::Arc;

use anyhow::anyhow;

use log::debug;

use libeir_diagnostics::{ByteIndex, FileMap};
use libeir_intern::{Ident, Symbol};
use libeir_ir as ir;
use libeir_ir::{AtomTerm, AtomicTerm, ConstKind, FunctionIdent};
use libeir_lowerutils::{FunctionData, LowerData};
use libeir_util_datastructures::pooled_entity_set::BoundEntitySet;

use liblumen_mlir::ir::*;
use liblumen_session::Options;

use crate::Result;

use super::block::{Block, BlockData};
use super::ffi::*;
use super::ops::builders::{ClosureBuilder, ConstantBuilder};
use super::ops::*;
use super::value::{Value, ValueData, ValueDef};
use super::ModuleBuilder;

/// The builder type used for lowering EIR functions to MLIR functions
///
/// Internally, it delegates the construction of the main function and
/// its lifted counterparts (closures) using `ScopedFunctionBuilder`.
pub struct FunctionBuilder<'a, 'm, 'f> {
    func: &'f ir::FunctionDefinition,
    builder: &'a mut ModuleBuilder<'m>,
}
impl<'a, 'm, 'f> FunctionBuilder<'a, 'm, 'f> {
    pub fn new(func: &'f ir::FunctionDefinition, builder: &'a mut ModuleBuilder<'m>) -> Self {
        Self { func, builder }
    }

    pub fn build(mut self, options: &Options) -> Result<()> {
        let f = self.func.function();
        let ident = f.ident();
        {
            self.builder.atoms_mut().insert(ident.name.name);
        }

        debug!("{}: building..", &ident);

        debug!("{}: performing lowering analysis..", &ident);
        let analysis = libeir_lowerutils::analyze(f);
        let loc = Span::from(f.span());

        // Gather atoms in this function and add them to the atom table for this module
        debug!("{}: gathering atoms for atom table", &ident);
        for val in f.iter_constants() {
            if let ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(s))) = value_to_const_kind(f, *val) {
                debug!("{}: found atom: {:?}", &ident, *s);
                self.builder.atoms_mut().insert(*s);
            }
        }

        let root_block = f.block_entry();
        for (index, (entry_block, data)) in analysis.functions.iter().enumerate() {
            let entry_block = *entry_block;
            let func = if entry_block == root_block {
                self.with_scope(ident.clone(), loc, f, &analysis, data, options)
                    .and_then(|scope| scope.build())?
            } else {
                let arity = f.block_args(entry_block).len() - 2;
                let fun = Ident::from_str(&format!("{}-fun-{}-{}", ident.name, index, arity));
                let fi = FunctionIdent {
                    module: ident.module.clone(),
                    name: fun,
                    arity,
                };
                {
                    self.builder.atoms_mut().insert(fi.name.name);
                }
                self.with_scope(fi, loc, f, &analysis, data, options)
                    .and_then(|scope| scope.build())?
            };
            unsafe { MLIRAddFunction(self.builder.as_ref(), func) }
        }

        Ok(())
    }

    pub fn with_scope<'s, 'o>(
        &mut self,
        name: FunctionIdent,
        loc: Span,
        eir: &'s ir::Function,
        analysis: &'s LowerData,
        data: &'s FunctionData,
        options: &'o Options,
    ) -> Result<ScopedFunctionBuilder<'s, 'o>> {
        debug!("entering scope for {}", &name);
        debug!("entry = {:?}", &data.entry);
        debug!("scope = {:?}", &data.scope);

        let ret = data
            .ret
            .expect("expected function to have return continuation");
        let esc = data
            .thr
            .expect("expected function to have escape continuation");

        let is_closure = analysis.live.live_at(data.entry).size() > 0;

        // Construct signature
        let mut signature = Signature::new(CallConv::Fast);
        let entry_args = eir.block_args(data.entry);
        {
            if is_closure {
                // Remove the return/escape continuation parameters,
                // and add one parameter, the closure env
                signature.params.reserve(entry_args.len() - 2 + 1);
                signature.params.push(Param {
                    ty: Type::Box,
                    span: Span::default(),
                    is_implicit: false,
                });
            } else {
                // Remove the return/escape continuation parameters
                signature.params.reserve(entry_args.len() - 2);
            }
            for arg in entry_args.iter().skip(2).copied() {
                signature
                    .params
                    .push(block_arg_to_param(eir, arg, /* is_implicit */ false));
            }
            signature.returns.push(Type::Term);
        }

        // Construct the parameter value metadata
        let mut entry_params = Vec::with_capacity(signature.params.len());
        let entry_arg_offset = if is_closure {
            entry_params.push((signature.params[0].clone(), None));
            1
        } else {
            0
        };
        for (i, v) in entry_args.iter().skip(2).copied().enumerate() {
            let offs = entry_arg_offset + i;
            entry_params.push((signature.params[offs].clone(), Some(v)));
        }

        // Create function
        let mut func = Function::with_name_signature(eir.span(), name, signature);
        let (mlir, entry_ref) = func.build(self.builder)?;

        // Mirror the entry block for our init block
        let init_block =
            func.new_block_with_params(Some(data.entry), entry_ref, entry_params.as_slice());
        // Initialize ret/esc continuations
        let ret = func.set_return_continuation(ret, init_block);
        let esc = func.set_escape_continuation(esc, init_block);

        Ok(ScopedFunctionBuilder {
            filemap: self.builder.filemap().clone(),
            filename: self.builder.filename().as_ptr(),
            func,
            name,
            loc,
            eir,
            mlir,
            analysis,
            data,
            builder: self.builder.as_ref(),
            options,
            pos: Position::at(init_block),
            ret,
            esc,
            is_closure,
        })
    }
}

/// This builder type is essentially a sub-type of FunctionBuilder, and handles
/// lowering EIR functions with a closed scope. In other words, EIR functions can
/// contain multiple functions due to closures; a "scoped" function is just a
/// function that contains no closures. A FunctionBuilder lowers each scoped
/// function using the ScopedFunctionBuilder.
pub struct ScopedFunctionBuilder<'f, 'o> {
    filename: *const libc::c_char,
    filemap: Arc<FileMap>,
    func: Function,
    name: FunctionIdent,
    loc: Span,
    eir: &'f ir::Function,
    mlir: FunctionOpRef,
    analysis: &'f LowerData,
    data: &'f FunctionData,
    builder: ModuleBuilderRef,
    options: &'o Options,
    pos: Position,
    ret: Value,
    esc: Value,
    is_closure: bool,
}

// Miscellaneous helper functions
impl<'f, 'o> ScopedFunctionBuilder<'f, 'o> {
    /// Returns the internal MLIR builder this builder wraps
    pub fn as_ref(&self) -> ModuleBuilderRef {
        self.builder
    }

    /// Returns the current compiler options
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Returns the current function identifier
    #[inline]
    pub fn name(&self) -> &FunctionIdent {
        self.func.name()
    }

    /// Prints debugging messages with the name of the function being built
    #[cfg(debug_assertions)]
    pub(super) fn debug(&self, message: &str) {
        debug!("{}: {}", self.name(), message);
    }

    #[cfg(not(debug_assertions))]
    pub(super) fn debug(&self, _message: &str) {}

    fn location(&self, index: ByteIndex) -> Option<SourceLocation> {
        let (li, ci) = self.filemap.location(index).ok()?;
        Some(SourceLocation {
            filename: self.filename,
            line: li.number().to_usize() as u32,
            column: ci.number().to_usize() as u32,
        })
    }
}

// EIR function metadata helpers
impl<'f, 'o> ScopedFunctionBuilder<'f, 'o> {
    /// Gets the set of EIR values that are live at the given block
    #[inline]
    pub fn live_at(&self, block: Block) -> BoundEntitySet<'f, ir::Value> {
        let ir_block = self.func.block_to_ir_block(block).unwrap();
        self.analysis.live.live_at(ir_block)
    }

    /// Same as `live_at`, but takes an EIR block as argument instead
    #[inline]
    pub fn ir_live_at(&self, ir_block: ir::Block) -> BoundEntitySet<'f, ir::Value> {
        self.analysis.live.live_at(ir_block)
    }

    /// Gets the set of EIR values that are live in the given block
    #[inline]
    pub fn live_in(&self, block: Block) -> BoundEntitySet<'f, ir::Value> {
        let ir_block = self.get_ir_block(block);
        self.analysis.live.live_in(ir_block)
    }

    /// Returns true if the given value is live at the given block
    ///
    /// Panics if either the value or block do not have EIR representation
    #[inline]
    pub fn is_live_at(&self, block: Block, value: Value) -> bool {
        let ir_block = self.get_ir_block(block);
        let ir_value = self.get_ir_value(value);
        self.analysis.live.is_live_at(ir_block, ir_value)
    }

    /// Returns true if the given value is live in the given block
    ///
    /// Panics if either the value or block do not have EIR representation
    #[inline]
    pub fn is_live_in(&self, block: Block, value: Value) -> bool {
        let ir_block = self.get_ir_block(block);
        let ir_value = self.get_ir_value(value);
        self.analysis.live.is_live_in(ir_block, ir_value)
    }

    /// Finds the EIR block the given block represents
    pub fn get_ir_block(&self, block: Block) -> ir::Block {
        self.func.block_to_ir_block(block).unwrap()
    }

    /// Maps a given entry block to its corresponding function identifier
    ///
    /// NOTE: This is intended for use only with blocks that are the capture
    /// target of a function call, i.e. a call to a closure. It does _not_ accept
    /// arbitrary blocks.
    pub fn block_to_closure_info(&self, block: ir::Block) -> ClosureInfo {
        for (i, (entry_block, _data)) in self.analysis.functions.iter().enumerate() {
            let entry_block = *entry_block;
            if entry_block == block {
                let containing_ident = self.eir.ident();
                let arity = self.eir.block_args(entry_block).len() - 2;
                let index = i as u32;
                let fun = Ident::from_str(&format!(
                    "{}-fun-{}-{}",
                    containing_ident.name, index, arity
                ));
                let ident = FunctionIdent {
                    module: containing_ident.module.clone(),
                    name: fun,
                    arity,
                };
                let unique = unsafe {
                    mem::transmute::<[u64; 2], [u8; 16]>([
                        fxhash::hash64(&ident),
                        fxhash::hash64(&index),
                    ])
                };
                let old_unique = fxhash::hash32(&unique);
                return ClosureInfo {
                    ident,
                    index,
                    old_unique,
                    unique,
                };
            }
        }
        panic!("expected block to correspond to the entry block in this functions' scope");
    }

    /// Finds the EIR value the given value represents
    ///
    /// Panics if the value does not have an EIR representation
    #[inline]
    pub fn get_ir_value(&self, value: Value) -> ir::Value {
        self.func.value_to_ir_value(value).unwrap()
    }

    /// Gets the value metadata for the given EIR value
    #[inline]
    pub fn value_kind(&self, ir_value: ir::Value) -> ir::ValueKind {
        self.eir.value_kind(ir_value)
    }

    /// Gets the location data for the given EIR value
    pub fn value_location(&self, ir_value: ir::Value) -> LocationRef {
        if let Some(locs) = self.eir.value_locations(ir_value) {
            let mut fused = Vec::with_capacity(locs.len());
            for loc in locs.iter().copied() {
                if let Some(sc) = self.location(loc.start()) {
                    fused.push(unsafe { MLIRCreateLocation(self.builder, sc) });
                }
            }
            if fused.len() > 0 {
                return unsafe {
                    MLIRCreateFusedLocation(
                        self.builder,
                        fused.as_ptr(),
                        fused.len() as libc::c_uint,
                    )
                };
            }
        }
        self.unknown_value_location()
    }

    #[inline(always)]
    pub fn unknown_value_location(&self) -> LocationRef {
        unsafe { MLIRUnknownLocation(self.builder) }
    }

    /// Returns true if the given EIR value is defined by an argument of the given block
    pub fn is_block_argument(&self, ir_value: ir::Value, block: Block) -> bool {
        for arg in self.eir.block_args(self.get_ir_block(block)) {
            if ir_value == *arg {
                return true;
            }
        }
        false
    }

    /// Gets the constant reference represented by the given value
    ///
    /// Panics if the value is not a constant
    #[inline]
    pub fn value_const(&self, ir_value: ir::Value) -> ir::Const {
        self.eir
            .value_const(ir_value)
            .expect("expected constant value")
    }

    /// Gets the raw symbol for the given constant atom
    ///
    /// Panics if the constant is not an atom
    pub fn constant_atom(&self, constant: ir::Const) -> Symbol {
        if let ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(a))) = self.eir.const_kind(constant) {
            *a
        } else {
            panic!("expected constant {:?} to be an atom", constant);
        }
    }

    /// Gets the raw value for the given constant integer
    ///
    /// Panics if the constant is not an integer
    pub fn constant_int(&self, constant: ir::Const) -> i64 {
        use ir::IntTerm;
        if let ConstKind::Atomic(AtomicTerm::Int(IntTerm(i))) = self.eir.const_kind(constant) {
            *i
        } else {
            panic!("expected constant {:?} to be an integer", constant);
        }
    }

    /// Returns the constant kind for the given const reference
    #[inline]
    pub fn const_kind(&self, constant: ir::Const) -> &ir::ConstKind {
        self.eir.const_kind(constant)
    }

    /// Returns the set of EIR constants for the given slice of const references
    #[inline]
    pub fn const_entries(
        &'f self,
        entries: &'f cranelift_entity::EntityList<ir::Const>,
    ) -> &'f [ir::Const] {
        self.eir.const_entries(entries)
    }

    /// Gets the primop reference represented by the given value
    ///
    /// Panics if the value is not a primop
    #[inline]
    pub fn get_primop(&self, value: ir::Value) -> ir::PrimOp {
        self.eir.value_primop(value).expect("expected primop value")
    }

    /// Gets primop metadata for the given primop reference
    #[inline]
    pub fn primop_kind(&self, primop: ir::PrimOp) -> &ir::PrimOpKind {
        self.eir.primop_kind(primop)
    }

    /// Gets the set of value reads performed by the given primop
    #[inline]
    pub fn primop_reads(&self, primop: ir::PrimOp) -> &[ir::Value] {
        self.eir.primop_reads(primop)
    }
}

// MLIR function metadata helpers
impl<'f, 'o> ScopedFunctionBuilder<'f, 'o> {
    /// Returns the block data for the given block
    #[inline]
    pub fn block_data(&self, block: Block) -> &BlockData {
        self.func.block_data(block)
    }

    /// Returns the MLIR block reference for the given block
    #[inline]
    pub fn block_ref(&self, block: Block) -> BlockRef {
        self.func.block_to_block_ref(block)
    }

    /// Returns the value data for the given value
    #[inline]
    pub fn value_data(&self, value: Value) -> &ValueData {
        self.func.value_data(value)
    }

    /// Returns the MLIR value reference for the given value
    #[inline]
    pub fn value_ref(&self, value: Value) -> ValueRef {
        self.func.value_to_value_ref(value)
    }

    /// Gets the Value corresponding to the given EIR value in the current block
    ///
    /// Panics if the value doesn't exist
    pub fn get_value(&self, ir_value: ir::Value) -> Value {
        self.find_value(ir_value)
            .expect("expected value to be defined in the current block")
    }

    /// Searches for the Value corresponding to the given EIR value in the current block
    ///
    /// Returns None if it is not defined in this block (yet)
    pub fn find_value(&self, ir_value: ir::Value) -> Option<Value> {
        let blocks = &self.func.value_mapping[ir_value];
        let mut defs = blocks
            .values()
            .cloned()
            .filter_map(|o| o.expand())
            .collect::<Vec<_>>();
        assert!(
            defs.len() < 2,
            "expected no more than one definition per eir value"
        );
        defs.pop()
    }

    /// Gets the Block corresponding to the given EIR value
    ///
    /// It is expected that the value is an EIR block reference.
    ///
    /// Panics if the value is not a block, or if the block doesn't exist
    pub fn get_block_by_value(&self, ir_value: ir::Value) -> Block {
        self.eir
            .value_block(ir_value)
            .map(|b| self.get_block(b))
            .expect(&format!(
                "the given value is not a known block: {:?} ({:?})",
                ir_value,
                self.eir.value_kind(ir_value),
            ))
    }

    /// Gets the block corresponding to the given EIR block
    ///
    /// Panics if the block doesn't exist
    #[inline]
    pub fn get_block(&self, ir_block: ir::Block) -> Block {
        self.func
            .block_mapping
            .get(ir_block)
            .copied()
            .unwrap_or_else(|| panic!("eir block has no corresponding mlir block{:?}", ir_block))
    }

    /// Registers an EIR value with the MLIR value that corresponds to it
    #[inline]
    pub fn new_value(&mut self, ev: Option<ir::Value>, mv: ValueRef, data: ValueDef) -> Value {
        self.func.new_value(self.current_block(), ev, mv, data)
    }

    /// Returns true if the given symbol is the same as the current functions' module name
    pub fn is_current_module(&self, m: Symbol) -> bool {
        self.func.name().module.name == m
    }

    /// Returns the current block arguments as values
    #[inline]
    pub fn block_args(&self, block: Block) -> Vec<Value> {
        let block_data = self.func.block_data(block);
        block_data.param_values.clone()
    }

    /// Returns the (EIR) block arguments for a given EIR block
    #[inline]
    pub fn ir_block_args(&self, block: ir::Block) -> &[ir::Value] {
        self.eir.block_args(block)
    }

    /// Returns the current block
    #[inline]
    pub fn current_block(&self) -> Block {
        self.pos
            .current()
            .expect("called current_block when not positioned in a block!")
    }
}

// Builder implementation
impl<'f, 'o> ScopedFunctionBuilder<'f, 'o> {
    /// Builds the function
    pub fn build(mut self) -> Result<FunctionOpRef> {
        debug_in!(self, "building..");

        // NOTE: We start out in the entry block
        let entry_block = self.current_block();

        // If this is a closure, extract the environment
        // A closure will have more than 1 live value, otherwise it is a regular function
        let live_at = self.live_at(entry_block);
        debug_in!(
            self,
            "found {} live values at the entry block",
            live_at.size()
        );
        if live_at.size() > 0 {
            self.unpack_closure_env(entry_block, &live_at)?;
        }

        let root_block = self.data.entry;
        debug_in!(self, "root block = {:?}", root_block);
        debug_in!(self, "entry block = {:?}", entry_block);

        // Make sure all blocks are created first
        let mut blocks = Vec::with_capacity(self.data.scope.len());
        blocks.push((root_block, entry_block));
        for ir_block in self.data.scope.iter().copied() {
            // We've already taken care of the entry block
            if ir_block == root_block {
                continue;
            }
            blocks.push(self.prepare_block(ir_block)?);
        }

        // Then finish filling out the blocks
        for (ir_block, block) in blocks.drain(..) {
            self.build_block(block, ir_block)?;
        }

        Ok(self.mlir)
    }

    fn unpack_closure_env(
        &mut self,
        entry: Block,
        live_at: &BoundEntitySet<'f, ir::Value>,
    ) -> Result<()> {
        debug_in!(self, "unpacking closure environment: {:?}", live_at);
        for live in live_at.iter() {
            debug_in!(
                self,
                "env value origin: {:?} <= {:?}",
                live,
                self.eir.value_kind(live)
            );
            debug_assert_eq!(
                None,
                self.find_value(live),
                "expected env value to be unmapped at entry block"
            );
        }

        let loc = {
            let (li, ci) = self
                .filemap
                .location(self.func.span.start())
                .expect("expected source span for function");
            let loc = SourceLocation {
                filename: self.filename,
                line: li.number().to_usize() as u32,
                column: ci.number().to_usize() as u32,
            };
            unsafe { MLIRCreateLocation(self.builder, loc) }
        };

        let env_value = self
            .block_args(entry)
            .get(0)
            .copied()
            .expect("expected closure env argument from block");
        let env = self.value_ref(env_value);
        let num_values = live_at.size();
        let mut values: Vec<ValueRef> = Vec::with_capacity(num_values);
        unsafe {
            let result = MLIRBuildUnpackEnv(
                self.builder,
                loc,
                env,
                values.as_mut_ptr(),
                num_values as libc::c_uint,
            );
            if !result {
                return Err(anyhow!("failed to unpack closure environment"));
            }
            values.set_len(num_values);
        }
        for (i, (ir_value, value_ref)) in live_at.iter().zip(values.iter()).enumerate() {
            debug_in!(self, "env value mapped: {:?} => {:?}", ir_value, value_ref);
            self.new_value(Some(ir_value), *value_ref, ValueDef::Env(i));
        }

        Ok(())
    }

    // Prepares an MLIR block for lowering from an EIR block
    //
    // This function creates a new block, and handles promoting
    // implicit arguments to explicit arguments in the translation
    // from EIR.
    fn prepare_block(&mut self, ir_block: ir::Block) -> Result<(ir::Block, Block)> {
        debug_in!(self, "preparing block {:?}", ir_block);
        // Construct parameter list for the block
        // The result is implicit arguments followed by explicit arguments
        let block_args = self.eir.block_args(ir_block);
        debug_in!(self, "prepare: block_args: {:?}", &block_args);
        // Build the final parameter list
        let mut params = Vec::with_capacity(block_args.len());
        params.extend(block_args.iter().copied().map(|a| {
            (
                block_arg_to_param(self.eir, a, /* is_implicit */ false),
                Some(a),
            )
        }));

        // Construct a new MLIR block to match the EIR block
        let block = self.create_block(Some(ir_block), params.as_slice())?;

        Ok((ir_block, block))
    }

    /// Creates a new block
    ///
    /// Can optionally track the new block as corresponding to a given EIR block.
    ///
    /// The parameter info given will be used to instantiate the block parameter list,
    /// and track source EIR value information
    fn create_block(
        &mut self,
        ir_block: Option<ir::Block>,
        param_info: &[(Param, Option<ir::Value>)],
    ) -> Result<Block> {
        debug_in!(
            self,
            "creating block {:?} with params {:?}",
            ir_block,
            param_info
        );

        let params = param_info
            .iter()
            .map(|(p, _)| p.clone())
            .collect::<Vec<_>>();
        let params_ptr = match params.len() {
            0 => ptr::null(),
            _ => params.as_ptr(),
        };
        let block_ref = unsafe {
            MLIRAppendBasicBlock(
                self.builder,
                self.mlir,
                params_ptr,
                params.len() as libc::c_uint,
            )
        };
        assert!(!block_ref.is_null());

        debug_in!(self, "created block ref {:?}", block_ref);

        let block = self
            .func
            .new_block_with_params(ir_block, block_ref, param_info);

        debug_in!(self, "created block {:?}", block);

        self.position_at_end(block);

        Ok(block)
    }

    /// Positions the builder at the end of the given block
    fn position_at_end(&mut self, block: Block) {
        debug_in!(self, "positioning builder at end of block {:?}", block);
        let block_ref = self.block_ref(block);
        self.pos.position_at_end(self.builder, block, block_ref);
    }

    /// This is the entry point for populating a block with instructions.
    ///
    /// Each EIR block has a primary operation it performs, which also
    /// acts as a terminator for that block. Each primary operation has
    /// a set of value reads it uses to fulfill its purpose. Those reads
    /// may themselves be primitive operations (non-terminator ops) which
    /// need to be recursively lowered.
    ///
    /// When this is called, all function blocks have been created, so we
    /// assert that any block value an operation requires, is a known block.
    /// Each value we encounter may or may not have been lowered yet, so we
    /// delegate to `build_value` to handle looking up the existing definition
    /// if it exists, or if not, lowering the value and defining it for the
    /// remainder of the block.
    ///
    /// Use of `get_value` should not occur here, unless we know that a value
    /// _must_ be defined in the block already - namely, that the value must
    /// be a block argument.
    fn build_block(&mut self, block: Block, ir_block: ir::Block) -> Result<()> {
        debug_in!(self, "building block {:?} (origin = {:?})", block, ir_block);
        // Switch to the block
        self.position_at_end(block);
        // Get the set of values this block reads in its body
        let reads = self.eir.block_reads(ir_block);
        let num_reads = reads.len();
        let loc = self.value_location(self.eir.block_value(ir_block));
        // Build the operation contained in this block
        let op = match self.eir.block_kind(ir_block).unwrap().clone() {
            // Branch to another block in this function, return or throw
            ir::OpKind::Call(ir::CallKind::ControlFlow) => {
                debug_in!(self, "block contains control flow operation");
                let ir_dest = reads[0];

                if self.func.is_return_ir(ir_dest) {
                    // get return values from reads
                    // Returning from this function
                    // TODO: Support multi-value return
                    assert!(num_reads < 3, "unexpected multi-value return");
                    let return_value = if num_reads >= 2 {
                        debug_in!(self, "control flow type: return with value");
                        Some(self.build_value(reads[1])?)
                    } else {
                        debug_in!(self, "control flow type: return void");
                        None
                    };
                    OpKind::Return(Return {
                        loc,
                        value: return_value,
                    })
                } else if self.func.is_throw_ir(ir_dest) {
                    debug_in!(self, "control flow type: throw");
                    // get exception value from reads
                    assert!(
                        num_reads == 4,
                        "expected exception value to throw! reads: {:?}",
                        reads
                    );
                    let error_kind = self.build_value(reads[1])?;
                    let error_class = self.build_value(reads[2])?;
                    let error_reason = self.build_value(reads[3])?;
                    OpKind::Throw(Throw {
                        loc,
                        kind: error_kind,
                        class: error_class,
                        reason: error_reason,
                    })
                } else {
                    debug_in!(self, "control flow type: branch");
                    let block = self.get_block_by_value(ir_dest);
                    let args = self.build_target_block_args(block, &reads[1..]);
                    OpKind::Branch(Br {
                        loc,
                        dest: Branch { block, args },
                    })
                }
            }
            // Call a function
            ir::OpKind::Call(ir::CallKind::Function) => {
                debug_in!(self, "block contains call operation");
                debug_in!(
                    self,
                    "reads = {:?}",
                    reads
                        .iter()
                        .map(|r| (r, self.eir.value_kind(*r)))
                        .collect::<Vec<_>>()
                );
                let ir_callee = reads[0];
                let ir_ok = reads[1];
                let ir_err = reads[2];
                let mut args = Vec::with_capacity(num_reads - 3);
                let is_tail = self.func.is_return_ir(ir_ok) && self.func.is_throw_ir(ir_err);
                debug_in!(self, "is tail call = {}", is_tail);
                for read in reads.iter().skip(3).copied() {
                    let value = self.build_value(read)?;
                    args.push(value);
                }
                let callee = Callee::new(self, ir_callee)?;
                debug_in!(self, "callee = {}", &callee);
                let ok = if self.func.is_return_ir(ir_ok) {
                    CallSuccess::Return
                } else {
                    if let Some(ok_ir_block) = self.eir.value_block(ir_ok) {
                        debug_in!(self, "ok continues to {:?}", ok_ir_block);
                        let ok_block = self.get_block(ok_ir_block);
                        let ok_args = self.build_target_block_args(ok_block, &reads[3..]);
                        CallSuccess::Branch(Branch {
                            block: ok_block,
                            args: ok_args,
                        })
                    } else {
                        panic!(
                            "invalid value used as success continuation ({:?}): {:?}",
                            ir_ok,
                            self.eir.value_kind(ir_ok)
                        );
                    }
                };
                debug_in!(self, "on success = {:?}", ok);
                let err = if self.func.is_throw_ir(ir_err) {
                    CallError::Throw
                } else {
                    if let Some(err_ir_block) = self.eir.value_block(ir_err) {
                        debug_in!(self, "exception continues to {:?}", err_ir_block);
                        let err_block = self.get_block(err_ir_block);
                        CallError::Branch(Branch {
                            block: err_block,
                            args: Default::default(),
                        })
                    } else {
                        panic!(
                            "invalid value used as error continuation ({:?}): {:?}",
                            ir_err,
                            self.eir.value_kind(ir_err)
                        );
                    }
                };
                debug_in!(self, "on error = {:?}", err);
                OpKind::Call(Call {
                    loc,
                    callee,
                    args,
                    is_tail,
                    ok,
                    err,
                })
            }
            // Conditionally branch to another block based on `value`
            ir::OpKind::IfBool => {
                debug_in!(self, "block contains if operation");
                // Map yes/no/other to block refs
                let yes = self.get_block_by_value(reads[0]);
                let no = self.get_block_by_value(reads[1]);
                // The arguments to this operation vary based on whether or
                // not the `else` block is implicitly unreachable. If the #
                // of reads is 3, it is unreachable, if 4, then a block argument
                // is given to fall back to
                let (reads_start, cond, other) = if num_reads > 3 {
                    let other = self.get_block_by_value(reads[2]);
                    let cond = self.build_value(reads[3])?;
                    (3, cond, Some(other))
                } else {
                    let cond = self.build_value(reads[2])?;
                    (2, cond, None)
                };
                debug_in!(self, "yes = {:?}, no = {:?}", yes, no);
                debug_in!(self, "has otherwise branch = {}", other.is_some());

                let remaining_reads = &reads[reads_start..];
                debug_in!(self, "remaining reads = {:?}", remaining_reads);

                OpKind::If(If {
                    loc,
                    cond,
                    yes: Branch {
                        block: yes,
                        args: Default::default(),
                    },
                    no: Branch {
                        block: no,
                        args: Default::default(),
                    },
                    otherwise: other.map(|o| Branch {
                        block: o,
                        args: Default::default(),
                    }),
                })
            }
            // A map insertion/update operation
            // (ok: fn(new_map), err: fn(), map: map, keys: (keys..), values: (value..))
            ir::OpKind::MapPut { ref action, .. } => {
                debug_in!(self, "block contains map put operation");
                let ok = self.get_block_by_value(reads[0]);
                let err = self.get_block_by_value(reads[1]);
                let map = self.build_value(reads[2])?;

                let num_actions = action.len();
                let mut puts = Vec::with_capacity(num_actions);
                let mut idx = 3;
                for action in action.iter() {
                    let key = self.build_value(reads[idx])?;
                    let value = self.build_value(reads[idx + 1])?;
                    debug_in!(self, "put key    = {:?} ({:?})", key, reads[idx]);
                    debug_in!(self, "put value  = {:?} ({:?})", value, reads[idx + 1]);
                    idx += 2;

                    match action {
                        ir::MapPutUpdate::Put => {
                            debug_in!(self, "put action = insert");
                            puts.push(MapPut {
                                action: MapActionType::Insert,
                                key,
                                value,
                            });
                        }
                        ir::MapPutUpdate::Update => {
                            debug_in!(self, "put action = update");
                            puts.push(MapPut {
                                action: MapActionType::Update,
                                key,
                                value,
                            });
                        }
                    }
                }

                OpKind::MapPut(MapPuts {
                    loc,
                    ok,
                    err,
                    map,
                    puts,
                })
            }
            // Construct a binary piece by pice
            // (ok: fn(bin), fail: fn(), head, tail)
            // (ok: fn(bin), fail: fn(), head, tail, size)
            ir::OpKind::BinaryPush {
                specifier: ref spec,
                ..
            } => {
                debug_in!(self, "block contains binary push operation");
                let ok = self.get_block_by_value(reads[0]);
                let err = self.get_block_by_value(reads[1]);
                let head = self.build_value(reads[2])?;
                let tail = self.build_value(reads[3])?;
                let size = if num_reads > 4 {
                    Some(self.build_value(reads[4])?)
                } else {
                    None
                };
                OpKind::BinaryPush(BinaryPush {
                    loc,
                    ok,
                    err,
                    head,
                    tail,
                    size,
                    spec: spec.clone(),
                })
            }
            // Simplified pattern matching on a value; this is a terminator op
            ir::OpKind::Match { mut branches, .. } => {
                debug_in!(self, "block contains match operation");
                let dests = reads[0];
                let num_dests = self.eir.value_list_length(dests);
                let num_branches = branches.len();
                debug_in!(
                    self,
                    "match has {} successors and {} branches",
                    num_dests,
                    num_branches
                );
                debug_assert_eq!(
                    num_dests, num_branches,
                    "number of branches and destination blocks differs"
                );
                let branches = branches
                    .drain(..)
                    .enumerate()
                    .map(|(i, kind)| {
                        debug_in!(self, "branch {} has kind {:?}", i, kind);
                        let block_value = self.eir.value_list_get_n(dests, i).unwrap();
                        let branch_loc = self.value_location(block_value);
                        let block = self.get_block_by_value(block_value);
                        debug_in!(self, "branch {} has dest {:?}", i, block);
                        let args_vl = reads[i + 2];
                        let num_args = self.eir.value_list_length(args_vl);
                        debug_in!(self, "branch {} has {} args", i, num_args);
                        let mut args = Vec::with_capacity(num_args);
                        for n in 0..num_args {
                            args.push(self.eir.value_list_get_n(args_vl, n).unwrap());
                        }
                        Pattern {
                            kind,
                            loc: branch_loc,
                            block,
                            args,
                        }
                    })
                    .collect();

                OpKind::Match(Match {
                    loc,
                    selector: self.build_value(reads[1])?,
                    branches,
                    reads: (&reads[2..]).to_vec(),
                })
            }
            // Requests that a trace be saved
            // Takes a target block as argument
            ir::OpKind::TraceCaptureRaw => {
                debug_in!(self, "block contains trace capture operation");
                let block = self.get_block_by_value(reads[0]);
                OpKind::TraceCapture(TraceCapture {
                    loc,
                    dest: Branch {
                        block,
                        args: Default::default(),
                    },
                })
            }
            // Requests that a trace be constructed for consumption in a `catch`
            // Takes the captured trace reference as argument
            ir::OpKind::TraceConstruct => {
                debug_in!(self, "block contains trace construct operation");
                // We use get_value here because the capture must always be a block argument
                let capture = self.get_value(reads[0]);
                OpKind::TraceConstruct(TraceConstruct { loc, capture })
            }
            // Symbol + per-intrinsic args
            ir::OpKind::Intrinsic(name) => {
                debug_in!(self, "block contains intrinsic {:?}", name);
                OpKind::Intrinsic(Intrinsic {
                    loc,
                    name,
                    args: reads.to_vec(),
                })
            }
            // When encountered, this instruction traps; it also informs the optimizer that we
            // intend to never reach this point during execution
            ir::OpKind::Unreachable => {
                debug_in!(self, "block contains unreachable");
                OpKind::Unreachable(loc)
            }
            invalid => panic!("invalid operation kind: {:?}", invalid),
        };

        OpBuilder::build_void_result(self, op)
    }

    /// This function returns a Value that represents the given IR value
    ///
    /// If the value does not yet have a definition in the current block, then
    /// one is created, by lowering the value via its corresponding primitive operation.
    ///
    /// Panics if the given value is a block reference, or a block argument.
    /// The former are always invalid where values are expected, and the latter should
    /// never happen, as block arguments are defined when the block is created, so
    /// lookups should never fail. If the lookup fails, then we have a compiler bug.
    pub(super) fn build_value(&mut self, ir_value: ir::Value) -> Result<Value> {
        let value = self
            .build_value_opt(ir_value)?
            .expect("expected value, but got pseudo-value");
        Ok(value)
    }

    /// Same as above, but represents value lists as the absence of a value
    pub(super) fn build_value_opt(&mut self, ir_value: ir::Value) -> Result<Option<Value>> {
        match self.eir.value_kind(ir_value) {
            // Always lower constants as fresh values
            ir::ValueKind::Const(c) => {
                let loc = self.value_location(ir_value);
                self.build_constant_value(loc, c).map(|v| Some(v))
            }
            kind => {
                debug_in!(self, "building value {:?} (kind = {:?})", ir_value, kind);
                // If the value has already been lowered, return a reference to it
                if let Some(value) = self.find_value(ir_value) {
                    return Ok(Some(value));
                }
                // Otherwise construct it
                let value_opt = match kind {
                    ir::ValueKind::PrimOp(op) => self.build_primop_value(ir_value, op)?,
                    ir::ValueKind::Block(b) => Some(self.build_closure(ir_value, b)?),
                    ir::ValueKind::Argument(b, i) => {
                        let blk = self.get_block(b);
                        let args = self.block_args(blk);
                        args.get(i).map(|a| a.clone())
                    }
                    _ => unreachable!(),
                };
                Ok(value_opt)
            }
        }
    }

    #[inline]
    fn build_closure(&mut self, ir_value: ir::Value, target: ir::Block) -> Result<Value> {
        debug_in!(
            self,
            "building closure for value {:?} (target block = {:?})",
            ir_value,
            target
        );
        ClosureBuilder::build(self, Some(ir_value), target)
            .and_then(|vopt| vopt.ok_or_else(|| anyhow!("expected constant to have result")))
    }

    /// This function returns a Value that represents the given IR value lowered as a constant
    #[inline]
    fn build_constant_value(&mut self, loc: LocationRef, constant: ir::Const) -> Result<Value> {
        debug_in!(self, "building constant value {:?}", constant);
        let constant = Constant { loc, constant };
        ConstantBuilder::build(self, None, constant)
            .and_then(|vopt| vopt.ok_or_else(|| anyhow!("expected constant to have result")))
    }

    /// This function returns a Value that represents the result of the given IR value
    /// lowered as a primitive operation.
    ///
    /// Values which the primitive operation reads must themselves be lowered if not yet
    /// defined.
    #[inline]
    fn build_primop_value(
        &mut self,
        ir_value: ir::Value,
        primop: ir::PrimOp,
    ) -> Result<Option<Value>> {
        debug_in!(self, "building primop from value {:?}", ir_value);
        let loc = self.value_location(ir_value);
        let primop_kind = self.primop_kind(primop).clone();
        let reads = self.primop_reads(primop).to_vec();
        let num_reads = reads.len();
        let op = match primop_kind {
            // (lhs, rhs)
            ir::PrimOpKind::BinOp(kind) => {
                debug_in!(self, "primop is binary operator");
                debug_in!(self, "operator = {:?}", kind);
                assert_eq!(
                    num_reads, 2,
                    "expected binary operations to have two operands"
                );
                let lhs = self.build_value(reads[0])?;
                let rhs = self.build_value(reads[1])?;
                OpKind::BinOp(BinaryOperator {
                    loc,
                    kind,
                    lhs,
                    rhs,
                })
            }
            // (terms..)
            ir::PrimOpKind::LogicOp(kind) => {
                debug_in!(self, "primop is logical operator");
                debug_in!(self, "operator = {:?}", kind);
                assert_eq!(
                    num_reads, 2,
                    "expected logical operations to have two operands"
                );
                let lhs = self.build_value(reads[0])?;
                let rhs = self.build_value(reads[1])?;
                OpKind::LogicOp(LogicalOperator {
                    loc,
                    kind,
                    lhs,
                    rhs: Some(rhs),
                })
            }
            // (value)
            ir::PrimOpKind::IsType(bt) => {
                debug_in!(self, "primop is type check");
                debug_in!(self, "type = {:?}", bt);
                OpKind::IsType(IsType {
                    loc,
                    value: self.build_value(reads[0])?,
                    expected: bt.clone().into(),
                })
            }
            // (terms..)
            ir::PrimOpKind::Tuple => {
                debug_in!(self, "primop is tuple constructor");
                assert!(
                    num_reads > 1,
                    "expected tuple primop to have at least one operand"
                );
                let mut elements = Vec::with_capacity(num_reads);
                for read in reads {
                    let element = self.build_value(read)?;
                    elements.push(element);
                }
                OpKind::Tuple(Tuple { loc, elements })
            }
            // (head, tail)
            ir::PrimOpKind::ListCell => {
                debug_in!(self, "primop is cons constructor");
                assert_eq!(
                    num_reads, 2,
                    "expected cons cell primop to have two operands"
                );
                let head = self.build_value(reads[0])?;
                let tail = self.build_value(reads[1])?;
                OpKind::Cons(Cons { loc, head, tail })
            }
            // (k1, v1, ...)
            ir::PrimOpKind::Map => {
                debug_in!(self, "primop is map constructor");
                assert!(
                    num_reads >= 2,
                    "expected map primop to have at least two operands"
                );
                assert!(
                    num_reads % 2 == 0,
                    "expected map primop to have a number of operands divisible into pairs"
                );
                let mut elements = Vec::with_capacity(num_reads / 2);
                for chunk in reads.chunks_exact(2) {
                    let key = self.build_value(chunk[0])?;
                    let value = self.build_value(chunk[1])?;
                    elements.push((key, value));
                }
                OpKind::Map(Map { loc, elements })
            }
            ir::PrimOpKind::CaptureFunction => {
                debug_in!(self, "primop is function capture");
                assert_eq!(
                    num_reads, 4,
                    "expected capture function primop to have four operands"
                );
                let callee = Callee::new(self, ir_value)?;
                OpKind::FunctionRef(FunctionRef { loc, callee })
            }
            ir::PrimOpKind::ValueList => {
                debug_in!(self, "value list: {:?} with {} reads", ir_value, num_reads);
                return Ok(None);
            }
            invalid => panic!("unexpected primop kind: {:?}", invalid),
        };
        Ok(Some(OpBuilder::build_one_result(self, ir_value, op)?))
    }

    /// Constructs an argument list for a target block, from the current block
    ///
    /// This function takes care of lowering values used as block arguments, if
    /// those values do not yet have definitions in the current block.
    ///
    /// This function is carefully constructed: EIR does not pass arguments explicitly
    /// to successor blocks, instead they are implicitly passed. MLIR however does not
    /// permit this when blocks are siblings; requiring us to generate blocks with implicit
    /// arguments made explicit.
    ///
    /// In order to correctly construct the list of arguments to a target block, we have to
    /// get the set of parameters for the MLIR block, and then look up the sources of those
    /// values in the current block; these may be block arguments, or results of operations.
    pub(super) fn build_target_block_args(
        &mut self,
        target: Block,
        reads: &[ir::Value],
    ) -> Vec<Value> {
        debug_in!(self, "building target block args for {:?}", target);
        debug_in!(self, "reads:         {:?}", reads);
        // Get the set of parameters the target block expects
        let block_params = self.func.block_params(target);
        debug_in!(self, "params:        {:?}", &block_params);
        // We use the number of reads multiple times, so cache it
        let num_reads = reads.len();
        // We also use the number of implicit params for validation multiple times
        let num_implicits = block_params.iter().filter(|p| p.is_implicit).count();
        debug_in!(self, "num_implicits: {}", num_implicits);
        // Get the parameters which have source values not defined in the target block,
        // in other words, they have definitions that flow through this block or are
        // defined in this block.
        //
        // For each such parameter, map it back to the definition currently in scope, or,
        // if not defined yet, ensure they are defined.
        self.block_args(target)
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, v)| {
                debug_in!(self, "block arg {} is {:?}", i, v);
                let value_data = self.value_data(v);
                if let Some(ir_value) = value_data.ir_value.expand() {
                    debug_in!(self, "block arg has original eir value {:?}", ir_value);
                    if self.is_block_argument(ir_value, target) {
                        let read = i - num_implicits;
                        if read >= num_reads {
                            // This is an implicit argument provided by the operation
                            debug_in!(self, "block arg is implicit");
                            None
                        } else {
                            debug_in!(self, "block arg is explicit argument");
                            // This is an explicit argument corresponding to the operations' reads
                            //
                            // NOTE: When this is None, it was a value list
                            self.build_value_opt(reads[read]).unwrap()
                        }
                    } else {
                        debug_in!(self, "block arg is not a block argument, expected in scope");
                        // This value should have a definition in scope
                        self.build_value_opt(ir_value).unwrap()
                    }
                } else {
                    debug_in!(self, "block has no corresponding eir value");
                    // This value is a block argument with no corresponding EIR value,
                    // pull the value from the operation reads, otherwise it is an implicit
                    // argument provided by the operation
                    assert!(
                        i >= num_implicits,
                        "expected this value to correspond to a read or operation result"
                    );
                    let read = i - num_implicits;
                    if read >= num_reads {
                        // This is an implicit argument provided by the operation
                        debug_in!(self, "block arg is implicit");
                        None
                    } else {
                        debug_in!(self, "block arg is explicit argument");
                        // This is an explicit argument corresponding to the operations' reads
                        self.build_value_opt(reads[read]).unwrap()
                    }
                }
            })
            .collect()
    }
}

/// Maintains metadata about the current position of the builder
#[derive(Default, Clone, Copy)]
struct Position {
    block: Option<Block>,
}
impl Position {
    // Create a new Position representing the end of the given block
    fn at(block: Block) -> Self {
        Self { block: Some(block) }
    }

    // Return the current position
    fn current(&self) -> Option<Block> {
        self.block
    }

    // Reset the position to None
    fn reset(&mut self) -> Option<Block> {
        self.block.take()
    }

    // This is used to set a position that the underlying builder is already in,
    // careless use of this will result in the position of the builder here and
    // in MLIR falling out of sync
    unsafe fn set(&mut self, block: Block) -> Option<Block> {
        self.block.replace(block)
    }

    // Position the builder at the end of the provided block
    fn position_at_end(&mut self, builder: ModuleBuilderRef, block: Block, block_ref: BlockRef) {
        unsafe {
            MLIRBlockPositionAtEnd(builder, block_ref);
        }
        self.block = Some(block);
    }

    // When true, the builder is not positioned in a block
    fn is_default(&self) -> bool {
        self.block.is_none()
    }
}

/// Shared helper to map an EIR value to its constant kind
pub(super) fn value_to_const_kind<'f>(
    function: &'f ir::Function,
    val: ir::Value,
) -> &'f ir::ConstKind {
    let constant = function.value_const(val).expect("expected constant value");
    function.const_kind(constant)
}

/// Shared helper to construct a Param from an EIR value
pub(super) fn block_arg_to_param(f: &ir::Function, arg: ir::Value, is_implicit: bool) -> Param {
    let span = value_location(f, arg);

    Param {
        ty: Type::Term,
        span,
        is_implicit,
    }
}

/// Shared helper to construct a Span from the location info of an EIR value
pub(super) fn value_location(f: &ir::Function, value: ir::Value) -> Span {
    f.value_locations(value)
        .map(|locs| {
            let span = locs[0];
            let start = span.start().to_usize() as u32;
            let end = span.end().to_usize() as u32;
            Span::new(start, end)
        })
        .unwrap_or_else(Span::default)
}

pub(super) fn get_block_argument(block_ref: BlockRef, index: usize) -> ValueRef {
    let value_ref = unsafe { MLIRGetBlockArgument(block_ref, index as libc::c_uint) };
    assert!(!value_ref.is_null());
    value_ref
}
