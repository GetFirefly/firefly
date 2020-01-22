mod function;
pub use self::function::*;

use std::collections::HashSet;
use std::ffi::CString;
use std::ptr;

use anyhow::anyhow;

use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};

use log::debug;

use libeir_intern::{Ident, Symbol};
use libeir_ir as ir;
use libeir_ir::{AtomTerm, AtomicTerm, ConstKind, FunctionIdent};
use libeir_lowerutils::{FunctionData, LowerData};
use libeir_util_datastructures::pooled_entity_set::BoundEntitySet;

use crate::Result;

use liblumen_session::Options;

use super::block::{Block, BlockData};
use super::ffi::*;
use super::ops::builders::{BranchBuilder, CallBuilder, ConstantBuilder};
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
        let name = f.ident();

        debug!("{}: building..", &name);

        debug!("{}: performing lowering analysis..", &name);
        let analysis = libeir_lowerutils::analyze(f);
        let loc = Span::from(f.span());

        // Gather atoms in this function and add them to the atom table for this module
        debug!("{}: gathering atoms for atom table", &name);
        for val in f.iter_constants() {
            if let ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(s))) = value_to_const_kind(f, *val) {
                debug!("{}: found atom: {:?}", &name, *s);
                self.builder.atoms_mut().insert(*s);
            }
        }

        let root_block = f.block_entry();
        for (entry_block, data) in analysis.functions.iter() {
            let entry_block = *entry_block;
            if entry_block == root_block {
                self.with_scope(name.clone(), loc, f, &analysis, data, options)
                    .and_then(|scope| scope.build())?;
            } else {
                let arity = f.block_args(entry_block).len() - 2;
                let name = FunctionIdent {
                    module: name.module.clone(),
                    name: Ident::from_str(&format!("{}-fun-{}", name.name, arity)),
                    arity,
                };
                self.with_scope(name.clone(), loc, f, &analysis, data, options)
                    .and_then(|scope| scope.build())?;
            }
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
        let ret = data
            .ret
            .expect("expected function to have return continuation");
        let esc = data
            .thr
            .expect("expected function to have escape continuation");

        // Construct signature
        let mut signature = Signature::new(CallConv::Fast);
        let entry_args = eir.block_args(data.entry);
        {
            signature.params.reserve(entry_args.len() - 2);
            for arg in entry_args.iter().skip(2).copied() {
                signature
                    .params
                    .push(block_arg_to_param(eir, arg, /* is_implicit */ false));
            }
            signature.returns.push(Type::Term);
        }

        // Construct the parameter value metadata
        let entry_params = entry_args
            .iter()
            .skip(2)
            .copied()
            .enumerate()
            .map(|(i, v)| (signature.params[i].clone(), Some(v)))
            .collect::<Vec<_>>();

        // Create function
        let mut func = Function::with_name_signature(name, signature);
        let (mlir, entry_ref) = func.build(self.builder)?;

        // Mirror the entry block for our init block
        let init_block =
            func.new_block_with_params(Some(data.entry), entry_ref, entry_params.as_slice());
        // Initialize ret/esc continuations
        let ret = func.set_return_continuation(ret, init_block);
        let esc = func.set_escape_continuation(esc, init_block);

        Ok(ScopedFunctionBuilder {
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
        })
    }
}

/// This builder type is essentially a sub-type of FunctionBuilder, and handles
/// lowering EIR functions with a closed scope. In other words, EIR functions can
/// contain multiple functions due to closures; a "scoped" function is just a
/// function that contains no closures. A FunctionBuilder lowers each scoped
/// function using the ScopedFunctionBuilder.
pub struct ScopedFunctionBuilder<'f, 'o> {
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

    /// Prints debugging messages with the name of the function being built
    #[cfg(debug_assertions)]
    pub(super) fn debug(&self, message: &str) {
        debug!("{}: {}", self.func.name(), message);
    }

    #[cfg(not(debug_assertions))]
    pub(super) fn debug(&self, _message: &str) {}
}

// EIR function metadata helpers
impl<'f, 'o> ScopedFunctionBuilder<'f, 'o> {
    /// Gets the set of EIR values that are live at the given block
    #[inline]
    pub fn live_at(&self, block: Block) -> BoundEntitySet<'f, ir::Value> {
        let ir_block = self.func.block_to_ir_block(block).unwrap();
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
        let block = self.current_block();
        self.func.value_mapping[ir_value][block].into()
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
                "the given value is not a known block: {:?}",
                ir_value
            ))
    }

    /// Gets the block corresponding to the given EIR block
    ///
    /// Panics if the block doesn't exist
    #[inline]
    pub fn get_block(&self, ir_block: ir::Block) -> Block {
        self.func.block_mapping[ir_block]
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
    pub fn build(mut self) -> Result<()> {
        self.debug("building..");

        // NOTE: We start out in the init block
        let init_block = self.current_block();

        // TODO: Generate stack frame
        // My current thinking is that we keep the stack opaque in the runtime,
        // and let the generated code write directly to it to avoid calling into
        // the runtime to record frames. We will eventually use stack maps, and
        // in that case the runtime stack stuff that exists will mostly go away.
        self.debug("building stack frame for function in init block");

        // If this is a closure, extract the environment
        // A closure will have more than 1 live value, otherwise it is a regular function
        let live = self.live_at(init_block);
        self.debug(&format!(
            "found {} live values in the entry block",
            live.size()
        ));
        if live.size() > 0 {
            todo!("closure env unpacking");
        }

        // Clone the init block, the clone will be used like the EIR entry block
        // The init block, meanwhile, is used to set up any frame layout/init required
        self.debug("creating entry block..");
        let entry_block = self.clone_block();

        // Forward all of the init block arguments to the entry block
        self.debug("forwarding init block to entry block");
        let init_block_args = self.block_args(self.current_block());
        self.insert_branch(entry_block, init_block_args.as_slice())?;

        self.debug("switching to entry block");
        self.position_at_end(entry_block);

        let root_block = self.eir.block_entry();

        // Make sure all blocks are created first
        let mut blocks = Vec::with_capacity(self.data.scope.len() - 1);
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

        Ok(())
    }

    // Prepares an MLIR block for lowering from an EIR block
    //
    // This function creates a new block, and handles promoting
    // implicit arguments to explicit arguments in the translation
    // from EIR.
    fn prepare_block(&mut self, ir_block: ir::Block) -> Result<(ir::Block, Block)> {
        // Construct parameter list for the block
        // The result is implicit arguments followed by explicit arguments
        let block_args = self.eir.block_args(ir_block);
        let implicits = {
            let mut implicits = Vec::new();
            let live_at = self.analysis.live.live_at(ir_block);
            // Prefer to avoid extra work if we can
            if live_at.size() > 0 {
                // Build a temporary set to keep lookups efficient
                let mut explicit: HashSet<ir::Value> = HashSet::with_capacity(block_args.len());
                explicit.extend(block_args);
                // Check each value in the block live_at set to see which
                // are in the argument list. Those that are not, are implicit
                // arguments, coming from a dominating block.
                for live in live_at.iter() {
                    // If not an explicit argument, and live within the block, add it as implicit
                    if !explicit.contains(&live) && self.analysis.live.is_live_in(ir_block, live) {
                        implicits.push((
                            block_arg_to_param(self.eir, live, /* is_implicit */ true),
                            Some(live),
                        ));
                    }
                }
            }
            implicits
        };

        // Build the final parameter list
        let mut params = Vec::with_capacity(block_args.len() + implicits.len());
        params.extend(implicits);
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

    /// Inserts a branch operation at the current point, with the given destination and arguments
    fn insert_branch(&mut self, block: Block, args: &[Value]) -> Result<()> {
        BranchBuilder::build(
            self,
            Branch {
                block,
                args: args.to_vec(),
            },
        )
        .map(|_| ())
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

        let block = self
            .func
            .new_block_with_params(ir_block, block_ref, param_info);

        Ok(block)
    }

    /// Clones the current block by:
    ///
    /// 1. Creating a new block after the current block
    /// 2. Copying the number and type of block arguments from the current block
    ///
    /// The builder will be positioned at the end of the original block once complete
    fn clone_block(&mut self) -> Block {
        let current_block = self.current_block();

        // Map source parameter metadata to original EIR values
        let block_data = self.func.block_data(current_block);
        let mut params = Vec::with_capacity(block_data.params.len());
        for (i, &param) in block_data.params.iter().enumerate() {
            let value = block_data.param_values[i];
            let ir_value = self.func.value_to_ir_value(value);
            params.push((param, ir_value));
        }

        // Create a new block with the same params as the current one
        self.create_block(None, params.as_slice()).unwrap()
    }

    /// Positions the builder at the end of the given block
    fn position_at_end(&mut self, block: Block) {
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
        // Switch to the block
        self.position_at_end(block);
        // Get the set of values this block reads in its body
        let reads = self.eir.block_reads(ir_block);
        let num_reads = reads.len();
        // Build the operation contained in this block
        let op = match self.eir.block_kind(ir_block).unwrap().clone() {
            // Branch to another block in this function, return or throw
            ir::OpKind::Call(ir::CallKind::ControlFlow) => {
                let ir_dest = reads[0];
                let dest = self.get_value(ir_dest);

                if self.func.is_return(dest) {
                    // get return values from reads
                    // Returning from this function
                    // TODO: Support multi-value return
                    assert!(num_reads < 3, "unexpected multi-value return");
                    let return_value = if num_reads >= 2 {
                        Some(self.build_value(reads[1])?)
                    } else {
                        None
                    };
                    OpKind::Return(return_value)
                } else if self.func.is_throw(dest) {
                    // get exception value from reads
                    assert!(num_reads == 2, "expected exception value to throw");
                    OpKind::Throw(self.build_value(reads[1])?)
                } else {
                    let block = self.get_block_by_value(ir_dest);
                    let args = self.build_target_block_args(block, &reads[1..]);
                    OpKind::Branch(Branch { block, args })
                }
            }
            // Call a function
            ir::OpKind::Call(ir::CallKind::Function) => {
                let ir_callee = reads[0];
                let mut is_tail = false;
                let mut args = Vec::with_capacity(num_reads - 1);
                let is_tail = num_reads >= 3
                    && self.func.is_return_ir(reads[1])
                    && self.func.is_throw_ir(reads[2]);
                let start_read = if is_tail { 3 } else { 1 };
                for read in reads.iter().skip(start_read).copied() {
                    let value = self.build_value(read)?;
                    args.push(value);
                }
                let callee = Callee::new(self, ir_callee)?;
                OpKind::Call(Call {
                    callee,
                    args,
                    is_tail,
                })
            }
            // Conditionally branch to another block based on `value`
            ir::OpKind::IfBool => {
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

                let remaining_reads = &reads[reads_start..];
                let yes_args = self.build_target_block_args(yes, remaining_reads);
                let no_args = self.build_target_block_args(no, remaining_reads);
                let other_args = other.map(|b| self.build_target_block_args(b, remaining_reads));

                OpKind::If(If {
                    cond,
                    yes: Branch {
                        block: yes,
                        args: yes_args,
                    },
                    no: Branch {
                        block: no,
                        args: no_args,
                    },
                    otherwise: other.map(|o| Branch {
                        block: o,
                        args: other_args.unwrap_or_default(),
                    }),
                })
            }
            // A map insertion/update operation
            // (ok: fn(new_map), err: fn(), map: map, keys: (keys..), values: (value..))
            ir::OpKind::MapPut { ref action, .. } => {
                let ok = self.get_block_by_value(reads[0]);
                let err = self.get_block_by_value(reads[1]);
                let map = self.build_value(reads[2])?;

                let num_actions = action.len();
                let mut puts = Vec::with_capacity(num_actions);
                let mut idx = 3;
                for action in action.iter() {
                    debug!("  {} key = {:?}", idx - 3, reads[idx]);
                    debug!("  {} value = {:?}", idx - 3, reads[idx + 1]);
                    let key = self.build_value(reads[idx])?;
                    let value = self.build_value(reads[idx + 1])?;
                    idx += 2;

                    match action {
                        ir::MapPutUpdate::Put => {
                            puts.push(MapPut {
                                action: MapActionType::Insert,
                                key,
                                value,
                            });
                        }
                        ir::MapPutUpdate::Update => {
                            puts.push(MapPut {
                                action: MapActionType::Update,
                                key,
                                value,
                            });
                        }
                    }
                }

                OpKind::MapPut(MapPuts { ok, err, map, puts })
            }
            // Construct a binary piece by pice
            // (ok: fn(bin), fail: fn(), head, tail)
            // (ok: fn(bin), fail: fn(), head, tail, size)
            ir::OpKind::BinaryPush {
                specifier: ref spec,
                ..
            } => {
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
                let dests = reads[0];
                let num_dests = self.eir.value_list_length(dests);
                debug_assert_eq!(
                    num_dests,
                    branches.len(),
                    "number of branches and destination blocks differs"
                );
                let branches = branches
                    .drain(..)
                    .enumerate()
                    .map(|(i, kind)| {
                        let block_value = self.eir.value_list_get_n(dests, i).unwrap();
                        let block = self.get_block_by_value(block_value);
                        let args_vl = reads[i + 2];
                        let num_args = self.eir.value_list_length(args_vl);
                        let mut args = Vec::with_capacity(num_args);
                        for n in 0..num_args {
                            args.push(self.eir.value_list_get_n(args_vl, n).unwrap());
                        }
                        Pattern { kind, block, args }
                    })
                    .collect();

                OpKind::Match(Match {
                    selector: self.build_value(reads[1])?,
                    branches,
                    reads: (&reads[1..]).to_vec(),
                })
            }
            // Requests that a trace be saved
            // Takes a target block as argument
            ir::OpKind::TraceCaptureRaw => {
                let block = self.get_block_by_value(reads[0]);
                let args = self.build_target_block_args(block, &reads[1..]);
                OpKind::TraceCapture(Branch { block, args })
            }
            // Requests that a trace be constructed for consumption in a `catch`
            // Takes the captured trace reference as argument
            ir::OpKind::TraceConstruct => {
                // We use get_value here because the capture must always be a block argument
                let capture = self.get_value(reads[0]);
                OpKind::TraceConstruct(capture)
            }
            // Symbol + per-intrinsic args
            ir::OpKind::Intrinsic(name) => OpKind::Intrinsic(Intrinsic {
                name,
                args: reads.to_vec(),
            }),
            // When encountered, this instruction traps; it also informs the optimizer that we
            // intend to never reach this point during execution
            ir::OpKind::Unreachable => OpKind::Unreachable,
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
        if let Some(value) = self.find_value(ir_value) {
            return Ok(value);
        }
        match self.eir.value_kind(ir_value) {
            ir::ValueKind::Const(c) => self.build_constant_value(ir_value, c),
            ir::ValueKind::PrimOp(op) => self.build_primop_value(ir_value, op),
            ir::ValueKind::Block(b) => {
                unreachable!("block {:?} used as a value ({:?})", b, ir_value)
            }
            ir::ValueKind::Argument(b, i) => {
                unreachable!("block argument {:?}:{} should already be defined", b, i)
            }
        }
    }

    /// This function returns a Value that represents the given IR value lowered as a constant
    #[inline]
    fn build_constant_value(&mut self, ir_value: ir::Value, constant: ir::Const) -> Result<Value> {
        self.debug(&format!(
            "building constant value {:?} ({:?})",
            ir_value, constant
        ));
        ConstantBuilder::build(self, Some(ir_value), constant)
            .and_then(|vopt| vopt.ok_or_else(|| anyhow!("expected constant to have result")))
    }

    /// This function returns a Value that represents the result of the given IR value
    /// lowered as a primitive operation.
    ///
    /// Values which the primitive operation reads must themselves be lowered if not yet
    /// defined.
    #[inline]
    fn build_primop_value(&mut self, ir_value: ir::Value, primop: ir::PrimOp) -> Result<Value> {
        self.debug(&format!(
            "building primop for value {:?}: {:?}",
            ir_value, primop
        ));
        let primop_kind = self.primop_kind(primop).clone();
        let reads = self.primop_reads(primop).to_vec();
        let num_reads = reads.len();
        let op = match primop_kind {
            // (lhs, rhs)
            ir::PrimOpKind::BinOp(kind) => {
                assert_eq!(
                    num_reads, 2,
                    "expected binary operations to have two operands"
                );
                let lhs = self.build_value(reads[0])?;
                let rhs = self.build_value(reads[1])?;
                OpKind::BinOp(BinaryOperator { kind, lhs, rhs })
            }
            // (terms..)
            ir::PrimOpKind::LogicOp(kind) => {
                assert_eq!(
                    num_reads, 2,
                    "expected logical operations to have two operands"
                );
                let lhs = self.build_value(reads[0])?;
                let rhs = self.build_value(reads[1])?;
                OpKind::LogicOp(LogicalOperator {
                    kind,
                    lhs,
                    rhs: Some(rhs),
                })
            }
            // (value)
            ir::PrimOpKind::IsType(bt) => OpKind::IsType {
                value: self.build_value(reads[0])?,
                expected: bt.clone().into(),
            },
            // (terms..)
            ir::PrimOpKind::Tuple => {
                assert_eq!(
                    num_reads, 1,
                    "expected tuple primop to have at least one operand"
                );
                let mut elements = Vec::with_capacity(num_reads);
                for read in reads {
                    let element = self.build_value(read)?;
                    elements.push(element);
                }
                OpKind::Tuple(elements)
            }
            // (head, tail)
            ir::PrimOpKind::ListCell => {
                assert_eq!(
                    num_reads, 2,
                    "expected cons cell primop to have two operands"
                );
                let head = self.build_value(reads[0])?;
                let tail = self.build_value(reads[1])?;
                OpKind::Cons(head, tail)
            }
            // (k1, v1, ...)
            ir::PrimOpKind::Map => {
                assert!(
                    num_reads >= 2,
                    "expected map primop to have at least two operands"
                );
                assert!(
                    num_reads % 2 == 0,
                    "expected map primop to have a number of operands divisible into pairs"
                );
                let mut items = Vec::with_capacity(num_reads / 2);
                for chunk in reads.chunks_exact(2) {
                    let key = self.build_value(chunk[0])?;
                    let value = self.build_value(chunk[1])?;
                    items.push((key, value));
                }
                OpKind::Map(items)
            }
            ir::PrimOpKind::CaptureFunction => {
                assert_eq!(
                    num_reads, 4,
                    "expected capture function primop to have four operands"
                );
                let callee = Callee::new(self, ir_value)?;
                OpKind::FunctionRef(callee)
            }
            invalid => panic!("unexpected primop kind: {:?}", invalid),
        };
        OpBuilder::build_one_result(self, ir_value, op)
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
        // Get the set of parameters the target block expects
        let block_params = self.func.block_params(target);
        // We use the number of reads multiple times, so cache it
        let num_reads = reads.len();
        // We also use the number of implicit params for validation multiple times
        let num_implicits = block_params.iter().filter(|p| p.is_implicit).count();
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
                let value_data = self.value_data(v);
                if let Some(ir_value) = value_data.ir_value.expand() {
                    if self.is_block_argument(ir_value, target) {
                        // For EIR values that are defined as a block argument of the target,
                        // we need to select the read which corresponds to the parameter. If
                        // we don't have enough reads, then it is an argument we expect to be
                        // provided by an operation, so we ignore them. If this value corresponds
                        // to an implicit, we have a compiler bug, as we would not expect the
                        // source value to be a block argument of the target
                        assert!(
                            i >= num_implicits,
                            "expected this value to correspond to a read or operation result"
                        );
                        let read = i - num_implicits;
                        if read >= num_reads {
                            // This is an implicit argument provided by the operation
                            None
                        } else {
                            // This is an explicit argument corresponding to the operations' reads
                            Some(self.build_value(reads[read]).unwrap())
                        }
                    } else {
                        // This value should have a definition in scope
                        Some(self.build_value(ir_value).unwrap())
                    }
                } else {
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
                        None
                    } else {
                        // This is an explicit argument corresponding to the operations' reads
                        Some(self.build_value(reads[read]).unwrap())
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
