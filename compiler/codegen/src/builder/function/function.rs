#![allow(unused)]

use std::ffi::CString;
use std::fmt;
use std::ptr;

use anyhow::anyhow;

use cranelift_entity::packed_option::PackedOption;
use cranelift_entity::{PrimaryMap, SecondaryMap};

use log::debug;

use libeir_ir as ir;
use libeir_ir::FunctionIdent;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_mlir::ir::{BlockRef, FunctionOpRef, ValueRef};
use liblumen_util::diagnostics::SourceSpan;

use crate::builder::block::*;
use crate::builder::ffi::{self, Span, Type};
use crate::builder::value::*;
use crate::builder::ModuleBuilder;
use crate::Result;

pub struct Function {
    // MFA of this function
    name: ir::FunctionIdent,
    pub(super) span: SourceSpan,
    // Signature of this function
    signature: Signature,
    // Primary storage for blocks
    pub(super) blocks: PrimaryMap<Block, BlockData>,
    // Mapping of EIR blocks to MLIR blocks
    pub(super) block_mapping: SecondaryMap<ir::Block, Block>,
    // Primary storage for MLIR values -> value metadata
    pub(super) values: PrimaryMap<Value, ValueData>,
    // Mapping of EIR values to MLIR values by block
    //
    // Access is of the form "given value V from EIR, give me the MLIR value
    // which corresponds to that value in block B". This question is necessary
    // since MLIR requires values to be passed explicitly to successor blocks as
    // arguments, while EIR does not. Each time a value is used as an implicit argument
    // in EIR, we extend the block argument list in MLIR with a new value, and track
    // the connection between that new argument and its original definition. When we
    // encounter a usage of a value in EIR, we can do the following:
    //
    // - Find the value in the current block which represents the EIR value
    // - Efficiently look up the value metadata from the definition
    // - Perform data flow analysis and type synthesis
    pub(super) value_mapping: SecondaryMap<ir::Value, SecondaryMap<Block, PackedOption<Value>>>,

    ret: Value,
    ret_ir: ir::Value,
    esc: Value,
    esc_ir: ir::Value,
}
impl Function {
    /// Initializes this function definition with the given name and signature
    ///
    /// NOTE: This does not construct the function in MLIR
    pub fn with_name_signature(
        span: SourceSpan,
        name: FunctionIdent,
        signature: Signature,
    ) -> Self {
        Self {
            name,
            span,
            signature,
            blocks: PrimaryMap::new(),
            block_mapping: SecondaryMap::new(),
            values: PrimaryMap::new(),
            value_mapping: SecondaryMap::with_default(SecondaryMap::new()),
            ret: Default::default(),
            ret_ir: Default::default(),
            esc: Default::default(),
            esc_ir: Default::default(),
        }
    }

    /// Sets this functions return continuation
    pub fn set_return_continuation(&mut self, ret: ir::Value, entry: Block) -> Value {
        let value = self.values.push(ValueData {
            value: Default::default(),
            ir_value: PackedOption::from(ret),
            data: ValueDef::Return,
        });
        self.value_mapping[ret][entry] = PackedOption::from(value);
        self.ret_ir = ret;
        self.ret = value;
        self.ret
    }

    /// Sets this functions escape continuation
    pub fn set_escape_continuation(&mut self, esc: ir::Value, entry: Block) -> Value {
        let value = self.values.push(ValueData {
            value: Default::default(),
            ir_value: PackedOption::from(esc),
            data: ValueDef::Escape,
        });
        self.value_mapping[esc][entry] = PackedOption::from(value);
        self.esc_ir = esc;
        self.esc = value;
        self.esc
    }

    /// Constructs this function definition in MLIR
    pub fn build<'m>(&self, builder: &mut ModuleBuilder<'m>) -> Result<(FunctionOpRef, BlockRef)> {
        // Create function in MLIR
        let args = self.signature.params.as_slice();
        let argc = args.len();
        let returns = self.signature.returns.as_slice();

        debug!(
            "creating function {} with {} parameters and return types {:?}",
            &self.name, argc, returns,
        );

        let c_name = CString::new(self.name.to_string()).unwrap();
        // TODO: support multi-return
        let result_type = returns.get(0).unwrap_or(&Type::None);
        let loc = unsafe {
            let sl = builder
                .location(self.span.start().index())
                .expect("expected source location for function");
            ffi::MLIRCreateLocation(builder.as_ref(), sl)
        };

        let ffi::FunctionDeclResult {
            function,
            entry_block,
        } = unsafe {
            ffi::MLIRCreateFunction(
                builder.as_ref(),
                loc,
                c_name.as_ptr(),
                args.as_ptr(),
                argc as libc::c_uint,
                result_type,
            )
        };
        if function.is_null() {
            return Err(anyhow!(
                "failed to create function {}",
                c_name.to_string_lossy()
            ));
        }

        // Register function symbol globally
        builder.symbols_mut().insert(FunctionSymbol {
            module: self.name.module.name.as_usize(),
            function: self.name.name.name.as_usize(),
            arity: self.name.arity as u8,
            ptr: ptr::null(),
        });

        Ok((function, entry_block))
    }

    /// Returns a reference to the current functions' identifier
    #[inline]
    pub fn name(&self) -> &ir::FunctionIdent {
        &self.name
    }

    /// Returns true if the given value is this functions' return continuation
    #[inline]
    pub fn is_return(&self, value: Value, block: Block) -> bool {
        if let Some(expected) = self.value_mapping[self.ret_ir][block].expand() {
            value == expected
        } else {
            false
        }
    }

    /// Returns true if the given EIR value is this functions' return continuation
    #[inline]
    pub fn is_return_ir(&self, value: ir::Value) -> bool {
        self.ret_ir == value
    }

    /// Returns true if the given value is this functions' escape continuation
    #[inline]
    pub fn is_throw(&self, value: Value, block: Block) -> bool {
        if let Some(expected) = self.value_mapping[self.esc_ir][block].expand() {
            value == expected
        } else {
            false
        }
    }

    /// Returns true if the given EIR value is this functions' escape continuation
    #[inline]
    pub fn is_throw_ir(&self, value: ir::Value) -> bool {
        self.esc_ir == value
    }

    /// Creates a new block in this function, optionally tracked against the given EIR block
    ///
    /// The block is intantiated with an empty parameter list
    pub fn new_block(&mut self, ir_block: Option<ir::Block>, block_ref: BlockRef) -> Block {
        self.new_block_with_params(ir_block, block_ref, &[])
    }

    /// Same as `new_block`, but takes parameter metadata which specifies which arguments the
    /// block takes, and how they map back to the original EIR
    pub fn new_block_with_params(
        &mut self,
        ir_block: Option<ir::Block>,
        block_ref: BlockRef,
        parameters: &[(Param, Option<ir::Value>)],
    ) -> Block {
        // Gather just the parameters
        let params = parameters
            .iter()
            .map(|(p, _)| p.clone())
            .collect::<Vec<_>>();
        // Gather the MLIR values and match them up with their original EIR values
        let mut param_value_refs = Vec::with_capacity(params.len());
        for (i, (_p, pv)) in parameters.iter().enumerate() {
            let value_ref = super::get_block_argument(block_ref, i);
            let ir_value = pv.clone();
            param_value_refs.push((i, ir_value, value_ref));
        }

        // Construct the block so we get a block handle to use
        let block = self.blocks.push(BlockData {
            block: block_ref,
            ir_block: PackedOption::from(ir_block),
            params,
            param_values: Vec::new(),
        });

        // Update the block data with the mapped parameter values
        let mut param_values = Vec::with_capacity(param_value_refs.len());
        for (i, ir_value, value_ref) in param_value_refs.drain(..) {
            let value = self.new_value(block, ir_value, value_ref, ValueDef::Param(i));
            if let Some(irv) = ir_value {
                // Register EIR value in this block
                self.value_mapping[irv][block] = PackedOption::from(value);
            }
            param_values.push(value);
        }
        self.blocks[block].param_values = param_values;

        // Map the original EIR block to the MLIR block
        if let Some(irb) = ir_block {
            self.block_mapping[irb] = block;
        }
        block
    }

    /// Returns the block data for the given block
    #[inline]
    pub fn block_data(&self, block: Block) -> &BlockData {
        assert!(
            self.blocks.is_valid(block),
            "invalid block requested {:?}",
            block
        );
        self.blocks.get(block).unwrap()
    }

    /// Returns the block data for the given block mutably
    #[inline]
    pub fn block_data_mut(&mut self, block: Block) -> &mut BlockData {
        assert!(
            self.blocks.is_valid(block),
            "invalid block requested {:?}",
            block
        );
        self.blocks.get_mut(block).unwrap()
    }

    /// Returns the arity of the given block
    pub fn block_arity(&self, block: Block) -> usize {
        self.block_data(block).params.len()
    }

    /// Maps the given block back to its original EIR block
    pub fn block_to_ir_block(&self, block: Block) -> Option<ir::Block> {
        self.block_data(block).ir_block.into()
    }

    /// Maps the given block to its underlying MLIR block reference
    pub fn block_to_block_ref(&self, block: Block) -> BlockRef {
        self.block_data(block).block
    }

    /// Returns the parameter info for the given block
    pub fn block_params(&self, block: Block) -> Vec<Param> {
        self.block_data(block).params.clone()
    }

    /// Registers a new value in the given block, optionally tracking an EIR value
    pub fn new_value(
        &mut self,
        block: Block,
        ir_value: Option<libeir_ir::Value>,
        value_ref: ValueRef,
        data: ValueDef,
    ) -> Value {
        let value = self.values.push(ValueData {
            value: value_ref,
            ir_value: PackedOption::from(ir_value),
            data,
        });
        if let Some(irv) = ir_value {
            self.value_mapping[irv][block] = PackedOption::from(value);
        }
        value
    }

    /// Returns the value metadata for the given value
    #[inline]
    pub fn value_data(&self, value: Value) -> &ValueData {
        self.values.get(value).unwrap()
    }

    /// Returns the value metadata for the given value, mutably
    #[inline]
    pub fn value_data_mut(&mut self, value: Value) -> &mut ValueData {
        self.values.get_mut(value).unwrap()
    }

    /// Returns the EIR value the given value corresponds to
    pub fn value_to_ir_value(&self, value: Value) -> Option<ir::Value> {
        self.value_data(value).ir_value.into()
    }

    /// Returns the MLIR value reference for the given value
    pub fn value_to_value_ref(&self, value: Value) -> ValueRef {
        self.value_data(value).value
    }

    /// Returns the ValueDef structure for the given value
    pub fn value_to_value_def(&self, value: Value) -> ValueDef {
        self.value_data(value).data
    }
}

/// Function signature.
///
/// The function signature describes the types of formal parameters and return values along with
/// other details that are needed to call a function correctly.
///
/// A signature can optionally include ISA-specific ABI information which specifies exactly how
/// arguments and return values are passed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// The arguments passed to the function.
    pub params: Vec<Param>,
    /// Values returned from the function.
    pub returns: Vec<Type>,
    /// Calling convention.
    pub call_conv: CallConv,
}
impl Signature {
    pub fn new(cc: CallConv) -> Self {
        Self {
            params: Vec::new(),
            returns: Vec::new(),
            call_conv: cc,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Param {
    pub ty: Type,
    pub span: Span,
    // True when this parameter is an implicit argument in EIR
    pub is_implicit: bool,
}
impl fmt::Debug for Param {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_implicit {
            write!(f, "implicit({:?})", self.ty)
        } else {
            write!(f, "{:?}", self.ty)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CallConv {
    Fast,
    Rust,
    C,
}
impl Default for CallConv {
    fn default() -> Self {
        Self::Fast
    }
}
